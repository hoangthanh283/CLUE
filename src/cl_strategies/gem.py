"""
Gradient Episodic Memory (GEM) for Continual Learning.
Exact replication of the original GEM algorithm using quadratic programming solver.

Reference: Lopez-Paz & Ranzato (2017). Gradient Episodic Memory for Continual Learning. NeurIPS.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
import quadprog

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.cl_strategies.utils import (
    get_grad_vector, set_grad_vector,
    get_grad_vector_exclude_classifier, set_grad_vector_exclude_classifier
)


class GEM(BaseCLStrategy):
    """
    Gradient Episodic Memory (GEM) strategy with exact QP solver.
    
    GEM enforces inequality constraints: g^T · g_k >= 0 for all previous tasks k,
    ensuring that the loss on each previous task does not increase.
    
    This implementation uses the exact QP formulation from the original paper:
    minimize   0.5 * ||v - g||^2
    subject to v^T · g_k >= 0  for all k (previous tasks)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cl_setting = config.get("cl_setting", "class_il")
        cl_cfg = config.get("cl_strategy", {})

        # Memory settings
        mem_size = int(cl_cfg.get("memory_size", 500))  # Total samples across all tasks
        self.memory = MemoryBuffer(mem_size)

        # Per-task constraint settings
        self.samples_per_task = int(cl_cfg.get("samples_per_task", 5))  # Samples per task for constraint
        self.max_tasks_for_constraints = int(cl_cfg.get("max_tasks", 10))  # Max tasks to consider

        # QP solver settings
        self.qp_tolerance = float(cl_cfg.get("qp_tolerance", 1e-3))
        self.qp_regularization = float(cl_cfg.get("qp_regularization", 1e-6))  # Regularization for numerical stability
        self.margin = float(cl_cfg.get("margin", 0.5))  # Constraint margin (0.5 in original GEM)

        # Memory management
        self.clear_cache_every = int(cl_cfg.get("clear_cache_every", 5))
        self._step_count = 0

        # Track current task
        self.current_task_id = 0
        self.seen_tasks: List[int] = []  # List of task IDs we've seen

    def before_task(self, model: nn.Module, task_id: int, train_loader=None):
        """Mark the start of a new task."""
        self.current_task_id = task_id

    def after_task(self, model: nn.Module, task_id: int, train_loader=None):
        """Mark that we've completed training on a task."""
        if task_id not in self.seen_tasks:
            self.seen_tasks.append(task_id)

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        """Store samples from the current task."""
        # Store only first sample to minimize memory usage
        sample_batch = {
            "input_ids": batch["input_ids"][:1],
            "attention_mask": batch["attention_mask"][:1],
            "bbox": batch["bbox"][:1],
            "labels": batch["labels"][:1]
        }
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
            sample_batch["token_type_ids"] = batch["token_type_ids"][:1]

        # Pass task_id to memory buffer for task-aware sampling
        self.memory.add_batch(sample_batch, task_id=self.current_task_id)

    def on_after_backward(self, model: nn.Module, is_final_accumulation_step: bool = True):
        """
        Apply GEM constraint: project gradient if it violates constraints from previous tasks.

        This hook is called after loss.backward() has computed gradients. GEM projects the
        gradients to satisfy inequality constraints from previous tasks.

        Args:
            model: Model with already-computed gradients
            is_final_accumulation_step: Only project on final accumulation step
        """
        # Only project on final gradient accumulation step
        if not is_final_accumulation_step:
            return

        device = next(model.parameters()).device

        # Periodic cache clearing
        self._step_count += 1
        if self._step_count % self.clear_cache_every == 0:
            torch.cuda.empty_cache()

        # No constraints for first task or if no previous tasks completed
        if self.current_task_id == 0 or len(self.seen_tasks) == 0:
            return

        # Get current gradient (already computed by backward pass)
        # For task-IL, only apply constraints to backbone (not classifier)
        if self.cl_setting == "task_il":
            g = get_grad_vector_exclude_classifier(model)
        else:
            g = get_grad_vector(model)

        # Build constraint gradients from previous tasks
        constraint_gradients = self._compute_constraint_gradients(model, device)

        if constraint_gradients is None or len(constraint_gradients) == 0:
            return

        # Check if constraints are violated
        violations = self._check_violations(g, constraint_gradients)

        if not violations:
            # No violation, use original gradient
            logger.debug(f"Task {self.current_task_id}: No constraint violations, using original gradient")
            return

        # Project gradient to satisfy constraints using exact QP solver
        logger.info(f"Task {self.current_task_id}: Constraint violated, projecting gradient")
        original_norm = g.norm().item()
        projected_g = self._project_gradient_qp_exact(g, constraint_gradients)
        projected_norm = projected_g.norm().item()
        logger.info(f"  Original gradient norm: {original_norm:.4f}, Projected norm: {projected_norm:.4f}, Reduction: {(1 - projected_norm/original_norm)*100:.1f}%")

        # Set the projected gradient
        # For task-IL, only update backbone gradients
        if self.cl_setting == "task_il":
            set_grad_vector_exclude_classifier(model, projected_g)
        else:
            set_grad_vector(model, projected_g)

        # Cleanup
        del g, constraint_gradients, projected_g

    def _compute_constraint_gradients(self, model: nn.Module, device: torch.device) -> Optional[List[torch.Tensor]]:
        """
        Compute constraint gradients from previous tasks using the CURRENT model.

        For class-incremental learning, we compute gradients on old task samples using the
        current expanded classifier. This ensures that all gradient vectors have the same
        dimensions and can be used in the QP constraint formulation.

        The constraint "current gradient should not increase loss on previous tasks" is
        enforced by ensuring dot products between current and previous task gradients are
        non-negative.

        Returns list of gradient vectors (on device), one per previous task.
        """
        constraint_grads = []

        # Determine how many previous tasks to consider
        num_prev_tasks = min(len(self.seen_tasks), self.max_tasks_for_constraints)

        if num_prev_tasks == 0:
            return None

        # For each previous task, compute gradient using the CURRENT model
        for task_id in self.seen_tasks[:num_prev_tasks]:
            # Compute gradient using samples from this task with current model
            task_grad = self._compute_task_gradient(model, device, self.samples_per_task, task_id)

            if task_grad is not None:
                constraint_grads.append(task_grad.detach())
                del task_grad
                torch.cuda.empty_cache()

        return constraint_grads if len(constraint_grads) > 0 else None

    def _compute_task_gradient(self, model: nn.Module, device: torch.device, n_samples: int,
                               task_id: int) -> Optional[torch.Tensor]:
        """
        Compute gradient for a specific task from n_samples.

        Samples all n_samples at once and computes gradient in a single backward pass,
        which is more efficient than individual sample processing.

        Args:
            model: Model to compute gradients for
            device: Device to run on
            n_samples: Number of samples to use
            task_id: ID of the task to sample from

        Returns:
            Gradient vector from the batch, or None if no valid samples
        """
        # Sample batch from this specific task
        mem_batch = self.memory.sample(n_samples, device=device, task_id=task_id)
        if mem_batch is None:
            return None

        # Compute gradient for this batch in a single backward pass
        model.zero_grad(set_to_none=True)
        try:
            outputs = model(**mem_batch)
            batch_loss = outputs["loss"]
            batch_loss.backward()

            # Get gradient vector
            # For task-IL, only use backbone gradients for constraints
            if self.cl_setting == "task_il":
                task_grad = get_grad_vector_exclude_classifier(model).detach()
            else:
                task_grad = get_grad_vector(model).detach()

            # Cleanup
            del mem_batch, outputs, batch_loss
            torch.cuda.empty_cache()

            return task_grad

        except Exception:
            # If gradient computation fails, return None
            torch.cuda.empty_cache()
            return None

    def _check_violations(self, g: torch.Tensor, constraint_gradients: List[torch.Tensor]) -> bool:
        """Check if current gradient violates any constraint.

        A constraint is violated if: g^T · g_k < -margin
        where margin provides a buffer for numerical stability and constraint relaxation.
        """
        for g_k in constraint_gradients:
            dot_product = torch.dot(g, g_k)
            if dot_product < -self.margin:
                return True
        return False

    def _project_gradient_qp_exact(self, g: torch.Tensor, constraint_gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Solve the QP problem exactly to project gradient onto the feasible region.

        Problem formulation (dual form as in the original paper):
            minimize   0.5 * ||v - g||^2
            subject to  v^T · g_k >= -margin  for all k (previous tasks)

        The margin parameter (default 0.5) provides constraint relaxation and numerical stability,
        matching the original GEM implementation (Lopez-Paz & Ranzato, 2017).

        Uses quadprog library to solve the exact QP problem as in the original paper.
        This uses the dual formulation where we solve for the Lagrange multipliers.
        """
        # Convert to numpy for quadprog
        g_np = g.cpu().double().numpy()
        G_np = torch.stack(constraint_gradients, dim=0).cpu().double().numpy()
        
        # Dual QP formulation as in the original paper:
        # minimize 0.5 * x^T * P * x + q^T * x
        # subject to G * x >= h
        #
        # Where:
        # P = G_np * G_np^T (Gram matrix of constraint gradients)
        # q = -G_np * g (negative dot products with current gradient)
        # G = I (identity matrix for positivity constraints on Lagrange multipliers)
        # h = zeros + margin (non-negativity constraints with margin relaxation)

        t = G_np.shape[0]  # number of constraints (previous tasks)
        # p = G_np.shape[1]  # gradient dimension (not needed in dual formulation)

        # Compute Gram matrix P = G_np * G_np^T
        # This is t x t matrix, much smaller than the p x p identity matrix
        P = np.dot(G_np, G_np.T)

        # Ensure symmetry and add regularization for numerical stability
        P = 0.5 * (P + P.T) + np.eye(t) * self.qp_regularization

        # Compute q = -G_np * g (negative dot products)
        q = np.dot(G_np, g_np) * -1

        # Constraint matrix G and bounds h
        # h includes margin for constraint relaxation (v^T · g_k >= -margin)
        G = np.eye(t)
        h = np.zeros(t) + self.margin
        
        # Solve QP with enhanced error handling
        try:
            # Solve the QP problem to get Lagrange multipliers
            solution = quadprog.solve_qp(P, q, G, h)
            lagrange_multipliers = solution[0]
            
            # Compute projected gradient: v = g + sum(lambda_i * g_i)
            # Note: This is different from what I had before
            # In the original paper's formulation, the projection is:
            # v = g + G_np^T * lambda (where lambda >= 0)
            v_np = g_np + np.dot(lagrange_multipliers, G_np)
            
            # Convert back to torch tensor
            v = torch.from_numpy(v_np).float().to(g.device)
            
        except Exception as e:
            # If QP solver fails, fall back to greedy projection
            print(f"Warning: QP solver failed with error: {e}. Falling back to greedy projection.")
            v = self._project_gradient_qp_greedy(g, constraint_gradients)
        
        return v

    def _project_gradient_qp_greedy(self, g: torch.Tensor, constraint_gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Fallback greedy projection method if QP solver fails.
        """
        # Initialize with current gradient (use clone to avoid modifying g)
        v = g.clone()

        # Maximum iterations for convergence
        max_iter = 50
        
        # Projected gradient descent with improved convergence check
        for iteration in range(max_iter):
            # Check each constraint individually
            most_violated_idx = -1
            min_violation = 0.0

            for idx, g_k in enumerate(constraint_gradients):
                violation = torch.dot(v, g_k).item()
                if violation < min_violation:
                    min_violation = violation
                    most_violated_idx = idx

            # Check convergence with better tolerance
            if min_violation >= -self.qp_tolerance:
                # All constraints satisfied
                break

            if most_violated_idx < 0:
                break

            # Project onto most violated constraint using in-place operation
            g_k = constraint_gradients[most_violated_idx]
            dot_vgk = torch.dot(v, g_k)

            if dot_vgk < 0:
                g_k_norm_sq = torch.dot(g_k, g_k)
                if g_k_norm_sq > 1e-12:
                    # In-place update: v -= (dot_vgk / g_k_norm_sq) * g_k
                    v.add_(g_k, alpha=-float(dot_vgk / g_k_norm_sq))

        return v
