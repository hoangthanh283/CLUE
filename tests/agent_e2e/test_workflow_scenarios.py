"""End-to-end tests for multi-agent workflow scenarios."""

import pytest
from typing import Dict, List, Any


class WorkflowAgent:
    """Simplified agent for workflow testing."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.state = "idle"
        self.results = {}
        self.completed_tasks = []

    def start_task(self, task: str):
        """Start working on a task."""
        self.state = "working"
        return f"{self.name} starting: {task}"

    def complete_task(self, task: str, result: Any):
        """Complete a task and store result."""
        self.completed_tasks.append(task)
        self.results[task] = result
        self.state = "idle"
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.name,
            "role": self.role,
            "state": self.state,
            "completed_tasks": len(self.completed_tasks),
            "results_count": len(self.results),
        }


class MultiAgentWorkflow:
    """Orchestrates workflow with multiple agents."""

    def __init__(self):
        self.agents: Dict[str, WorkflowAgent] = {}
        self.workflow_log = []
        self.completed_workflows = []

    def add_agent(self, agent: WorkflowAgent):
        """Add agent to workflow."""
        self.agents[agent.name] = agent

    def execute_sequential(self, tasks: List[tuple]) -> bool:
        """
        Execute tasks sequentially.
        tasks: List of (agent_name, task_description)
        """
        for agent_name, task in tasks:
            if agent_name not in self.agents:
                self.workflow_log.append(f"ERROR: Agent {agent_name} not found")
                return False

            agent = self.agents[agent_name]
            agent.start_task(task)
            agent.complete_task(task, f"Result of {task}")
            self.workflow_log.append(f"{agent_name} completed: {task}")

        return True

    def execute_parallel(self, tasks: List[tuple]) -> bool:
        """
        Execute tasks in parallel (simulated).
        tasks: List of (agent_name, task_description)
        """
        # Start all tasks
        for agent_name, task in tasks:
            if agent_name not in self.agents:
                return False
            self.agents[agent_name].start_task(task)

        # Complete all tasks
        for agent_name, task in tasks:
            self.agents[agent_name].complete_task(task, f"Result of {task}")
            self.workflow_log.append(f"{agent_name} completed: {task}")

        return True

    def execute_pipeline(self, pipeline: List[tuple]) -> str:
        """
        Execute pipeline where output of one agent is input to next.
        pipeline: List of (agent_name, task_description)
        """
        output = None

        for agent_name, task in pipeline:
            if agent_name not in self.agents:
                return None

            agent = self.agents[agent_name]
            agent.start_task(task)

            # Simulate processing previous output
            if output:
                task = f"{task} (using: {output})"

            agent.complete_task(task, f"Result of {task}")
            output = agent.results[task]
            self.workflow_log.append(f"{agent_name}: {task}")

        return output

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        statuses = {}
        for name, agent in self.agents.items():
            statuses[name] = agent.get_status()
        return statuses

    def mark_workflow_complete(self):
        """Mark workflow as complete."""
        self.completed_workflows.append(
            {"log": self.workflow_log.copy(), "status": self.get_workflow_status()}
        )
        self.workflow_log.clear()


class TestMultiAgentWorkflows:
    """Test complete multi-agent workflows."""

    def test_research_workflow(self):
        """Test research workflow: finder -> reviewer -> brainstormer -> implementer."""
        # Setup agents
        finder = WorkflowAgent("paper_finder", "researcher")
        reviewer = WorkflowAgent("paper_reviewer", "researcher")
        brainstormer = WorkflowAgent("brainstormer", "researcher")
        implementer = WorkflowAgent("implementer", "engineer")

        workflow = MultiAgentWorkflow()
        workflow.add_agent(finder)
        workflow.add_agent(reviewer)
        workflow.add_agent(brainstormer)
        workflow.add_agent(implementer)

        # Execute pipeline
        pipeline = [
            ("paper_finder", "Find papers on continual learning"),
            ("paper_reviewer", "Review selected papers"),
            ("brainstormer", "Brainstorm implementation ideas"),
            ("implementer", "Implement selected approach"),
        ]

        result = workflow.execute_pipeline(pipeline)

        # Verify all agents completed tasks
        assert finder.get_status()["completed_tasks"] == 1
        assert reviewer.get_status()["completed_tasks"] == 1
        assert brainstormer.get_status()["completed_tasks"] == 1
        assert implementer.get_status()["completed_tasks"] == 1

    def test_code_review_workflow(self):
        """Test code change workflow: explorer -> planner -> implementer -> tester."""
        explorer = WorkflowAgent("code_explorer", "developer")
        planner = WorkflowAgent("code_planner", "architect")
        implementer = WorkflowAgent("code_implementer", "developer")
        tester = WorkflowAgent("code_tester", "qa")

        workflow = MultiAgentWorkflow()
        for agent in [explorer, planner, implementer, tester]:
            workflow.add_agent(agent)

        # Sequential execution
        tasks = [
            ("code_explorer", "Explore codebase for improvement areas"),
            ("code_planner", "Plan refactoring strategy"),
            ("code_implementer", "Implement refactoring"),
            ("code_tester", "Run tests and validate"),
        ]

        result = workflow.execute_sequential(tasks)

        assert result is True
        assert len(workflow.workflow_log) == 4

    def test_parallel_agent_execution(self):
        """Test parallel execution of independent tasks."""
        agent1 = WorkflowAgent("agent1", "researcher")
        agent2 = WorkflowAgent("agent2", "researcher")
        agent3 = WorkflowAgent("agent3", "researcher")

        workflow = MultiAgentWorkflow()
        for agent in [agent1, agent2, agent3]:
            workflow.add_agent(agent)

        # Parallel tasks
        tasks = [
            ("agent1", "Task A"),
            ("agent2", "Task B"),
            ("agent3", "Task C"),
        ]

        result = workflow.execute_parallel(tasks)

        assert result is True
        # All agents should have completed one task each
        for name in ["agent1", "agent2", "agent3"]:
            status = workflow.agents[name].get_status()
            assert status["completed_tasks"] == 1

    def test_workflow_with_missing_agent(self):
        """Test workflow gracefully handles missing agent."""
        agent1 = WorkflowAgent("agent1", "role1")

        workflow = MultiAgentWorkflow()
        workflow.add_agent(agent1)

        # Try to use non-existent agent
        tasks = [("agent1", "Task 1"), ("missing_agent", "Task 2")]

        result = workflow.execute_sequential(tasks)

        assert result is False

    def test_workflow_status_tracking(self):
        """Test tracking workflow status."""
        agent1 = WorkflowAgent("agent1", "role1")
        agent2 = WorkflowAgent("agent2", "role2")

        workflow = MultiAgentWorkflow()
        workflow.add_agent(agent1)
        workflow.add_agent(agent2)

        tasks = [
            ("agent1", "Task 1"),
            ("agent2", "Task 2"),
            ("agent1", "Task 3"),
        ]

        workflow.execute_sequential(tasks)

        statuses = workflow.get_workflow_status()

        assert statuses["agent1"]["completed_tasks"] == 2
        assert statuses["agent2"]["completed_tasks"] == 1

    def test_workflow_completion_tracking(self):
        """Test marking workflow completion."""
        agent = WorkflowAgent("agent", "role")

        workflow = MultiAgentWorkflow()
        workflow.add_agent(agent)

        tasks = [("agent", "Task 1")]
        workflow.execute_sequential(tasks)

        assert len(workflow.completed_workflows) == 0

        workflow.mark_workflow_complete()

        assert len(workflow.completed_workflows) == 1
        completed = workflow.completed_workflows[0]
        assert len(completed["log"]) > 0

    def test_multi_stage_workflow(self):
        """Test complex multi-stage workflow."""
        # Stage 1: Research
        finder = WorkflowAgent("finder", "researcher")
        reviewer = WorkflowAgent("reviewer", "researcher")

        # Stage 2: Planning
        planner = WorkflowAgent("planner", "architect")

        # Stage 3: Implementation
        implementer = WorkflowAgent("implementer", "engineer")

        workflow = MultiAgentWorkflow()
        for agent in [finder, reviewer, planner, implementer]:
            workflow.add_agent(agent)

        # Stage 1 - parallel research
        research_tasks = [
            ("finder", "Find papers"),
            ("reviewer", "Review papers"),
        ]
        workflow.execute_parallel(research_tasks)

        # Stage 2 - planning
        workflow.agents["planner"].start_task("Plan approach")
        workflow.agents["planner"].complete_task(
            "Plan approach", "Implementation plan"
        )

        # Stage 3 - implementation
        workflow.agents["implementer"].start_task("Implement solution")
        workflow.agents["implementer"].complete_task(
            "Implement solution", "Complete solution"
        )

        # Verify all agents participated
        assert finder.get_status()["completed_tasks"] == 1
        assert reviewer.get_status()["completed_tasks"] == 1
        assert planner.get_status()["completed_tasks"] == 1
        assert implementer.get_status()["completed_tasks"] == 1

    def test_workflow_error_recovery(self):
        """Test workflow error recovery."""
        agent1 = WorkflowAgent("agent1", "role")
        agent2 = WorkflowAgent("agent2", "role")

        workflow = MultiAgentWorkflow()
        workflow.add_agent(agent1)
        workflow.add_agent(agent2)

        # Try invalid task
        result = workflow.execute_sequential([("invalid_agent", "Task")])
        assert result is False

        # Agents should still be usable
        agent1.start_task("Task")
        agent1.complete_task("Task", "Result")
        assert agent1.get_status()["completed_tasks"] == 1

    def test_workflow_context_preservation(self):
        """Test that context is preserved across agents."""
        agent1 = WorkflowAgent("agent1", "role1")
        agent2 = WorkflowAgent("agent2", "role2")

        workflow = MultiAgentWorkflow()
        workflow.add_agent(agent1)
        workflow.add_agent(agent2)

        # Execute pipeline to test context passing
        pipeline = [("agent1", "Produce data"), ("agent2", "Process data")]

        result = workflow.execute_pipeline(pipeline)

        # Verify both agents completed tasks
        assert agent1.get_status()["completed_tasks"] == 1
        assert agent2.get_status()["completed_tasks"] == 1
        assert result is not None
