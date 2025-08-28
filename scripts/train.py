#!/usr/bin/env python3
"""
Training script for LayoutLM-based information extraction experiments
"""
import argparse
from pathlib import Path

from src.data.layoutlm_datasets import create_data_loader, get_dataset_loader
from src.models.layoutlm_models import get_model
from src.training.layoutlm_trainer import create_trainer
from src.utils import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Train LayoutLM model for information extraction"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Update config with command line arguments
    config["output_dir"] = args.output_dir
    if args.resume_from_checkpoint:
        config["resume_from_checkpoint"] = args.resume_from_checkpoint

    # Setup output directory
    output_dir = Path(args.output_dir) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)

    # Setup logging
    logger = setup_logging(
        str(output_dir / "logs"),
        config["experiment_name"]
    )

    logger.info(f"Starting experiment: {config['experiment_name']}")
    logger.info(f"Configuration: {config}")

    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset_loader = get_dataset_loader(config)
        train_dataset, test_dataset, val_dataset = dataset_loader.load_data()

        # Get label information
        label_list = dataset_loader.get_label_list()
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}

        logger.info(f"Dataset: {config['dataset']['name']}")
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Labels: {label_list}")

        # Create examples
        logger.info("Creating examples...")
        train_examples = dataset_loader.create_examples(train_dataset)
        eval_examples = dataset_loader.create_examples(val_dataset if val_dataset else test_dataset)

        # Create data loaders
        logger.info("Creating data loaders...")
        train_dataloader = create_data_loader(
            train_examples, dataset_loader.tokenizer, label2id, config, is_training=True
        )
        eval_dataloader = create_data_loader(
            eval_examples, dataset_loader.tokenizer, label2id, config, is_training=False
        )

        # Initialize model
        logger.info("Initializing model...")
        # Update model config with correct number of labels
        config["model"]["config"]["num_labels"] = len(label_list)
        model = get_model(config)

        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config,
            label_list=label_list,
            id2label=id2label
        )

        # Resume from checkpoint if specified
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.load_model(Path(args.resume_from_checkpoint))

        # Train model
        logger.info("Starting training...")
        training_results = trainer.train()

        logger.info(f"Training completed! Best metric: {training_results['best_metric']:.4f}")

        # Final evaluation on test set
        if test_dataset and val_dataset:  # Only if we have separate test set
            logger.info("Running final evaluation on test set...")
            test_examples = dataset_loader.create_examples(test_dataset)
            test_dataloader = create_data_loader(
                test_examples, dataset_loader.tokenizer, label2id, config, is_training=False
            )

            # Temporarily replace eval dataloader
            original_eval_dataloader = trainer.eval_dataloader
            trainer.eval_dataloader = test_dataloader

            test_results = trainer.evaluate()
            logger.info(f"Test results: {test_results}")

            # Restore original eval dataloader
            trainer.eval_dataloader = original_eval_dataloader

        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
