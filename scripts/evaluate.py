#!/usr/bin/env python3
"""
Evaluation script for LayoutLM-based information extraction experiments
"""
import argparse
import json
from pathlib import Path

from src.data.layoutlm_datasets import create_data_loader, get_dataset_loader, safe_dataset_length
from src.models.layoutlm_models import get_model
from src.training.layoutlm_trainer import create_trainer
from src.utils import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate LayoutLM model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save model predictions"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Update config with command line arguments
    config["output_dir"] = args.output_dir
    config["save_predictions"] = args.save_predictions

    # Setup output directory
    output_dir = Path(args.output_dir) / f"{config['experiment_name']}_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(
        str(output_dir / "logs"),
        f"{config['experiment_name']}_eval"
    )

    logger.info(f"Starting evaluation: {config['experiment_name']}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset split: {args.dataset_split}")

    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset_loader = get_dataset_loader(config)
        train_dataset, test_dataset, val_dataset = dataset_loader.load_data()

        # Select dataset split
        if args.dataset_split == "train":
            eval_dataset = train_dataset
        elif args.dataset_split == "validation":
            eval_dataset = val_dataset if val_dataset else test_dataset
        else:  # test
            eval_dataset = test_dataset

        # Get label information
        label_list = dataset_loader.get_label_list()
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}

        logger.info(f"Dataset: {config['dataset']['name']}")
        
        # Check if we're in streaming mode
        streaming_mode = config["dataset"].get("streaming", False)
        logger.info(f"Streaming mode: {streaming_mode}")
        
        # Use safe dataset length function with dataset loader
        eval_split_name = args.dataset_split if args.dataset_split != "validation" else "validation"
        logger.info(f"Evaluation samples: {safe_dataset_length(eval_dataset, streaming_mode, dataset_loader, eval_split_name)}")
        
        logger.info(f"Labels: {label_list}")

        # Create data loader
        logger.info("Creating data loader...")
        
        if streaming_mode:
            # Streaming mode: use LayoutLMStreamingDataset
            logger.info("Using streaming data loader...")
            eval_dataloader = create_data_loader(
                config=config,
                is_training=False,
                dataset_loader=dataset_loader,
                hf_dataset=eval_dataset,
                tokenizer=dataset_loader.tokenizer,
                label2id=label2id,
                split_name=eval_split_name
            )
        else:
            # Memory mode: create examples first, then use LayoutLMDataset
            logger.info("Creating examples...")
            eval_examples = dataset_loader.create_examples(eval_dataset)
            
            logger.info("Using memory-based data loader...")
            eval_dataloader = create_data_loader(
                examples=eval_examples,
                tokenizer=dataset_loader.tokenizer,
                label2id=label2id,
                config=config,
                is_training=False
            )

        # Initialize model
        logger.info("Initializing model...")
        # Update model config with correct number of labels
        config["model"]["config"]["num_labels"] = len(label_list)
        model = get_model(config)

        logger.info(f"Model: {model.__class__.__name__}")

        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(
            model=model,
            train_dataloader=None,  # Not needed for evaluation
            eval_dataloader=eval_dataloader,
            config=config,
            label_list=label_list,
            id2label=id2label
        )

        # Load model
        logger.info(f"Loading model from {args.model_path}")
        trainer.load_model(Path(args.model_path))

        # Run evaluation
        logger.info("Running evaluation...")
        eval_results = trainer.evaluate()

        logger.info("Evaluation Results:")
        for metric, value in eval_results.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Results saved to: {results_file}")

        # Save predictions if requested
        if args.save_predictions:
            logger.info("Saving predictions...")
            predictions_file = output_dir / "predictions.json"

            # Get predictions
            model.eval()
            all_predictions = []
            all_labels = []

            import torch
            with torch.no_grad():
                for i, batch in enumerate(eval_dataloader):
                    batch = {k: v.to(trainer.device) for k, v in batch.items()}
                    outputs = model(**batch)

                    predictions = torch.argmax(outputs["logits"], dim=-1)

                    # Convert to labels
                    batch_predictions = []
                    batch_labels = []

                    for j in range(predictions.shape[0]):
                        pred_labels = []
                        true_labels = []

                        for k in range(predictions.shape[1]):
                            if batch["attention_mask"][j][k] == 1 and batch["labels"][j][k] != -100:
                                pred_labels.append(id2label[predictions[j][k].item()])
                                true_labels.append(id2label[batch["labels"][j][k].item()])

                        batch_predictions.append(pred_labels)
                        batch_labels.append(true_labels)

                    all_predictions.extend(batch_predictions)
                    all_labels.extend(batch_labels)

            predictions_data = {
                "predictions": all_predictions,
                "labels": all_labels,
                "label_list": label_list
            }

            with open(predictions_file, "w") as f:
                json.dump(predictions_data, f, indent=2)

            logger.info(f"Predictions saved to: {predictions_file}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
