"""
Copyright 2025 AI21 Labs

Test for AmalgamDataset that analyzes data distribution from real DAPO_TATQA files.

This test loads the actual DAPO_TATQA dataset files from GCS and analyzes
the data distribution per batch from different data sources.
"""

import os

# Import the dataset classes
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add the project root to Python path (where verl package is located)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from verl.utils.dataset.amalgam_dataset import AmalgamDataset  # noqa: E402
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402


def analyze_batch_distribution(dataloader, num_batches=50):
    """Analyze the distribution of data sources across batches."""
    batch_distributions = []
    total_distribution = Counter()

    iterator = iter(dataloader)

    for batch_idx in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            print(f"Dataset exhausted after {batch_idx} batches")
            break

        # Count data sources in this batch
        batch_counter = Counter()
        for data_source in batch["data_source"]:
            batch_counter[data_source] += 1
            total_distribution[data_source] += 1

        batch_distributions.append(dict(batch_counter))

        if batch_idx < 10:  # Print first 10 batches for debugging
            print(f"Batch {batch_idx}: {dict(batch_counter)}")

    return batch_distributions, dict(total_distribution)


@pytest.mark.skip(reason="Test excluded - dataset distribution analysis")
def test_real_amalgam_dapo_tatqa_distribution():
    """
    Test that loads the real DAPO_TATQA dataset files and analyzes data distribution.
    This test checks the data distribution per batch from different data sources.
    """

    # Configuration matching the shell script
    config = {
        "train_files": "gs://ai21-algo-studio-research/verl_amalgam_experiments/DAPO_TATQA_train.json",
        "val_files": "gs://ai21-algo-studio-research/verl_amalgam_experiments/DAPO_TATQA_val.json",
        "train_batch_size": 32,
        "val_batch_size": 16,
        "max_prompt_length": 2048,
        "max_response_length": 4096,
        "prompt_key": "prompt",
        "return_raw_chat": True,
        "truncation": "left",
        "shuffle": True,
        "filter_overlong_prompts": True,
        "train_datasources_finite": False,
        "val_datasources_finite": True,
        "cache_dir": "/dev/shm/.gs_cache",
        "amalgam_inter_ds_seed": 42,
        "amalgam_intra_ds_seed": 42,
        "seed": 42,
    }

    print("Loading real DAPO_TATQA dataset files:")
    print(f"  Train: {config['train_files']}")
    print(f"  Val: {config['val_files']}")

    # Initialize tokenizer (using Qwen2.5-7B-Instruct to match the training config)
    try:
        print("Loading tokenizer: Qwen/Qwen2.5-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Unable to load Qwen tokenizer, falling back to gpt2: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e2:
            print(f"Unable to load any tokenizer for testing: {e2}")
            return

    try:
        print("\nCreating AmalgamDataset for training...")
        # Create a config object similar to what RayPPOTrainer would pass
        train_config = SimpleNamespace(
            prompt_key=config["prompt_key"],
            max_prompt_length=config["max_prompt_length"],
            filter_overlong_prompts=config["filter_overlong_prompts"],
            cache_dir=config["cache_dir"],
            return_raw_chat=config["return_raw_chat"],
            truncation=config["truncation"],
            shuffle=config["shuffle"],
            amalgam_inter_ds_seed=config["amalgam_inter_ds_seed"],
            amalgam_intra_ds_seed=config["amalgam_intra_ds_seed"],
            train_datasources_finite=config["train_datasources_finite"],
            val_datasources_finite=config["val_datasources_finite"],
            seed=config["seed"],
        )
        # Add get method to mimic OmegaConf behavior
        train_config.get = lambda key, default=None: getattr(train_config, key, default)

        # Create AmalgamDataset for training (finite=False)
        train_dataset = AmalgamDataset(
            data_files=config["train_files"],
            tokenizer=tokenizer,
            processor=None,  # Not used but kept for interface compatibility
            config=train_config,
            is_val_dataset=False,  # Training dataset
        )

        # Print dataset statistics
        print("\n" + "=" * 60)
        print("TRAINING DATASET STATISTICS")
        print("=" * 60)
        try:
            stats = train_dataset.get_dataset_stats()
            print(stats)
        except Exception as e:
            print(f"Could not get dataset stats: {e}")

        # Create DataLoader with the same batch size as training config
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config["train_batch_size"],
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Analyze batch distribution
        print("\n" + "=" * 60)
        print("TRAINING BATCH DISTRIBUTION ANALYSIS")
        print("=" * 60)

        batch_distributions, total_distribution = analyze_batch_distribution(
            train_dataloader,
            num_batches=50,  # Analyze more batches for better statistics
        )

        # Print overall statistics
        print(f"\nTotal samples processed: {sum(total_distribution.values())}")
        print("Overall distribution:")
        for data_source, count in sorted(total_distribution.items()):
            percentage = (count / sum(total_distribution.values())) * 100
            print(f"  {data_source}: {count} samples ({percentage:.1f}%)")

        # Analyze variance across batches
        print("\n" + "=" * 60)
        print("TRAINING BATCH-TO-BATCH VARIANCE ANALYSIS")
        print("=" * 60)

        # Calculate per-source variance across batches
        data_sources = list(total_distribution.keys())
        source_variances = {}

        for source in data_sources:
            batch_counts = [batch_dist.get(source, 0) for batch_dist in batch_distributions]
            if len(batch_counts) > 1:
                variance = np.var(batch_counts)
                mean = np.mean(batch_counts)
                cv = np.sqrt(variance) / mean if mean > 0 else 0  # Coefficient of variation
                source_variances[source] = {"mean": mean, "variance": variance, "std": np.sqrt(variance), "cv": cv}

        print("Per-source statistics across batches:")
        for source, stats in source_variances.items():
            print(f"  {source}:")
            print(f"    Mean per batch: {stats['mean']:.2f}")
            print(f"    Std deviation: {stats['std']:.2f}")
            print(f"    Coefficient of variation: {stats['cv']:.2f}")

        print(f"\nNumber of datasets in training amalgam: {len(train_dataset.datasets)}")
        print("Dataset names:")
        for name in train_dataset.datasets.keys():
            print(f"  - {name}")

    except Exception as e:
        print(f"Error creating or testing training AmalgamDataset: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Now test validation dataset
    try:
        print("\n" + "=" * 60)
        print("VALIDATION DATASET ANALYSIS")
        print("=" * 60)

        print("Creating AmalgamDataset for validation...")
        val_config = SimpleNamespace(
            prompt_key=config["prompt_key"],
            max_prompt_length=config["max_prompt_length"],
            filter_overlong_prompts=config["filter_overlong_prompts"],
            cache_dir=config["cache_dir"],
            return_raw_chat=config["return_raw_chat"],
            truncation=config["truncation"],
            shuffle=False,  # No shuffle for validation
            amalgam_inter_ds_seed=config["amalgam_inter_ds_seed"],
            amalgam_intra_ds_seed=config["amalgam_intra_ds_seed"],
            train_datasources_finite=config["train_datasources_finite"],
            val_datasources_finite=config["val_datasources_finite"],
            seed=config["seed"],
        )
        # Add get method to mimic OmegaConf behavior
        val_config.get = lambda key, default=None: getattr(val_config, key, default)

        val_dataset = AmalgamDataset(
            data_files=config["val_files"],
            tokenizer=tokenizer,
            processor=None,  # Not used but kept for interface compatibility
            config=val_config,
            is_val_dataset=True,  # Validation dataset should be finite
        )

        # Print validation dataset statistics
        try:
            stats = val_dataset.get_dataset_stats()
            print(stats)
        except Exception as e:
            print(f"Could not get validation dataset stats: {e}")

        # Validation dataset should have a length since it's finite
        if hasattr(val_dataset, "__len__"):
            dataset_length = len(val_dataset)
            print(f"\nValidation dataset length: {dataset_length}")
        else:
            print("\nValidation dataset does not have __len__ method")

        # Create validation DataLoader
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config["val_batch_size"],
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn,
        )

        print("\nAnalyzing validation batch distribution...")
        val_batch_distributions, val_total_distribution = analyze_batch_distribution(
            val_dataloader,
            num_batches=20,  # Fewer batches for validation
        )

        print(f"\nValidation total samples processed: {sum(val_total_distribution.values())}")
        print("Validation distribution:")
        for data_source, count in sorted(val_total_distribution.items()):
            percentage = (count / sum(val_total_distribution.values())) * 100
            print(f"  {data_source}: {count} samples ({percentage:.1f}%)")

        print(f"\nNumber of datasets in validation amalgam: {len(val_dataset.datasets)}")
        print("Validation dataset names:")
        for name in val_dataset.datasets.keys():
            print(f"  - {name}")

    except Exception as e:
        print(f"Error creating or testing validation AmalgamDataset: {e}")
        import traceback

        traceback.print_exc()
        # Don't re-raise for validation errors - training analysis is more important


if __name__ == "__main__":
    # Run the test directly
    # Skip test if running in build environment

    print("Running Real DAPO_TATQA Amalgam Dataset Distribution Analysis...")
    try:
        test_real_amalgam_dapo_tatqa_distribution()
        print("\n\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback

        traceback.print_exc()
