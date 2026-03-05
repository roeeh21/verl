# Copyright 2025 AI21 Labs
import os

import pandas as pd

# The jsonl files in this dir are expected to have the following fields:
# - messages: list of messages
# - reward_metrics: dict of reward metrics
#   Each item is a dict with keys:
#   "metric_name" (str), "metric_args" (dict), "metric_weight" (float)

source_json_dir = "gs://ai21-publishing-studio-datasets/foundation_models/jlm_pairs_ifeval_emails/online_rl_drafting"
target_parquet_dir = "gs://ai21-algo-studio-research/noamg/verl/online_rl_drafting"

splits = ["train", "dev"]


for split in splits:
    src_file = os.path.join(source_json_dir, f"{split}.jsonl")
    dst_file = os.path.join(target_parquet_dir, f"{split}.parquet")
    df = pd.read_json(src_file, lines=True)
    df["data_source"] = "ai21"
    df = df.rename(columns={"messages": "prompt"})
    df["ability"] = "ai21"
    df["reward_model"] = df["reward_metrics"].apply(lambda x: {"style": "rule", "ground_truth": x})
    # TODO(RH): determine if the usage of split here is correct (ruff says it's not correctly bound)
    df["extra_info"] = df.apply(lambda row: {"split": split, "index": row.name}, axis=1)  # noqa: B023
    df = df[["data_source", "prompt", "ability", "reward_model", "extra_info"]]
    df.to_parquet(dst_file)
    print("Created parquet file at", dst_file)
