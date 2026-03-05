# ruff: noqa: E501,B904,UP038,E402
# Copyright 2025 AI21 Labs
"""
Dataset classes for working with amalgamated datasets in VERL.

The AmalgamDataset class is compatible with the RLHFDataset interface expected by ray_trainer.py.
It can be used as a custom_cls in the configuration for RayPPOTrainer.
"""

import copy
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed as concurrent_as_completed
from types import SimpleNamespace

import numpy as np
import pandas as pd
from ai21_file_utils.fast_gs_utils import copy_gs_dir_to_local_cache, copy_gs_file_to_local_cache
from ai21_file_utils.file_utils import open as ai21_open
from omegaconf import ListConfig
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.amalgam_dataset_mp import compute_valid_mask_for_chunk, init_process_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask

DEFAULT_AGGREGATION_CONFIG = {"expression": "mean"}


def read_file(file):
    return pd.read_json(file, lines=True)


def download_file(path, cache_dir, dataset_name):
    """
    Download a file/directory from a remote source to a local cache directory.

    Args:
        path (str): The path to the file/directory to download.
        cache_dir (str): The local cache directory to store the file.
        dataset_name (str): The name of the dataset to use for the cache directory.
    Returns:
        str: The local path to the file.
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset_name)
    if path.startswith("gs://"):
        if "*" in path:
            return copy_gs_dir_to_local_cache(dirpath=os.path.dirname(path), cache_path_base=dataset_cache_dir)
        else:
            return copy_gs_file_to_local_cache(path, dataset_cache_dir)
    else:
        return path


# used for parallel tokenization for length filtering
_process_pool = None


def set_global_process_pool(process_pool):
    global _process_pool
    _process_pool = process_pool


def fix_tools_val(tools) -> list:
    if isinstance(tools, list):
        return tools
    if np.isnan(tools) or tools is None:
        return []
    raise ValueError(f"`tools` is expected to be a list, got unexpected type `{type(tools)}` (value: {tools})")


class SingleSourceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_file: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int,
        cache_dir: str,
        filter_overlong_prompts=True,
        truncation="left",
        prompt_key="text",
        filter_overlong_prompts_workers=None,
        cache_tokenized_data=True,
        return_raw_chat=False,
        force_thinking=False,
        aggregation_config=None,
        jlm_model_names=None,
        jlm_model_names_seed=None,
    ):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.max_prompt_length = max_prompt_length
        self.filter_overlong_prompts = filter_overlong_prompts

        self.num_workers = filter_overlong_prompts_workers or max(1, os.cpu_count() // 4)
        self.num_workers = min(self.num_workers, os.cpu_count())

        self.cache_dir = os.path.expanduser(cache_dir)
        self.truncation = truncation
        self.prompt_key = prompt_key
        self.tokenizer = tokenizer
        self.cache_tokenized_data = cache_tokenized_data
        if self.cache_tokenized_data:
            self.tokenized_data = {}
        self.return_raw_chat = return_raw_chat
        self.force_thinking = force_thinking
        self.aggregation_config = aggregation_config
        self.num_examples_yielded = 0

        self.jlm_model_names = list(jlm_model_names) if isinstance(jlm_model_names, ListConfig) else jlm_model_names
        assert isinstance(self.jlm_model_names, list | None), (
            f"{self.dataset_name}: jlm_model_names must be a list or None, got {type(self.jlm_model_names)}"
        )
        if self.jlm_model_names is not None:
            print(
                f"Applying jlm_model_names override: One of [{', '.join(self.jlm_model_names)}] → {self.dataset_name} (will be overidden by 'change_verifiable_task_model_name' function from ai21-evaluators)"
            )
        self.rng = random.Random(jlm_model_names_seed)

        self._download()
        self._read_files_and_tokenize()

    def shuffle(self, seed):
        self.dataframe = self.dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)

    def _download(self):
        self.local_path = download_file(self.dataset_file, self.cache_dir, self.dataset_name)

    def _set_jlm_model_name(self):
        if self.jlm_model_names is None:
            return None
        return self.rng.choice(self.jlm_model_names)

    def _read_files_and_tokenize(self):
        prompt_key = self.prompt_key
        local_path = self.local_path
        dataset_name = self.dataset_name
        print(f"Reading {dataset_name} from {local_path}")
        # Check if the path is a file or directory
        if os.path.isfile(local_path):
            # Handle single file case
            df_list = [read_file(local_path)]
        else:
            # Handle directory case
            files = sorted(os.listdir(local_path))
            json_files = [file for file in files if file.endswith(".jsonl") and "*" not in file]
            with ThreadPoolExecutor() as executor:
                df_list = list(executor.map(read_file, [os.path.join(local_path, file) for file in json_files]))

        dataframe = pd.concat(df_list, ignore_index=True)
        data_source_name = dataset_name
        if not data_source_name.startswith("ai21/"):
            data_source_name = "ai21/" + data_source_name
        dataframe["data_source"] = data_source_name

        if "ability" not in dataframe.columns:
            dataframe["ability"] = "ai21"

        # ensure unique & indicative id for each example
        id = dataframe["data_source"].astype(str) + "_" + dataframe.index.astype(str)
        if "id" in dataframe.columns:
            id = id + "_" + dataframe["id"].astype(str)
        dataframe["id"] = id

        # Handle the new AI21-Evaluators format with 'evaluations' field
        if "evaluations" in dataframe.columns:
            dataframe["reward_metrics"] = dataframe["evaluations"]
        if "reward_metrics" in dataframe.columns:
            if "evaluations" in dataframe.columns:
                """
                Corresponding to the EvaluationEntry class in ai21_evaluators
                class EvaluationEntry(BaseModel):
                    query_type: Optional[str] = None
                    evaluator_name: Optional[str] = None
                    evaluator_config: Optional[dict[str, Any]] = None
                    query_args: Optional[dict[str, Any]] = None
                
                The evaluations field should contain a list of EvaluationEntry objects
                """
                # reward_model should contain the full evaluations list for ai21_evaluators_compute_score_ray_task
                dataframe["reward_model"] = dataframe["reward_metrics"]  # Keep the full list
            else:
                dataframe["reward_model"] = dataframe["reward_metrics"].apply(
                    lambda x: {"style": "rule", "ground_truth": x}
                )
            dataframe["jlm_model_name"] = None

        if "extra_info" not in dataframe.columns:
            if "metadata" in dataframe.columns:
                no_metadata_mask = dataframe["metadata"].isna()
                if sum(no_metadata_mask) > 0:
                    dataframe["metadata"] = dataframe["metadata"].astype(object)
                    dataframe.loc[no_metadata_mask, "metadata"] = [{} for _ in range(sum(no_metadata_mask))]
                dataframe["extra_info"] = dataframe["metadata"]
            else:
                dataframe["extra_info"] = dataframe.apply(
                    lambda row: {
                        "split": row.get("split", None),  # Use None if split column doesn't exist
                        "index": row.name,
                    },
                    axis=1,
                )

        if prompt_key not in dataframe.columns:
            # rename the existing key of ['text', 'messages', 'prompt'] to prompt_key
            renamed = None
            if "text" in dataframe.columns:
                renamed = "text"
            elif "messages" in dataframe.columns:
                renamed = "messages"
            elif "prompt" in dataframe.columns:
                renamed = "prompt"
            else:
                raise ValueError(f"not found {prompt_key} in {dataset_name}")
            dataframe.rename(columns={renamed: prompt_key}, inplace=True)
            print(f"renamed {renamed} to {prompt_key} in {dataset_name}")

        if self.aggregation_config is not None:
            dataframe["aggregation_config"] = [copy.deepcopy(self.aggregation_config) for _ in range(len(dataframe))]
        else:
            if "aggregation_config" not in dataframe.columns:
                dataframe["aggregation_config"] = [
                    copy.deepcopy(DEFAULT_AGGREGATION_CONFIG) for _ in range(len(dataframe))
                ]
            else:
                dataframe["aggregation_config"] = dataframe["aggregation_config"].apply(
                    lambda x: copy.deepcopy(DEFAULT_AGGREGATION_CONFIG) if pd.isna(x) or x is None else x
                )
        if "tools" not in dataframe.columns:
            dataframe["tools"] = [[] for _ in range(len(dataframe))]

        # handle `tools` entries with `nan`
        dataframe["tools"] = dataframe["tools"].apply(fix_tools_val)

        dataframe = dataframe[
            [
                "data_source",
                prompt_key,
                "tools",
                "ability",
                "reward_model",
                "extra_info",
                "id",
                "aggregation_config",
                "jlm_model_name",
            ]
        ]

        print(f"original dataset len: {len(dataframe)}, dataset: {dataset_name}")

        # Filter out too long prompts using pandas (no HuggingFace datasets conversion)
        if self.filter_overlong_prompts:
            original_len = len(dataframe)

            conversations = dataframe[prompt_key].tolist()
            tools = dataframe["tools"].tolist()
            valid_mask = []
            batch_size = 100
            start_time = time.time()
            # Parallelize with a process pool; create a tokenizer per process
            chunks = [conversations[i : i + batch_size] for i in range(0, len(conversations), batch_size)]
            tool_chunks = [tools[i : i + batch_size] for i in range(0, len(tools), batch_size)]
            if len(chunks) > 0:
                futures = [
                    _process_pool.submit(
                        compute_valid_mask_for_chunk, c, t, self.force_thinking, self.max_prompt_length
                    )
                    for c, t in zip(chunks, tool_chunks, strict=False)
                ]
                # Preserve order by iterating over futures sequentially
                for f in futures:
                    valid_mask.extend(f.result())
            dataframe = dataframe[valid_mask].reset_index(drop=True)

            filtered_len = len(dataframe)
            print(
                f"filter dataset len: {filtered_len}, took: {time.time() - start_time:.2f} seconds (filtered out {original_len - filtered_len} examples, dataset: {dataset_name})"
            )
            if filtered_len == 0:
                raise RuntimeError(
                    "All examples were filtered out by max_prompt_length. "
                    f"Dataset='{dataset_name}', path='{local_path}', "
                    f"max_prompt_length={self.max_prompt_length}, truncation='{self.truncation}', "
                    f"force_thinking={self.force_thinking}. "
                    "Consider increasing data.max_prompt_length, changing data.truncation, or disabling "
                    "data.filter_overlong_prompts."
                )

        # Keep as pandas DataFrame (no HuggingFace conversion to preserve reward_model schema)
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        """
        Returns an iterator over the dataset that yields each row as a dictionary,
        discarding the index from the pandas DataFrame's iterrows() method.
        """
        i = 0
        while i < len(self):
            yield self[i]
            self.num_examples_yielded += 1
            i += 1

    # to align with the logic in RLHFDataset
    def _build_messages(self, example: dict):
        messages = example.pop(self.prompt_key)
        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        if self.cache_tokenized_data and item in self.tokenized_data:
            row_dict = self.tokenized_data[item].copy()
            row_dict["jlm_model_name"] = self._set_jlm_model_name()
            return row_dict

        # Use pandas iloc for indexing
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        # Handle pandas NA values (NaN, None, NaT, etc.) after to_dict()
        for key, value in row_dict.items():
            if isinstance(value, np.float64) and pd.isna(value):
                row_dict[key] = None

        messages = self._build_messages(row_dict)
        model_inputs = {}

        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            force_thinking=self.force_thinking,
            tools=row_dict["tools"],
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["force_thinking"] = self.force_thinking

        if self.cache_tokenized_data:
            self.tokenized_data[item] = row_dict

        # sample jlm_model_name dynamically
        row_dict["jlm_model_name"] = self._set_jlm_model_name()

        return row_dict


class AmalgamDataset(IterableDataset):
    def __init__(
        self, data_files=None, tokenizer=None, processor=None, config=None, is_train=True, max_samples: int = None
    ):
        """
        AmalgamDataset constructor compatible with RLHFDataset interface.

        Args:
            data_files: Not used directly, kept for interface compatibility
            tokenizer: The tokenizer to use
            processor: Not used directly, kept for interface compatibility
            config: Configuration object containing dataset parameters
        """
        # Validate required parameters
        assert processor is None, "processor must be None. Multi-modality is not supported in this dataset"
        assert data_files is not None, "data_files must be provided"
        assert (isinstance(data_files, list) and len(data_files) == 1) or (isinstance(data_files, str)), (
            "data_files must be a list of length 1 or a string. It should be the mix json file"
        )
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if config is None:
            raise ValueError("config must be provided")
        if not hasattr(config, "prompt_key"):
            raise ValueError("config.prompt_key must be provided")
        if not hasattr(config, "max_prompt_length"):
            raise ValueError("config.max_prompt_length must be provided")
        assert config.cache_dir is not None, "config.cache_dir must be provided"

        # Extract parameters from config
        prompt_key = config.prompt_key
        max_prompt_length = config.max_prompt_length
        filter_overlong_prompts = config.filter_overlong_prompts
        cache_dir = config.cache_dir
        return_raw_chat = config.return_raw_chat
        truncation = config.truncation

        # Default to a deterministic seed when not provided; explicit None keeps behavior unseeded/random.
        seed = config.get("seed", 1)

        inter_ds_seed = config.get("amalgam_inter_ds_seed")
        intra_ds_seed = config.get("amalgam_intra_ds_seed")
        if inter_ds_seed is None:
            inter_ds_seed = seed
        if intra_ds_seed is None:
            intra_ds_seed = seed

        # Get shuffle setting
        shuffle = config.shuffle

        # Get finite setting
        if not is_train:
            finite = config.get("val_datasources_finite", True)
        else:
            finite = config.get("train_datasources_finite", False)

        # Get jlm_model_names setting
        self.jlm_model_names = config.get("jlm_model_names", None)
        self.jlm_model_names_seed = config.get("jlm_model_names_seed", None)

        self.mix_json_file = data_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.return_raw_chat = return_raw_chat
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_overlong_prompts = filter_overlong_prompts
        self.truncation = truncation
        self.shuffle = shuffle
        self.inter_ds_seed = inter_ds_seed
        self.intra_ds_seed = intra_ds_seed
        self.tokenizer = tokenizer
        self.finite = finite
        # TODO AI21: Actually do something with max samples
        self.max_samples = max_samples
        self._construct_datasets()

    def __getstate__(self):
        state = self.__dict__.copy()
        # delete datasets to make the pickle smaller
        if "datasets" in state:
            del state["datasets"]
        return state

    def __len__(self):
        if self.finite:
            return sum(len(dataset) for dataset in self.datasets.values())
        else:
            raise NotImplementedError("Infinite AmalgamDataset doesn't have a length!")

    def __iter__(self):
        iterators = {}
        weights = []

        def create_iterator(dataset: SingleSourceDataset):
            if self.shuffle:
                dataset.shuffle(self.intra_ds_seed)
            return iter(dataset)

        for dataset_name, dataset in self.datasets.items():
            iterators[dataset_name] = create_iterator(dataset)
            weight = self.weights[dataset_name]
            # this logic is taken from mammoth-dataloaders
            weights.extend([(dataset_name, 1.0)] * int(weight))
            if float(weight) != int(weight):
                p = float(weight) - int(weight)
                assert p
                weights.extend([(dataset_name, p)])

        rand = np.random.RandomState(self.inter_ds_seed)
        sentinel = object()
        exhausted_datasets = set()
        while True:
            rand.shuffle(weights)

            for dataset_name, weight in weights:
                if weight < 1.0 and rand.rand() > weight:
                    continue
                candidate = next(iterators[dataset_name], sentinel)
                if candidate is sentinel:
                    # iterator is exhausted, if finite, we skip this dataset
                    # we terminate the loop if all datasets are exhausted
                    if self.finite:
                        exhausted_datasets.add(dataset_name)
                        if len(exhausted_datasets) == len(self.datasets):
                            return
                        else:
                            continue
                    # if not finite, we reset the iterator
                    iterators[dataset_name] = create_iterator(self.datasets[dataset_name])
                    candidate = next(iterators[dataset_name])

                yield candidate

    def _construct_datasets(self):
        with ai21_open(self.mix_json_file) as f:
            mix = json.load(f)

        # Track all downloaded files
        self.datasets = {}
        self.weights = {}

        # Count total files to download for progress tracking
        total_datasets = len(mix)

        future_to_dataset_info = {}
        tokenizer_name_or_path = getattr(self.tokenizer, "name_or_path", None)
        # Add support for max_tokenizer_process_pool_workers configuration
        # If provided as an argument, use it; fall back to env or default
        max_tokenizer_process_pool_workers = int(
            os.environ.get("MAX_AMALGAM_TOKENIZER_PROCESS_POOL_WORKERS", 0.9 * os.cpu_count())
        )
        with ProcessPoolExecutor(
            initializer=init_process_tokenizer,
            initargs=[tokenizer_name_or_path],
            max_workers=max_tokenizer_process_pool_workers,
        ) as pp_executor:
            set_global_process_pool(pp_executor)
            with ThreadPoolExecutor(max_workers=32) as executor:
                pbar = tqdm(total=total_datasets, desc="Downloading datasets", unit="file")
                for dataset_name, dataset_config in mix.items():
                    path = dataset_config["path"]
                    # Submit all download tasks
                    force_thinking = dataset_config.get("chat_template_kwargs", {}).get("force_thinking", False)
                    aggregation_config = dataset_config.get("aggregation_config", None)
                    jlm_model_names = (
                        dataset_config["jlm_model_names"]
                        if "jlm_model_names" in dataset_config
                        else self.jlm_model_names
                    )
                    jlm_model_names_seed = (
                        dataset_config["jlm_model_names_seed"]
                        if "jlm_model_names_seed" in dataset_config
                        else self.jlm_model_names_seed
                    )
                    future_to_dataset_info[
                        executor.submit(
                            SingleSourceDataset,
                            dataset_name=dataset_name,
                            dataset_file=path,
                            tokenizer=self.tokenizer,
                            max_prompt_length=self.max_prompt_length,
                            cache_dir=self.cache_dir,
                            filter_overlong_prompts=self.filter_overlong_prompts,
                            prompt_key=self.prompt_key,
                            truncation=self.truncation,
                            return_raw_chat=self.return_raw_chat,
                            force_thinking=force_thinking,
                            aggregation_config=aggregation_config,
                            jlm_model_names=jlm_model_names,
                            jlm_model_names_seed=jlm_model_names_seed,
                        )
                    ] = (dataset_name, path)
                    self.weights[dataset_name] = float(dataset_config["weight"])

                for future in concurrent_as_completed(future_to_dataset_info):
                    dataset_name, dataset_path = future_to_dataset_info[future]
                    try:
                        dataset = future.result()
                        self.datasets[dataset_name] = dataset
                    except Exception as e:
                        print(f"Download failed for {dataset_path}: {e}")
                        raise e

                    # Update progress bar
                    pbar.update(1)
                # Recreate the dictionary by sorted keys to be agnostic of threading order
                # not doing this causes inconsistent data order between runs
                self.datasets = {k: self.datasets[k] for k in sorted(self.datasets)}

        set_global_process_pool(None)

    def get_dataset_stats(self):
        return [
            {
                "dataset": dataset_name,
                "size": len(dataset),
                "weight": self.weights[dataset_name],
                "num_examples_yielded": dataset.num_examples_yielded,
            }
            for dataset_name, dataset in self.datasets.items()
        ]

    def resume_dataset_state(self):
        self._construct_datasets()


if __name__ == "__main__":
    from types import SimpleNamespace

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # Create a config object similar to what RayPPOTrainer would pass
    config = SimpleNamespace(
        prompt_key="messages",
        max_prompt_length=8192,
        filter_overlong_prompts=True,
        truncation="left",
        shuffle=False,
        return_raw_chat=True,
        finite=True,
    )

    # Add get method to mimic OmegaConf behavior
    config.get = lambda key, default=None: getattr(config, key, default)

    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Mini-1.6")
    pth = "gs://ai21-algo-studio-research/verl_amalgam_experiments/amalgam_small.json"
    dataset = AmalgamDataset(
        data_files=pth,
        tokenizer=tokenizer,
        processor=None,  # Not used but kept for interface compatibility
        config=config,
    )

    print(dataset.get_dataset_stats())
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, drop_last=False, collate_fn=collate_fn)

    dl_iter = iter(dataloader)
    for batch in range(10):
        batch = next(dl_iter)
        print(len(batch["raw_prompt"]))
        print(batch["raw_prompt"][:10][:1])
