import sys
from itertools import islice

from ai21_evaluators.file_formats.verifiable_task_dataset import VerifiableTask
from ai21_file_utils import iter_json_lines, load_json

train_data_config = sys.argv[1]
data_config = load_json(train_data_config)
for x in data_config:
    print(f"Checking {x}")
    data = list(islice(iter_json_lines(data_config[x]["path"]), 100))
    data = [VerifiableTask(**x) for x in data]
    print(f"{x} loaded successfully")
    print("=" * 100)
print("Data check finished successfully")
