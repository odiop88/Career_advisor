# Science2Voice Career Matching Experiment

This lab uses the Streamlit assessment fields and the 15 career pathway families in the pilot. You fine-tune an embedding model to rank pathways, then compare three ranking methods on the same profiles. The experiment runs without a generative AI API.

## 1. Set up your Jupyter notebook

Extract `science2voice_career_matching_lab.zip`. In JupyterLab or Jupyter Notebook, create a Python notebook inside the extracted `career_matching_lab` folder, next to `lab.py` and `requirements.txt`. Select a Python 3.11 or 3.12 kernel. On a remote bootcamp server, use a kernel running on the GPU machine.

Copy each code block into its own **code cell** and run the cells in order. The optional cells are marked below.

```python
from pathlib import Path
import sys

# If the notebook starts elsewhere, use Path(r"/path/to/career_matching_lab").
LAB_DIR = Path.cwd().resolve()

required_files = ("lab.py", "prepare_data.py", "requirements.txt")
missing_files = [name for name in required_files if not (LAB_DIR / name).is_file()]
if missing_files:
    raise FileNotFoundError(f"Set LAB_DIR to the extracted lab folder. Missing: {missing_files}")

print("Lab folder:", LAB_DIR)
print("Kernel Python:", sys.executable)
print("Python version:", sys.version.split()[0])
```

If the bootcamp kernel already has PyTorch 2.6 or later with working CUDA support, keep that installation. If you need to install PyTorch, choose the pip command for the GPU machine using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Paste the command into a separate notebook cell, replacing its initial `pip install` or `pip3 install` with `%pip install` and keeping the package arguments and index URL. Ask the bootcamp team which CUDA build matches the server's driver if you are unsure.

Install the lab dependencies in the selected kernel.

```python
requirements_path = LAB_DIR / "requirements.txt"
%pip install -r "{requirements_path}"
```

Restart the kernel after installing or updating packages. Then rerun the first setup cell to restore `LAB_DIR` and continue with the GPU check below.

```python
import os
import subprocess
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

DEVICE = "cuda"  # Change to "cpu" if you intentionally want a CPU run.
if DEVICE == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU kernel with CUDA-enabled PyTorch, or set DEVICE to 'cpu'.")
    print("GPU:", torch.cuda.get_device_name(0))

def run_lab(*args):
    """Run a lab script with the notebook kernel's Python interpreter."""
    return subprocess.run([sys.executable, "-u", *args], cwd=LAB_DIR, check=True)
```

The first run downloads the selected model weights from Hugging Face. Once the weights are cached, you can set `os.environ["HF_HUB_OFFLINE"] = "1"` before running the lab cells offline. If you run out of GPU memory, lower `--batch-size` to 4 or 2. On a shared server, run `!nvidia-smi` in a separate cell to check available GPU memory.

## 2. Review the synthetic data and reference rankings

```python
run_lab("prepare_data.py")
```

| Split | Distinct scenarios | Assessment responses |
|---|---:|---:|
| Training | 24 | 72 |
| Validation | 8 | 16 |
| Final test | 8 | 16 |

The `data/scenarios.tsv` file contains the 40 scenarios written for this exercise. Open `data/reference_examples.csv` in Excel to review one example per scenario and its three reference pathway titles. The three JSONL files contain the complete assessment responses, the scenario ID and the ordered reference ranking. Variants change the order of the narrative or omit some optional information. All variants of a scenario stay in the same split. The responses combine English form choices with fictional French narratives. The dataset contains only fictional records.

Each reference ranking lists two pathways from the person’s original field, in preference order, and one pivot from a predefined set of options to explore. For example, a laboratory profile that describes documenting duplicates and wanting to digitize a register has three reference pathways. They are laboratory quality management support, diagnostic laboratory operations and laboratory informatics support. These rankings are illustrative. Expert review and eligibility checks are separate steps.

## 3. Evaluate the base model

```python
run_lab(
    "lab.py", "evaluate", "--device", DEVICE,
    "--split", "dev",
    "--report", "runs/e5-before-dev.json",
)
```

The default model, [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), is a multilingual encoder released under the MIT license. It is suitable for a quick first run. The script applies the model’s `query:` and `passage:` prefixes. The report also includes a rule-based baseline that ranks the same eligible candidate groups using activities and interests.

## 4. Run a quick check and fine-tune the model

Run the two-step check first to verify that the training pipeline starts and can save a checkpoint.

```python
run_lab(
    "lab.py", "train", "--device", DEVICE,
    "--epochs", "1",
    "--max-steps", "2",
    "--batch-size", "4",
    "--output", "runs/smoke",
)
```

Then run the full three-epoch experiment.

```python
run_lab(
    "lab.py", "train", "--device", DEVICE,
    "--epochs", "3",
    "--batch-size", "8",
    "--learning-rate", "0.00002",
    "--output", "runs/e5",
)
```

Each training example combines a profile, a preferred pathway and a less relevant option from the same candidate group. The script creates three types of comparisons. It ranks the primary pathway ahead of the second choice, the second choice ahead of the remaining pathways in the original field, and the reference pivot ahead of the other pivot candidates. The model learns with a triplet loss based on cosine similarity. The initial run uses a margin of 0.10 and truncates text to 256 tokens. The script uses bfloat16 on compatible GPUs and float32 elsewhere.

The script selects the best checkpoint by exact ordered top-three accuracy on the validation split. It uses pivot accuracy to break ties. The exported model is in `runs/e5/model`. The `runs/e5/run.json` file records the settings and scores for the selected epoch. Use a new output folder for each training run.

## 5. Compare results and recommend pathways for a profile

Evaluate the fine-tuned model on validation first.

```python
run_lab(
    "lab.py", "evaluate", "--device", DEVICE,
    "--split", "dev",
    "--tuned", "runs/e5/model",
    "--report", "runs/e5-after-dev.json",
)
```

After choosing your settings, run the final test. If you plan to compare with BGE-M3, complete section 6 on validation first.

```python
run_lab(
    "lab.py", "evaluate", "--device", DEVICE,
    "--split", "test",
    "--tuned", "runs/e5/model",
    "--report", "runs/e5-test.json",
)
```

Generate a top-three recommendation for the supplied example profile.

```python
run_lab(
    "lab.py", "recommend", "--device", DEVICE,
    "--model", "runs/e5/model",
    "--input", "data/example_profile.json",
)
```

- `primary_accuracy` measures how often the first recommendation matches the reference.
- `aligned_pair_accuracy` measures how often both recommendations in the original field match the reference, regardless of order.
- `pivot_accuracy` measures how often the suggested career pivot matches the reference.
- `ordered_top3_accuracy` measures how often all three recommendations match the reference in the correct order.

The script averages results across variants within each scenario before averaging across scenarios. Use the `comparisons` section of the report to inspect individual predictions. Reserve the test split for the final comparison after choosing settings on the validation split. The eight test scenarios provide a preliminary result for the exercise. Assessing real-world performance requires anonymized real profiles or independent scenarios reviewed by health professionals in Senegal. Several pathways can be reasonable even when the reference ranking includes only one of them.

To open the validation report directly in the notebook, run this optional cell.

```python
import json

report_path = LAB_DIR / "runs" / "e5-after-dev.json"
with report_path.open(encoding="utf-8") as report_file:
    report = json.load(report_file)
report
```

The `recommend` command accepts a JSON object with the Streamlit form fields, as in `data/example_profile.json`. It displays practical first steps and entry checks from the catalog. The pilot rules return two options from the original field and one of the predefined pivots for exploration. Verify eligibility for employment, training or regulated practice with the relevant organization.

## 6. Optional comparison with BGE-M3

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) is a multilingual model released under the MIT license. The script uses its dense embeddings. The `--family bge` option omits the E5 prefixes. Compare models on the validation split before choosing which model to evaluate on the test split.

```python
run_lab(
    "lab.py", "evaluate", "--device", DEVICE,
    "--model", "BAAI/bge-m3",
    "--family", "bge",
    "--batch-size", "4",
    "--split", "dev",
)
```

```python
run_lab(
    "lab.py", "train", "--device", DEVICE,
    "--model", "BAAI/bge-m3",
    "--family", "bge",
    "--epochs", "3",
    "--batch-size", "4",
    "--output", "runs/bge",
)
```

```python
run_lab(
    "lab.py", "evaluate", "--device", DEVICE,
    "--model", "BAAI/bge-m3",
    "--family", "bge",
    "--batch-size", "4",
    "--split", "dev",
    "--tuned", "runs/bge/model",
    "--report", "runs/bge-dev.json",
)
```

These settings are starting points. With 24 training scenarios, a larger model can still overfit. Keep the base model if fine-tuning fails to improve validation performance.
