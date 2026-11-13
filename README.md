# Science2Voice Career Matching Experiment

This lab uses the Streamlit assessment fields and the 15 career pathway families in the pilot. You fine-tune an embedding model to rank pathways, then compare three ranking methods on the same profiles. The experiment runs without a generative AI API.

## 1. Set up the GPU

In a Linux or WSL terminal, go to the extracted `career_matching_lab` folder. Use Python 3.11 or 3.12. On native Windows, replace `source .venv/bin/activate` with `.\.venv\Scripts\Activate.ps1` and `python3` with `py -3.12`.

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch 2.6 or later using the official selector. Use the CUDA installation command recommended by the bootcamp team for the installed GPU driver. If the bootcamp environment already has a working CUDA-enabled PyTorch installation, you can use it directly.

- [PyTorch installation selector for Linux, Windows and CUDA](https://pytorch.org/get-started/locally/)

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The last command should print the GPU name. To run on a CPU, install a CPU build of PyTorch and replace `--device cuda` with `--device cpu`. The first run downloads the model weights from Hugging Face. Once the weights are cached, set `HF_HUB_OFFLINE=1` to work offline. If you run out of GPU memory, try `--batch-size 4`, then `--batch-size 2`. On a shared server, also check how much GPU memory is available.

## 2. Review the synthetic data and reference rankings

```bash
python prepare_data.py
```

| Split | Distinct scenarios | Assessment responses |
|---|---:|---:|
| Training | 24 | 72 |
| Validation | 8 | 16 |
| Final test | 8 | 16 |

The `data/scenarios.tsv` file contains the 40 scenarios written for this exercise. Open `data/reference_examples.csv` in Excel to review one example per scenario and its three reference pathway titles. The three JSONL files contain the complete assessment responses, the scenario ID and the ordered reference ranking. Variants change the order of the narrative or omit some optional information. All variants of a scenario stay in the same split. The responses combine English form choices with fictional French narratives. The dataset contains only fictional records.

Each reference ranking lists two pathways from the person’s original field, in preference order, and one pivot from a predefined set of options to explore. For example, a laboratory profile that describes documenting duplicates and wanting to digitize a register has three reference pathways. They are laboratory quality management support, diagnostic laboratory operations and laboratory informatics support. These rankings are illustrative. Expert review and eligibility checks are separate steps.

## 3. Evaluate the base model

```bash
python lab.py evaluate --device cuda --split dev --report runs/e5-before-dev.json
```

The default model, [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), is a multilingual encoder released under the MIT license. It is suitable for a quick first run. The script applies the model’s `query:` and `passage:` prefixes. The report also includes a rule-based baseline that ranks the same eligible candidate groups using activities and interests.

## 4. Run a quick check and fine-tune the model

```bash
python lab.py train --device cuda --epochs 1 --max-steps 2 --batch-size 4 --output runs/smoke
python lab.py train --device cuda --epochs 3 --batch-size 8 --learning-rate 0.00002 --output runs/e5
```

Each training example combines a profile, a preferred pathway and a less relevant option from the same candidate group. The script creates three types of comparisons. It ranks the primary pathway ahead of the second choice, the second choice ahead of the remaining pathways in the original field, and the reference pivot ahead of the other pivot candidates. The model learns with a triplet loss based on cosine similarity. The initial run uses a margin of 0.10 and truncates text to 256 tokens. The script uses bfloat16 on compatible GPUs and float32 elsewhere.

The script selects the best checkpoint by exact ordered top-three accuracy on the validation split. It uses pivot accuracy to break ties. The exported model is in `runs/e5/model`. The `runs/e5/run.json` file records the settings and scores for the selected epoch. Use a new output folder for each training run.

## 5. Compare results and recommend pathways for a profile

```bash
python lab.py evaluate --device cuda --split dev --tuned runs/e5/model --report runs/e5-after-dev.json
python lab.py evaluate --device cuda --split test --tuned runs/e5/model --report runs/e5-test.json
python lab.py recommend --device cuda --model runs/e5/model --input data/example_profile.json
```

- `primary_accuracy` measures how often the first recommendation matches the reference.
- `aligned_pair_accuracy` measures how often both recommendations in the original field match the reference, regardless of order.
- `pivot_accuracy` measures how often the suggested career pivot matches the reference.
- `ordered_top3_accuracy` measures how often all three recommendations match the reference in the correct order.

The script averages results across variants within each scenario before averaging across scenarios. Use the `comparisons` section of the report to inspect individual predictions. Reserve the test split for the final comparison after choosing settings on the validation split. The eight test scenarios provide a preliminary result for the exercise. Assessing real-world performance requires anonymized real profiles or independent scenarios reviewed by health professionals in Senegal. Several pathways can be reasonable even when the reference ranking includes only one of them.

The `recommend` command accepts a JSON object with the Streamlit form fields, as in `data/example_profile.json`. It displays practical first steps and entry checks from the catalog. The pilot rules return two options from the original field and one of the predefined pivots for exploration. Verify eligibility for employment, training or regulated practice with the relevant organization.

## 6. Compare with BGE-M3 if GPU capacity permits

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) is a multilingual model released under the MIT license. The script uses its dense embeddings. The `--family bge` option omits the E5 prefixes. Compare models on the validation split before choosing which model to evaluate on the test split.

```bash
python lab.py evaluate --device cuda --model BAAI/bge-m3 --family bge --batch-size 4 --split dev
python lab.py train --device cuda --model BAAI/bge-m3 --family bge --epochs 3 --batch-size 4 --output runs/bge
python lab.py evaluate --device cuda --model BAAI/bge-m3 --family bge --batch-size 4 --split dev --tuned runs/bge/model --report runs/bge-dev.json
```

These settings are starting points. With 24 training scenarios, a larger model can still overfit. Keep the base model if fine-tuning fails to improve validation performance.
