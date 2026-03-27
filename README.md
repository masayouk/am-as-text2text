# Argument Mining as a Text-to-Text Generation Task

[![EACL2024](https://img.shields.io/badge/EACL-2024-red)](https://aclanthology.org/volumes/2024.eacl-long/)
[![Paper](https://img.shields.io/badge/Paper-ACL%20Anthology-orange)](https://aclanthology.org/2024.eacl-long.121/)
[![DOI](https://img.shields.io/badge/DOI-10.18653%2Fv1%2F2024.eacl--long.121-blue)](https://doi.org/10.18653/v1/2024.eacl-long.121)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Mixed-lightgrey)](LICENSE)

Implementation of the approach described in **"Argument Mining as a Text-to-Text Generation Task"** (Kawarada et al., EACL 2024).

> **Abstract**: Argument Mining (AM) aims to uncover the argumentative structures within a text. Previous methods require several subtasks, such as span identification, component classification, and relation classification. Consequently, these methods need rule-based postprocessing to derive argumentative structures from the output of each subtask. This approach adds to the complexity of the model and expands the search space of the hyperparameters. To address this difficulty, we propose a simple yet strong method based on a text-to-text generation approach using a pretrained encoder-decoder language model. Our method simultaneously generates argumentatively annotated text for spans, components, and relations, eliminating the need for task-specific postprocessing and hyperparameter tuning. Furthermore, because it is a straightforward text-to-text generation method, we can easily adapt our approach to various types of argumentative structures. Experimental results demonstrate the effectiveness of our method, as it achieves state-of-the-art performance on three different types of benchmark datasets: the Argument-annotated Essays Corpus (AAEC), AbstRCT, and the Cornell eRulemaking Corpus (CDCP).

## Overview

This repository provides a unified pipeline for the argument mining approach described in the paper, using text-to-text generation with the [TANL](https://github.com/amazon-science/tanl) framework. Given dataset inputs in `conll` or `mrp` format, we fine-tune a pretrained encoder-decoder model (T5 / FLAN-T5) to generate argumentatively annotated text, from which spans, components, and relations are extracted.

**Key features:**

- **Text-to-Text Generation** &mdash; Unified approach that simultaneously generates spans, components, and relations without task-specific postprocessing
- **LoRA / QLoRA Fine-tuning** &mdash; Memory-efficient fine-tuning using LoRA or 4-bit QLoRA, enabling training of large models (up to 11B) on a single GPU
- **Seq2Seq Support** &mdash; Works with encoder-decoder models such as T5 and FLAN-T5
- **Multi-GPU Training** &mdash; Single-GPU training uses the standard Hugging Face Trainer, while multi-GPU training uses DeepSpeed
- **Single-GPU Selection / Annotation** &mdash; Checkpoint selection and final inference use a single GPU, keeping evaluation and resume logic simple
- **Automated Pipeline** &mdash; End-to-end workflow from data preparation through training, checkpoint selection, inference, and evaluation via a single `run` command
- **Unified Evaluation** &mdash; Evaluation framework supporting multiple argument mining datasets:
  - **AAEC (Essay/Paragraph)**: Token-based evaluation using ACL2017 metrics
  - **CDCP/AbstRCT**: Character-based evaluation using MRP scorer

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)

All runtime dependencies (PyTorch, Transformers, PEFT, bitsandbytes, DeepSpeed, Accelerate, spaCy, etc.) are declared in `pyproject.toml` and installed automatically.

## Installation

```bash
uv sync
```

## Data Preparation

Place the prepared dataset files under `am_datasets/conll/` (AAEC) and `am_datasets/mrp/` (AbstRCT, CDCP). If you need to prepare them from scratch, follow the instructions below.

### Prerequisites

Before preprocessing, you need to obtain the original datasets:

- **AAEC (Essay/Paragraph)**: Pre-processed CoNLL format files from [acl2017-neural_end2end_am](https://github.com/UKPLab/acl2017-neural_end2end_am/tree/master/data/conll) ([Eger et al., ACL 2017](https://aclanthology.org/P17-1002/))
- **AbstRCT**: [GitLab](https://gitlab.com/tomaye/abstrct/) &mdash; Original brat format files need to be converted to MRP format
- **CDCP**: [cdcp_acl17.zip](https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip) &mdash; Original JSON format files need to be converted to MRP format

### AAEC

Download the pre-processed CoNLL format files from the [acl2017-neural_end2end_am](https://github.com/UKPLab/acl2017-neural_end2end_am/tree/master/data/conll) repository ([Eger et al., ACL 2017](https://aclanthology.org/P17-1002/)). Place the `Essay_Level/` and `Paragraph_Level/` directories under `am_datasets/conll/`:

```
am_datasets/conll/Essay_Level/
am_datasets/conll/Paragraph_Level/
```

The original (raw) AAEC corpus is available from [TU Darmstadt UKP Lab](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422).

### CDCP

The converter modules used in the commands below are **not included in this
repository**. They are expected to be run from an external preprocessing
repository used in prior work; the commands here are provided only as a
reference for how the MRP files were prepared.

```bash
# Download and extract CDCP dataset
wget --no-check-certificate https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip
unzip cdcp_acl17.zip

# Convert CDCP training set to MRP format
uv run python -m src.preprocess.graph_parser_converter.cdcp2mrp \
  --dir_cdcp cdcp/train \
  --prefix CDCP_ \
  --output am_datasets/mrp/cdcp/cdcp_train.mrp

# Split into train and dev sets
uv run python -m src.preprocess.graph_parser_converter.split_mrp \
  --input am_datasets/mrp/cdcp/cdcp_train.mrp \
  --output1 am_datasets/mrp/cdcp/cdcp_train.mrp \
  --output2 am_datasets/mrp/cdcp/cdcp_dev.mrp \
  --output2_rate 0.1 \
  --seed 42

# Convert CDCP test set to MRP format
uv run python -m src.preprocess.graph_parser_converter.cdcp2mrp \
  --dir_cdcp cdcp/test \
  --prefix CDCP_ \
  --output am_datasets/mrp/cdcp/cdcp_test.mrp
```

### AbstRCT

The converter modules used in the commands below are **not included in this
repository**. They are expected to be run from an external preprocessing
repository used in prior work; the commands here are provided only as a
reference for how the MRP files were prepared.

```bash
# Download and extract AbstRCT dataset
wget --no-check-certificate https://gitlab.com/tomaye/abstrct/-/archive/master/abstrct-master.zip
unzip abstrct-master.zip

# Convert AbstRCT training set to MRP format
uv run python -m src.preprocess.graph_parser_converter.abstrct2mrp \
  --dir_abstrct abstrct-master/AbstRCT_corpus/data/train/neoplasm_train \
  --prefix AbstRCT_ \
  --output am_datasets/mrp/abstrct/abstrct_train.mrp

# Convert AbstRCT dev set to MRP format
uv run python -m src.preprocess.graph_parser_converter.abstrct2mrp \
  --dir_abstrct abstrct-master/AbstRCT_corpus/data/dev/neoplasm_dev \
  --prefix AbstRCT_ \
  --output am_datasets/mrp/abstrct/abstrct_dev.mrp

# Convert AbstRCT test set to MRP format
uv run python -m src.preprocess.graph_parser_converter.abstrct2mrp \
  --dir_abstrct abstrct-master/AbstRCT_corpus/data/test/neoplasm_test \
  --prefix AbstRCT_ \
  --output am_datasets/mrp/abstrct/abstrct_test.mrp
```

### Directory Structure

After data preparation, the datasets should be organized as follows:

```
am_datasets/
├── conll/                         # CoNLL formatted files (AAEC)
│   ├── Essay_Level/
│   │   ├── train.dat / train.dat.abs
│   │   ├── dev.dat   / dev.dat.abs
│   │   └── test.dat  / test.dat.abs
│   └── Paragraph_Level/
│       ├── train.dat / train.dat.abs
│       ├── dev.dat   / dev.dat.abs
│       └── test.dat  / test.dat.abs
└── mrp/                           # MRP formatted files (CDCP / AbstRCT)
    ├── abstrct/
    │   ├── abstrct_train.mrp
    │   ├── abstrct_dev.mrp
    │   └── abstrct_test.mrp
    └── cdcp/
        ├── cdcp_train.mrp
        ├── cdcp_dev.mrp
        └── cdcp_test.mrp
```

## Usage

All experiments are driven by a single YAML configuration file. See `examples/` for sample configs.

### Running the Full Pipeline

The `run` command executes the entire pipeline end-to-end:

```bash
am-text2text run --config examples/aaec_flan_t5_base_1gpu.yaml
```

If the same `run.name` is executed again with the same training configuration, completed training is reused automatically. If training stopped mid-run and checkpoints exist, the latest checkpoint is used to resume training.

Training runs in two modes only:
- **Single GPU**: standard Hugging Face `Trainer`
- **Multi GPU**: DeepSpeed launcher with the configured ZeRO setup

Checkpoint selection and final annotation currently support only a single GPU. In LoRA / QLoRA runs, checkpoint selection reuses the loaded base model and tokenizer across checkpoints instead of reloading them each time.

This sequentially performs:
1. **Dataset materialization** &mdash; Loads CoNLL or MRP source files into a canonical format
2. **Annotator dataset preparation** &mdash; Converts the canonical dataset into TANL-formatted training data
3. **Training** &mdash; Fine-tunes the model with LoRA / QLoRA
4. **Checkpoint selection** &mdash; Evaluates all checkpoints on the dev set and selects the best one
5. **Annotation** &mdash; Runs inference on the test set using the best checkpoint
6. **Evaluation** &mdash; Computes Span-F1, Component-F1, and Relation-F1 scores

For MRP datasets, evaluation uses dataset-specific label parsing for `abstrct` and `cdcp`, and converts token spans back to character offsets from the canonical dataset instead of re-tokenizing raw text.

### Running Individual Steps

Each step can also be run independently:

```bash
# Step 1: Materialize the source dataset into canonical format
am-text2text materialize-dataset --config examples/aaec_flan_t5_base_1gpu.yaml

# Step 2: Prepare TANL-formatted training data
am-text2text prepare-annotator-dataset --config examples/aaec_flan_t5_base_1gpu.yaml

# Step 3: Fine-tune the model
am-text2text train-annotator --config examples/aaec_flan_t5_base_1gpu.yaml

# Step 4: Select the best checkpoint based on dev set performance
am-text2text select-checkpoint --config examples/aaec_flan_t5_base_1gpu.yaml

# Step 5: Run inference on the test set
am-text2text annotate --config examples/aaec_flan_t5_base_1gpu.yaml

# Step 6: Evaluate predictions
am-text2text evaluate --config examples/aaec_flan_t5_base_1gpu.yaml
```

## Configuration

The configuration file follows a structured YAML schema validated by Pydantic. Below are the top-level sections:

| Section | Description |
|---------|-------------|
| `run` | Run name and random seed |
| `dataset` | Dataset name, source type (`conll` / `mrp`), and source directory |
| `prepared_data` | TANL preprocessing option and fulltext option |
| `model` | Base model, prompt configuration, adapter (LoRA), and quantization settings |
| `train` | Training hyperparameters, GPU count, and DeepSpeed settings |
| `checkpoint_selection` | Dev-set checkpoint selection strategy and decoding parameters |
| `annotation` | Test-set inference configuration |
| `evaluation` | Evaluation output format |
| `outputs` | Root directory for all outputs |

### Example Configs

- [`examples/aaec_flan_t5_base_1gpu.yaml`](examples/aaec_flan_t5_base_1gpu.yaml) &mdash; FLAN-T5-Base with LoRA on AAEC Essay Level (single GPU, `lr=5.0e-4`)
- [`examples/aaec_flan_t5_xxl_8gpu_deepspeed.yaml`](examples/aaec_flan_t5_xxl_8gpu_deepspeed.yaml) &mdash; FLAN-T5-XXL with LoRA on AAEC Essay Level using 8 GPUs with DeepSpeed ZeRO-2 (`lr=2.0e-4`, `per_device_train_batch_size=1`)
- [`examples/cdcp_flan_t5_base_1gpu.yaml`](examples/cdcp_flan_t5_base_1gpu.yaml) &mdash; FLAN-T5-Base with LoRA on CDCP (single GPU, `lr=5.0e-4`)
- [`examples/cdcp_flan_t5_xxl_8gpu_deepspeed.yaml`](examples/cdcp_flan_t5_xxl_8gpu_deepspeed.yaml) &mdash; FLAN-T5-XXL with LoRA on CDCP using 8 GPUs with DeepSpeed ZeRO-2 (`lr=2.0e-4`, `per_device_train_batch_size=1`)
- [`examples/abstrct_flan_t5_base_1gpu.yaml`](examples/abstrct_flan_t5_base_1gpu.yaml) &mdash; FLAN-T5-Base with LoRA on AbstRCT (single GPU, `lr=5.0e-4`)
- [`examples/abstrct_flan_t5_xxl_8gpu_deepspeed.yaml`](examples/abstrct_flan_t5_xxl_8gpu_deepspeed.yaml) &mdash; FLAN-T5-XXL with LoRA on AbstRCT using 8 GPUs with DeepSpeed ZeRO-2 (`lr=2.0e-4`, `per_device_train_batch_size=1`)

### Adapter & Quantization Modes

| Mode | `adapter.enabled` | `quantization.mode` | Description |
|------|-------------------|---------------------|-------------|
| LoRA | `true` | `none` | Standard LoRA without quantization |
| QLoRA | `true` | `4bit` | 4-bit quantized model with LoRA adapters |

## Output Structure

All outputs are organized under the configured `outputs.root_dir`:

```
outputs/
├── prepared/
│   ├── canonical/<dataset_name>/
│   │   ├── canonical.json              # Canonical dataset representation
│   │   └── manifest.json               # Dataset metadata
│   └── annotator_dataset/<dataset_name>/
│       ├── <dataset_name>_train.json
│       ├── <dataset_name>_dev.json
│       ├── <dataset_name>_test.json
│       └── manifest.json
└── runs/<run_name>/
    ├── config.yaml                     # Snapshot of the config used
    ├── run.json                        # Run status and metadata
    ├── train/
    │   ├── model/                      # Trained model, checkpoints, and done.json
    │   └── request.json
    ├── checkpoint_selection/
    │   ├── request.json                # Checkpoint-selection request snapshot
    │   ├── dev/checkpoint-*/           # Per-checkpoint request, predictions, and metrics
    │   └── selected_checkpoint.json    # Best checkpoint info
    └── test/
        ├── request.json                # Final annotation request snapshot
        ├── raw_predictions.json        # Raw model outputs
        ├── parsed_predictions.json     # Structured predictions
        └── metrics.json                # Span-F1, Component-F1, Relation-F1 scores
```

## Evaluation

The evaluation framework supports two evaluation protocols depending on the dataset:

- **AAEC (essay / paragraph)**: Uses the ACL2017 evaluation scripts in [`src/am_text2text/evaluation/upstream/aaec_acl2017/`](src/am_text2text/evaluation/upstream/aaec_acl2017), implemented here as a Python 3 reimplementation of the original Python 2 code from [UKPLab/acl2017-neural_end2end_am](https://github.com/UKPLab/acl2017-neural_end2end_am) for [Eger et al., ACL 2017](https://aclanthology.org/P17-1002/). The overlap threshold (`ratio`) is set to 0.999, which is effectively equivalent to the exact match used in ACL2017.
- **CDCP / AbstRCT**: Uses the local MRP evaluation adapter together with [`src/am_text2text/evaluation/upstream/graph_parser_scorer/scorer.py`](src/am_text2text/evaluation/upstream/graph_parser_scorer/scorer.py), which is adapted from [hitachi-nlp/graph_parser](https://github.com/hitachi-nlp/graph_parser) for [Morio et al., TACL 2022](https://aclanthology.org/2022.tacl-1.37/), with character-based matching

Both evaluators compute **Span-F1**, **Component-F1**, and **Relation-F1** scores. Checkpoint selection uses the average of Component-F1 and Relation-F1 by default.

Note that the EACL 2024 paper reported AbstRCT and CDCP results with the token-based evaluator used at publication time. This repository instead uses the character-based evaluator to match prior work on these datasets, so the reported numbers for AbstRCT and CDCP may differ from the values in the paper.

For example, in the FLAN-T5-Base experiments:

- **AbstRCT**
  - token-based evaluation in the paper: Component-F1 = 68.76, Relation-F1 = 38.31
  - character-based evaluation in this repository: Component-F1 = 72.62, Relation-F1 = 38.90
- **CDCP**
  - token-based evaluation in the paper: Component-F1 = 66.80, Relation-F1 = 23.19
  - character-based evaluation in this repository: Component-F1 = 58.84, Relation-F1 = 24.75

In practice, the difference is especially visible in Component-F1. A likely reason is that the character-based evaluator is more sensitive to span-to-offset conversion details, including whitespace and boundary handling, than the token-based evaluator used in the paper.


## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{kawarada-etal-2024-argument,
    title = "Argument Mining as a Text-to-Text Generation Task",
    author = "Kawarada, Masayuki  and
      Hirao, Tsutomu  and
      Uchida, Wataru  and
      Nagata, Masaaki",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.121/",
    doi = "10.18653/v1/2024.eacl-long.121",
    pages = "2002--2014",
    abstract = "Argument Mining (AM) aims to uncover the argumentative structures within a text. Previous methods require several subtasks, such as span identification, component classification, and relation classification. Consequently, these methods need rule-based postprocessing to derive argumentative structures from the output of each subtask. This approach adds to the complexity of the model and expands the search space of the hyperparameters. To address this difficulty, we propose a simple yet strong method based on a text-to-text generation approach using a pretrained encoder-decoder language model. Our method simultaneously generates argumentatively annotated text for spans, components, and relations, eliminating the need for task-specific postprocessing and hyperparameter tuning. Furthermore, because it is a straightforward text-to-text generation method, we can easily adapt our approach to various types of argumentative structures.Experimental results demonstrate the effectiveness of our method, as it achieves state-of-the-art performance on three different types of benchmark datasets: the Argument-annotated Essays Corpus (AAEC), AbstRCT, and the Cornell eRulemaking Corpus (CDCP)."
}
```

**Paper**: [Argument Mining as a Text-to-Text Generation Task](https://aclanthology.org/2024.eacl-long.121/) (EACL 2024)

**DOI**: [10.18653/v1/2024.eacl-long.121](https://doi.org/10.18653/v1/2024.eacl-long.121)

## License

This repository uses mixed licensing.

- Unless otherwise noted, self-authored original code and documentation in this repository are licensed under **CC BY-NC-SA 4.0**.
- Files in `src/am_text2text/evaluation/upstream/aaec_acl2017/` are Python 3 reimplementations of upstream evaluation code released under **Apache License 2.0**.
- [`src/am_text2text/evaluation/upstream/graph_parser_scorer/scorer.py`](src/am_text2text/evaluation/upstream/graph_parser_scorer/scorer.py) follows the upstream **CC BY-NC-SA 4.0** licensing context of [hitachi-nlp/graph_parser](https://github.com/hitachi-nlp/graph_parser).
- Earlier releases of this repository that were published under **Apache License 2.0** remain available under those earlier terms.

See the [LICENSE](LICENSE) file for the repository-wide notice, [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance details, and `LICENSES/` for local copies of the referenced license texts.

## Acknowledgments

This project builds upon the following works:

- **TANL**: The TANL parser implementation is adapted from the [Amazon TANL repository](https://github.com/amazon-science/tanl) (Paolini et al., ICLR 2021), licensed under Apache 2.0.
- **AAEC Evaluation Scripts**: The AAEC evaluation modules in `src/am_text2text/evaluation/upstream/aaec_acl2017/` are Python 3 reimplementations of the original Python 2 code from [acl2017-neural_end2end_am](https://github.com/UKPLab/acl2017-neural_end2end_am) ([Eger et al., ACL 2017](https://aclanthology.org/P17-1002/)), licensed under Apache 2.0.
- **MRP Evaluation**: [`src/am_text2text/evaluation/upstream/graph_parser_scorer/scorer.py`](src/am_text2text/evaluation/upstream/graph_parser_scorer/scorer.py) is adapted from [hitachi-nlp/graph_parser](https://github.com/hitachi-nlp/graph_parser) ([Morio et al., TACL 2022](https://aclanthology.org/2022.tacl-1.37/)) under the upstream CC BY-NC-SA 4.0 licensing context. The surrounding conversion and evaluation adapter code is project-specific.
