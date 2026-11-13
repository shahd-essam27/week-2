<div align="center">

# AI Engineering — Week 2
### Pretrained Model Inference · Benchmarking · Interactive AI Applications

**Hands-on experiments with Hugging Face Transformers, comparative model evaluation, and Gradio deployment.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-Interactive%20Apps-F97316)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Project-Educational-64748B)

[Overview](#overview) · [Projects & Labs](#projects--labs) · [Getting Started](#getting-started) · [Model Evaluation](#model-evaluation) · [Learning Outcomes](#learning-outcomes)

</div>

---

## Overview

This repository documents **Week 2 of my hands-on AI engineering learning journey**. It explores how to discover and compare pretrained models, run inference without training from scratch, package predictions into reusable Python functions, and build accessible web interfaces.

The work progresses from model research and sentiment analysis to performance benchmarking and a final **automated language-detection model scout**. Rather than selecting a model solely by name or popularity, the final project evaluates candidates on a small multilingual test suite and chooses a model using a combined accuracy–latency score.

> **Scope:** These are educational prototypes and small-scale experiments—not production-grade model evaluations or hosted services.

## Tech Stack

| Technology | Role |
| --- | --- |
| **Python** | Application logic, inference, and evaluation |
| **Hugging Face Transformers** | Loading and running pretrained NLP models through `pipeline()` |
| **PyTorch** | Model inference backend |
| **Gradio** | Interactive browser-based prediction interfaces |
| **Jupyter Notebook** | Exploratory experiments and model calls |
| **VS Code / Git / GitHub** | Development and version control |

## Projects & Labs

| Component | File | Purpose |
| --- | --- | --- |
| **Lab 2.1 — Model Research** | [`lab2.1.txt`](lab2.1.txt) | Compare two summarization models by model characteristics, licensing, and reported performance. |
| **Lab 2.2 — Reusable Inference** | [`lab2_2.ipynb`](lab2_2.ipynb), [`inference .py`](inference%20.py) | Load DistilBERT for sentiment classification and expose a reusable prediction function. |
| **Notebook Experiment** | [`notebook_one.ipynb`](notebook_one.ipynb) | Experiment with importing and calling the inference function; includes an integration issue investigated during development. |
| **Lab 2.3 — Benchmarking** | [`lab_2_3.py`](lab_2_3.py) | Compare DistilBERT and a Twitter-trained RoBERTa sentiment model on a small labeled test set. |
| **Lab 2.4 — Interactive UI** | [`lab_4.py`](lab_4.py) | Build a Gradio sentiment-analysis interface with predictions and confidence scores. |
| **Final Mini Project — Language Model Scout** | [`project .py`](project%20.py) | Benchmark language-identification models, select a candidate, and launch a Gradio demo. |

### Lab 2.1 — Exploring Pretrained Models

Research and compare the Hugging Face summarization models `facebook/bart-large-cnn` and `Falconsai/text_summarization`. The comparison considers licensing, model size, tensor information, and published performance details to build model-selection literacy before implementation.

### Lab 2.2 — Modular Sentiment Inference

Load `distilbert-base-uncased-finetuned-sst-2-english` with a Transformers text-classification pipeline and wrap predictions in a reusable Python function.

The function returns a sentiment label and a rounded model confidence score. This lab introduces the separation of **model initialization** from **application-level inference**.

### Lab 2.3 — Comparative Model Benchmarking

Evaluate two sentiment classifiers:

- `distilbert-base-uncased-finetuned-sst-2-english`
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

The script tests five labeled inputs and reports accuracy on the sample set, average prediction confidence, and average latency in milliseconds. These measurements provide a practical introduction to evaluating the trade-off between prediction behavior and responsiveness.

**Important:** The test set is intentionally small, and the models do not share identical sentiment label spaces (notably for neutral sentiment). Results are illustrative rather than a definitive head-to-head ranking.

### Lab 2.4 — Sentiment Analysis Web App

Use **Gradio** to turn the sentiment classifier into a browser-based application. Users can enter a sentence or review and receive a predicted sentiment label with its confidence score. The interface also handles empty input.

## Final Mini Project: Automated Language Detection & Model Selection

The capstone combines model scouting, benchmarking, automatic selection, and an interactive inference application.

### Candidate Models

| Model | Hugging Face model ID |
| --- | --- |
| **XLM-RoBERTa** | [`papluca/xlm-roberta-base-language-detection`](https://huggingface.co/papluca/xlm-roberta-base-language-detection) |
| **BERT language detector** | [`alexneakameni/language_detection`](https://huggingface.co/alexneakameni/language_detection) |

### How It Works

```text
                 ┌───────────────────────────────┐
                 │   Candidate language models   │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  Multilingual test samples    │
                 │     EN · FR · ES · AR · DE     │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │ Measure accuracy and latency  │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │   Compute selection score     │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │ Select highest-scoring model  │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  Gradio language detector     │
                 │  Language · Confidence · Name │
                 └───────────────────────────────┘
```

The code evaluates candidate models using English, French, Spanish, Arabic, and German examples. For each successfully loaded model, it measures prediction accuracy and average inference latency and computes:

```text
selection_score = accuracy_percent - (average_latency_ms / 100)
```

The model with the highest score is used for the interactive demo. This is a **custom educational heuristic**, not a standardized benchmark or a universally optimal ranking. The script also includes exception handling for model-loading failures and a fallback model if no candidate is successfully selected.

**Demo output:** Detected language, prediction confidence, and the selected model name.

> The demo runs locally and is configured to request a **temporary public Gradio sharing link** (`share=True`) when launched. This is not a permanent hosted deployment. Do not enter confidential text while public sharing is enabled.

## Getting Started

### Prerequisites

- Python 3 and `pip`
- Git (optional, for cloning the repository)
- Internet access for initial downloads of pretrained models
- Enough disk space and memory for model weights and dependencies

### 1. Clone the repository

```bash
git clone https://github.com/shahd-essam27/week-2.git
cd week-2
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Compatibility note:** The committed `requirements.txt` contains a large set of tightly pinned packages, including a CPU-specific PyTorch build. Installation may be platform-dependent. If the installation fails, verify your Python version, operating system, and appropriate [PyTorch installation instructions](https://pytorch.org/get-started/locally/). The main code dependencies are `transformers`, `torch`, and `gradio`; Jupyter is needed for the notebooks.

### 4. Run a lab or the final project

**Sentiment model benchmark:**

```bash
python lab_2_3.py
```

**Sentiment analysis Gradio app:**

```bash
python lab_4.py
```

**Automated language detector (final project):**

```bash
python "project .py"
```

Follow the URL displayed in your terminal to open the Gradio interface. The final project performs candidate model evaluation before launching the app, so startup time may be longer on the first run.

For notebooks, open `lab2_2.ipynb` or `notebook_one.ipynb` in VS Code with a Python/Jupyter kernel.

> **Filename note:** The repository currently contains filenames with spaces immediately before `.py`: `inference .py` and `project .py`. Use the exact names shown above until they are renamed.

## Model Evaluation

The experiments introduce three useful measurements:

| Metric | Meaning | Interpretation |
| --- | --- | --- |
| **Accuracy** | Proportion of test samples whose predicted labels match expected labels | Higher is better on the chosen test set. |
| **Average latency** | Average elapsed inference time per example, in milliseconds | Lower is generally better; depends on hardware and runtime conditions. |
| **Prediction confidence** | Average score returned by the model for its selected label | A model-reported score; **not** a direct measure of accuracy or calibration. |

### Evaluation Limitations

These exercises use small manually selected samples and do not establish generalization across real-world inputs. The benchmarks also do not comprehensively control for warm-up, hardware differences, class imbalance, or language coverage. A stronger future evaluation would add a larger held-out dataset, standardized label mapping, repeat measurements, and error analysis by language or sentiment category.

## Learning Outcomes

Through this project, I practiced:

- Discovering and comparing pretrained models using Hugging Face.
- Running NLP inference with Transformers pipelines instead of training models from scratch.
- Encapsulating model predictions in reusable Python functions.
- Measuring and interpreting accuracy, inference latency, and confidence scores.
- Comparing multiple candidate models with a reproducible evaluation script.
- Creating simple interactive machine-learning applications using Gradio.
- Combining model evaluation and user-facing inference into one end-to-end workflow.

## Future Improvements

- Expand the benchmark datasets and use held-out evaluation examples.
- Normalize sentiment labels before comparing models with different output classes.
- Repeat latency measurements after model warm-up for more reliable comparisons.
- Add unit tests for inference functions, empty input, and unexpected model outputs.
- Export benchmark results to CSV or JSON and add visualizations.
- Move model configurations into a dedicated module and simplify dependency pinning.
- Rename filenames consistently and separate experiments from application code.
- Add screenshots and an optional persistent deployment for the final Gradio demo.

## Author

**Shahd Essam**  
AI Engineering Learner · Python & NLP Projects  
[GitHub Profile](https://github.com/shahd-essam27)

---

<div align="center">

*Week 2 — From pretrained model exploration to evaluated, interactive NLP prototypes.*

</div>