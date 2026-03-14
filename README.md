# ACRONYM

> Micro AI for extracting acronyms and their definitions from `.docx` files.
> Training and detection are language-specific (**English** and **Italian**).

---

## Overview

ACRONYM is a small, trainable Python project that:

1. **Extracts candidate acronym-definition pairs** from Word documents (`.docx`) using language-aware regular expressions.
2. **Classifies each candidate** with a language-specific ML model (scikit-learn `LogisticRegression`) trained on labelled examples.
3. **Exposes a clean API** and a command-line interface for both training and detection.

Supported languages: `en` (English) and `it` (Italian).

---

## Project Structure

```
ACRONYM/
├── acronym/            # Main package
│   ├── __init__.py
│   ├── reader.py       # Read text from .docx files
│   ├── patterns.py     # Language-aware regex candidate extraction
│   ├── features.py     # Numeric feature extraction for the classifier
│   ├── model.py        # AcronymModel – sklearn pipeline wrapper
│   ├── trainer.py      # Training API (from JSON file or in-memory)
│   ├── detector.py     # Detection API (from .docx or plain text)
│   └── cli.py          # Command-line interface
├── data/
│   ├── en/samples.json # Labelled English training data
│   └── it/samples.json # Labelled Italian training data
├── models/             # Serialised trained models (auto-generated)
│   ├── model_en.pkl
│   └── model_it.pkl
├── tests/              # pytest test suite
├── requirements.txt
└── pyproject.toml
```

---

## Installation

```bash
pip install -r requirements.txt
# or install the package in editable mode
pip install -e .
```

---

## Quick Start

### 1 – Train a model

```bash
# English model
python -m acronym.cli train data/en/samples.json --lang en

# Italian model
python -m acronym.cli train data/it/samples.json --lang it
```

Pre-trained models are already included in `models/`, so you can skip this step and go straight to detection.

### 2 – Detect acronyms in a document

```bash
# Table output (default)
python -m acronym.cli detect report.docx --lang en

# JSON output
python -m acronym.cli detect report.docx --lang en --format json

# Italian document
python -m acronym.cli detect documento.docx --lang it
```

**Example output:**

```
Found 4 acronym(s):

Acronym         Confidence Definition
----------------------------------------------------------------------
AI              0.957      Artificial Intelligence
API             0.978      Application Programming Interface
CPU             0.979      Central Processing Unit
NATO            0.994      North Atlantic Treaty Organization
```

---

## Python API

```python
from acronym import detect_acronyms, detect_acronyms_from_text, train_from_file

# Train
train_from_file("data/en/samples.json", lang="en")

# Detect in a .docx file
results = detect_acronyms("report.docx", lang="en")
for r in results:
    print(f"{r['acronym']}  ({r['confidence']:.2f})  →  {r['definition']}")

# Detect in plain text
results = detect_acronyms_from_text(
    "The CPU (Central Processing Unit) runs at 3 GHz.",
    lang="en",
)
```

---

## Training Data Format

Training data is a JSON array of objects. Each object must have:

| Key            | Type    | Description                                        |
|----------------|---------|----------------------------------------------------|
| `acronym`      | string  | The acronym (e.g. `"NATO"`)                        |
| `definition`   | string  | The long form (e.g. `"North Atlantic Treaty Org…"`) |
| `pattern_type` | string  | `"before"` or `"after"` (optional, default `"before"`) |
| `label`        | integer | `1` = valid pair, `0` = invalid / noise            |

```json
[
  {
    "acronym": "NATO",
    "definition": "North Atlantic Treaty Organization",
    "pattern_type": "before",
    "label": 1
  },
  {
    "acronym": "IT",
    "definition": "Italy",
    "pattern_type": "before",
    "label": 0
  }
]
```

---

## How It Works

### Step 1 – Candidate Extraction (regex)

Two surface patterns are matched in both languages:

| Pattern | Example |
|---------|---------|
| Acronym before definition | `NATO (North Atlantic Treaty Organization)` |
| Definition before acronym | `North Atlantic Treaty Organization (NATO)` |

### Step 2 – Classification (micro AI)

Each candidate is converted to a numeric feature vector:

- Acronym length
- Definition word count (all words and significant words, excluding stop words)
- `is_all_caps` flag
- First-letter match ratio (how many acronym letters match word initials)
- Length ratio (acronym length ÷ word count)
- Whether the acronym appears verbatim in concatenated word initials
- Definition character length
- Pattern type flag (`before` / `after`)

A `StandardScaler + LogisticRegression` pipeline is trained per language and predicts a confidence score `[0, 1]` for each candidate. Candidates above the threshold (default `0.5`) are returned.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## CLI Reference

```
acronym train <data.json> [--lang {en,it}] [--model-dir DIR]
acronym detect <file.docx> [--lang {en,it}] [--model-dir DIR]
                            [--threshold FLOAT] [--format {table,json}]
```
