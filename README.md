# COM6513 QA Assistant

Extractive document question-answering pipeline for the COM6513/COM4513 mid-semester assignment.

## Setup (macOS/Linux)

1. Open a terminal in the project root.
2. Create a Python virtual environment (Python 3.10 recommended):
   ```bash
   /opt/miniconda3/envs/nlp/bin/python -m venv .venv
   ```
   If you already have Python 3.10 on PATH, you can also use:
   ```bash
   python3.10 -m venv .venv
   ```
3. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Run

Generate predictions from `data/input.json`:

```bash
python src/qa_system.py
```

Predictions are written to `data/predictions.json`.

## Evaluate

Compare predictions against gold answers:

```bash
python src/qa_evaluate.py
```

## Optional: Rebuild Dataset from PDFs

If needed, place PDFs in `data/raw/` and run:

```bash
python src/build_dataset.py
```

This writes:

- `data/input.json`
- `data/gold_answers.json`
