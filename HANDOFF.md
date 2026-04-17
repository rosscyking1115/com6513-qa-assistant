# COM6513/COM4513 Mid-Semester Assignment — Handoff Guide

This document is a full handoff for the next AI session to continue or finalise
Ross's QA assignment. Read this before touching any code.

---

## 1. Assignment Overview

**Module:** COM6513 / COM4513 — Natural Language Processing  
**Task:** Build an extractive Document QA Assistant  
**Student:** Ross King (leaffeng1115@gmail.com)  
**Machine:** MacBook 14 M4 Pro (Apple Silicon, no CUDA)

### What the assignment requires

1. **input.json** — 30 QA pairs (question_id, question, document) built from 3 Wikipedia PDFs
2. **qa_system.py** — extractive QA pipeline: chunks document → runs HuggingFace QA model → returns best answer → writes `predictions.json`
3. **evaluate.py** (named `qa_evaluate.py` here — see Known Issues) — computes EM and F1 against gold answers
4. **1-page LaTeX report** — System Description, Design Choices, Error Analysis

### Grading constraints
- `qa_system.py` must run end-to-end in under 10 seconds **per question** at test time
- Must use `pipeline("question-answering")` from HuggingFace Transformers
- No fine-tuning — pretrained model only
- Code must align with lab syntax (no type hints, lab-style patterns)

---

## 2. Environment

### Python version
**Python 3.10 is required.** The tokenizers package (transformers dependency) does not
support Python 3.12+ at build time. Ross's system Python is 3.14 — do NOT use it.

### Conda environment (always use this)
```bash
conda activate nlp          # Python 3.10.20
```

The project also has a `.venv` directory inside the project folder — this is an old
venv created during early debugging. **Ignore it.** Always use the `nlp` conda env.

### Install dependencies
```bash
conda activate nlp
pip install "numpy<2" torch "transformers==4.44.2" accelerate pymupdf evaluate
```

**Transformers must be pinned to 4.44.2.** Transformers v5 removed the
`question-answering` pipeline task entirely. v4.44.2 is the last stable v4 release
that works cleanly with Python 3.10 and the tokenizers wheel.

---

## 3. Project Structure

```
com6513-qa-assistant/           ← git repo root (cd here to run scripts)
├── src/
│   ├── build_dataset.py        ← DONE: extracts PDFs, builds input.json + gold_answers.json
│   ├── qa_system.py            ← DONE: main QA pipeline, writes predictions.json
│   └── qa_evaluate.py          ← DONE: EM + F1 evaluation (see Known Issues for filename)
├── data/
│   ├── raw/
│   │   ├── Computer_security.pdf
│   │   ├── Stuxnet.pdf
│   │   └── WannaCry_ransomware_attack.pdf
│   ├── input.json              ← GENERATED: 30 QA pairs for model input
│   ├── gold_answers.json       ← GENERATED: gold answers for local eval only
│   └── predictions.json        ← GENERATED: model predictions
├── report.tex                  ← DONE: 1-page LaTeX report (ready for Overleaf)
├── requirements.txt            ← numpy<2, torch, transformers==4.44.2, accelerate, pymupdf, evaluate
├── README.md
└── .gitignore
```

All scripts must be run from the project root (`com6513-qa-assistant/`):
```bash
cd ~/Documents/Files/Master/Semester\ 2/COM6113\ \(NLP\)/Assignment/Mid-Semester\ Assignment\ \(30%\)/com6513-qa-assistant/com6513-qa-assistant
```

---

## 4. How to Run Everything

```bash
# Step 1 — rebuild dataset (only needed if PDFs or QA pairs change)
python src/build_dataset.py

# Step 2 — run QA pipeline (takes ~3 min on CPU, writes predictions.json)
python src/qa_system.py

# Step 3 — evaluate predictions against gold answers
python src/qa_evaluate.py
```

---

## 5. Key Design Decisions (already made — do not change without good reason)

| Decision | Choice | Reason |
|---|---|---|
| Model | `deepset/roberta-base-squad2` | Strong SQuAD2-trained extractive model, lab-compatible |
| Chunking | Sentence-based sliding window | Preserves sentence boundaries vs fixed-length chunks |
| Window size | 5 sentences | Fits 512-token limit, keeps per-question time under 10s on CPU |
| Stride | 3 sentences | 2-sentence overlap ensures boundary answers appear in 2 chunks |
| Answer selection | Highest raw confidence score | Simple, effective, no extra dependencies |
| Device | `device = 0 if torch.cuda.is_available() else -1` | Exact Lab 1 pattern; no MPS handling (not in labs) |

---

## 6. Current Results

| Metric | Score |
|---|---|
| Exact Match (EM) | 43.33 |
| F1 | 61.42 |
| Questions | 30 (10 per document) |

### Per-document breakdown (approximate)
- **Computer Security** — longer document (~14,700 words), slower per-question (~8s), several definition questions where model returns short spans
- **Stuxnet** — medium (~7,000 words), ~5.7s per question, mostly good
- **WannaCry** — shortest (~3,500 words), ~3s per question, mostly good

### Notable failures (good for error analysis)
- q01 — wrong chunk retrieved entirely
- q05 — model picked car microphone span instead of eavesdropping definition  
- q07 — "social engineering" instead of full phishing definition (span too short)
- q24 — "Wanakiwi" (nearby distractor tool) instead of "Microsoft Visual C++ 6.0"
- q12 — "industrial infrastructure" instead of "SCADA systems"

---

## 7. Code Style Rules (important — enforced throughout)

All code must follow the lab syntax exactly:

- **No type hints** on functions (`def foo(x):` not `def foo(x: str) -> str:`)
- **No emoji** in print statements
- **Device detection** exactly as Lab 1: `device = 0 if torch.cuda.is_available() else -1`
- **Pipeline import**: `from transformers import pipeline` only
- **No MPS/Apple Silicon handling** — labs don't teach it
- **No extra imports** not seen in lab notebooks (no `time`, no `sys`, no `typing`)
- Short, plain docstrings — no Args/Returns sections

The 6 lab notebooks are at:
`~/Documents/.../mnt/uploads/` (uploaded in the previous session)  
Key reference: Lab 1 (device + pipeline pattern), Lab 5 (evaluate library pattern)

---

## 8. Known Issues / Gotchas

### evaluate.py naming conflict
The evaluation script is saved as `qa_evaluate.py`, **not** `evaluate.py`.  
If you name it `evaluate.py`, Python imports your file instead of the HuggingFace
`evaluate` library (circular import). Always keep it as `qa_evaluate.py`.

### FutureWarning from transformers
```
FutureWarning: `clean_up_tokenization_spaces` was not set.
```
This is harmless — it's a deprecation notice in transformers 4.44.2 about a default
that will change in 4.45. It does not affect results. Ignore it.

### Two-layer directory structure
The git repo was cloned such that there is a nested duplicate:
```
com6513-qa-assistant/           ← workspace mount (not the repo)
└── com6513-qa-assistant/       ← actual git repo root ← RUN SCRIPTS FROM HERE
```
Always `cd` into the inner `com6513-qa-assistant` before running any script.

### `.venv` vs `nlp` conda env
There is a `.venv` directory inside the project. It was created during early setup
when Python 3.14 was the system Python and broke tokenizers. It is now unused.
The correct environment is the `nlp` conda environment (Python 3.10).

---

## 9. Report Status

`report.tex` is complete and ready for Overleaf. It is exactly 1 page at 10pt on A4.

**Ross must replace `[YOUR REGISTRATION NUMBER]` on line 17 before submitting.**

The report covers:
- System Description — full pipeline description + EM 43.33 / F1 61.42
- Design Choices — model choice, chunking strategy, answer selection rationale
- Error Analysis — 3 concrete failures: wrong chunk (q05), span boundary (q08/q26), distractor (q24)

To compile: paste into Overleaf as a new blank LaTeX project and click Compile.
No extra packages needed beyond standard Overleaf defaults.

---

## 10. What Is Still TODO

- [ ] Ross fills in registration number in `report.tex`
- [ ] Compile report in Overleaf and check it fits 1 page
- [ ] Final `git add`, `git commit`, `git push` to submit the repo
- [ ] Verify `predictions.json` is present and has 30 entries before submitting

### Git push command
```bash
cd ~/Documents/Files/Master/Semester\ 2/COM6113\ \(NLP\)/Assignment/Mid-Semester\ Assignment\ \(30%\)/com6513-qa-assistant/com6513-qa-assistant
git add src/build_dataset.py src/qa_system.py src/qa_evaluate.py report.tex requirements.txt
git commit -m "Complete QA pipeline, evaluation, and report"
git push
```

---

## 11. What Was Completed in This Session

1. Designed 30 QA pairs across 3 Wikipedia documents (10 each)
2. Built `build_dataset.py` — PDF extraction with PyMuPDF, citation stripping, stop-before-References logic
3. Built `qa_system.py` — sentence chunking, sliding window, roberta-base-squad2 pipeline, highest-score selection
4. Debugged Python 3.14 / tokenizers incompatibility → resolved with conda Python 3.10
5. Debugged transformers v5 `question-answering` removal → resolved with `transformers==4.44.2`
6. Aligned all code with lab syntax (no type hints, no emoji, exact Lab 1 device pattern)
7. Built `qa_evaluate.py` using `evaluate.load("squad")` — same library taught in Lab 5
8. Ran full pipeline: EM 43.33, F1 61.42 over 30 questions
9. Wrote complete `report.tex` with real results and concrete error analysis
