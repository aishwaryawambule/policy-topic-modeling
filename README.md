# Policy Topic Modeling — Nepal Plans & Budget Speeches

Coursework Task 1, *Advanced Machine Learning* (ST7085CEM),
Softwarica College of IT & E-Commerce in collaboration with
Coventry University.

**Topic.** Tracing Policy Priorities in Nepal: A Comparative Topic
Modeling Analysis of Government Five-Year Plans and Budget Speeches.

**Group.** Sandesh Tamang & Aishwarya Rai.

**Algorithms.** Latent Dirichlet Allocation (LDA) for unsupervised
topic discovery + Gaussian Process classifier and Gaussian Process
regressor over the per-document topic distributions.

## Layout
```
policy_topic_modeling/
├── code/
│   ├── 01_download.py        # downloads PDFs from verified sources
│   ├── 02_extract.py         # PDF → plain text
│   └── 03_analyze.py         # preprocessing, LDA, GP, figures
├── data/
│   ├── sources.json          # URL list with labels
│   ├── raw_pdfs/             # downloaded PDFs
│   ├── plans/                # extracted Periodic Plan text
│   ├── speeches/             # extracted Budget Speech text
│   ├── processed/            # metrics.json, topic-doc matrix, ...
│   ├── download_report.json
│   └── extract_report.json
├── figures/                  # all PNGs used by the paper
└── report/
    ├── main.tex              # IEEE conference, two-column source
    └── figures/              # copy of /figures so Overleaf works
```

## Reproduce

```bash
# from project root /Users/aishwaryarai/Documents/College
uv sync                                                # install deps
uv run python policy_topic_modeling/code/01_download.py
uv run python policy_topic_modeling/code/02_extract.py
uv run python policy_topic_modeling/code/03_analyze.py
```

Total wall-clock ≈ 3–4 min on Apple Silicon.  All random seeds are
fixed at 42.

## Compile the report

The paper uses the IEEE Conference template (`IEEEtran.cls`).
Easiest path:

1. Open <https://overleaf.com>, create a new project from the IEEE
   Conference template (or upload an empty one).
2. Upload `report/main.tex` and the entire `report/figures/` folder.
3. Compile with pdfLaTeX. The output PDF is a 6-page, two-column
   paper.

Alternatively, locally with MacTeX or TeX Live:
```bash
cd policy_topic_modeling/report
pdflatex main.tex && pdflatex main.tex
```

## Key results

| Task                                   | Metric                | Score                |
|----------------------------------------|-----------------------|----------------------|
| LDA model selection                    | perplexity @ K=8      | 1220                 |
| Plan vs Speech (GP, RBF)               | accuracy / F₁ (5-CV)  | **0.907** / **0.901** |
| Plan vs Speech (Logistic Regression)   | accuracy / F₁ (5-CV)  | 0.820 / 0.798        |
| Year prediction (GP, Matérn-5/2)       | MAE / R² (5-CV)       | **3.48 yr** / **0.353** |
| Year prediction (Ridge)                | MAE / R² (5-CV)       | 3.70 yr / 0.324      |
| Era (year thresholded @2015, GP RBF)   | accuracy / F₁ (5-CV)  | 0.728 / 0.716        |
| Era (year thresholded @2015, Logistic) | accuracy / F₁ (5-CV)  | 0.745 / 0.736        |

## Data sources

Three Periodic Plans and eight Budget Speeches were downloaded from
official Government of Nepal portals and verified mirrors. The full
list with URLs is in `data/sources.json` and `data/download_report.json`.

## Notes on academic integrity

The text of the report (`report/main.tex`) was drafted by the authors
for this coursework. All third-party text (plan/speech corpora,
prior literature) is cited. All numbers in the report come from
`data/processed/metrics.json`, which is regenerated deterministically
by `03_analyze.py` with seed 42.
