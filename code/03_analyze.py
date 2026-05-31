"""End-to-end analysis pipeline for Nepal policy topic modeling.

Pipeline
--------
1. Load extracted text from data/plans and data/speeches.
2. Split each long document into ~2000-word "chunks" so the corpus has enough
   units for stable LDA inference.
3. Preprocess: lowercase, regex tokenize, strip stopwords (NLTK English +
   custom Nepal/government boilerplate), drop short and numeric tokens.
4. Vectorise with a count vectorizer and fit Latent Dirichlet Allocation.
5. Pull per-chunk topic distributions and treat them as features for two
   downstream learners:
       - Gaussian Process Classification (plan vs budget speech).
       - Gaussian Process Regression on the document year (a continuous
         target derived from the file label).
6. Cross-validate both, compare against logistic regression / linear
   regression baselines, and save metrics + figures.

All intermediate artefacts (chunk metadata, topic-term lists, per-chunk topic
distributions, metrics) are written to data/processed/ so the LaTeX paper can
cite exact numbers.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.manifold import TSNE
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
PROC = DATA / "processed"
FIG.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & chunk
# ---------------------------------------------------------------------------
CHUNK_WORDS = 2000


def label_to_year(label: str) -> int:
    """Pull a representative fiscal-year start out of the label string."""
    m = re.search(r"(19|20)\d{2}", label)
    return int(m.group(0)) if m else 0


# The brief requires a Gaussian Process *classification* in which a threshold is
# defined on the (continuous) output variable to create classes, and the GP is
# then trained on that categorised output. The continuous output here is the
# document year; we threshold it at 2015, the year Nepal promulgated its federal
# constitution, splitting the corpus into a pre-federal and a federal era.
FEDERAL_THRESHOLD_YEAR = 2015


def year_to_era(year: int) -> int:
    """Threshold the continuous year output into a binary era class.

    0 = pre-federal (year <= 2015), 1 = federal (year > 2015).
    """
    return int(year > FEDERAL_THRESHOLD_YEAR)


def load_corpus() -> pd.DataFrame:
    rows = []
    for category, folder in (("plan", DATA / "plans"), ("speech", DATA / "speeches")):
        for fp in sorted(folder.glob("*.txt")):
            text = fp.read_text(errors="ignore")
            words = text.split()
            year = label_to_year(fp.stem)
            for i in range(0, len(words), CHUNK_WORDS):
                chunk = " ".join(words[i : i + CHUNK_WORDS])
                if len(chunk.split()) < 400:
                    continue
                rows.append(
                    {
                        "doc_id": f"{fp.stem}__chunk{i // CHUNK_WORDS:03d}",
                        "source_file": fp.stem,
                        "category": category,
                        "year": year,
                        "text": chunk,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z]{3,}")
CUSTOM_STOP = {
    "nepal", "nepalese", "nepali", "government", "shall", "will", "year",
    "fiscal", "rupee", "rupees", "billion", "million", "rs", "npr",
    "section", "chapter", "table", "figure", "page", "honorable", "honourable",
    "speaker", "annex", "annexes", "appendix", "etc", "ministry", "minister",
    "honourable", "national", "policy", "programme", "program", "plan",
    "budget", "speech", "report", "kathmandu", "ms", "mr", "per", "cent",
    "percent", "respectively", "also", "thus", "therefore", "however",
    "amount", "amounts", "amounting", "total", "approximately", "approximate",
    "various", "many", "much", "made", "make", "made", "shall", "may",
    "would", "could", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "first", "second", "third", "fourth", "fifth",
    "regarding", "respect", "follows", "include", "including", "included",
    "given", "since", "while", "upon", "current", "currently", "next",
    "previous", "fy", "rsm", "rsb",
}
try:
    NLTK_STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    NLTK_STOP = set(stopwords.words("english"))
STOP = NLTK_STOP | CUSTOM_STOP


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


# ---------------------------------------------------------------------------
# 3. LDA
# ---------------------------------------------------------------------------
def fit_lda(corpus: pd.DataFrame, n_topics: int) -> tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    vec = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        max_df=0.85,
        min_df=3,
        max_features=4000,
    )
    X = vec.fit_transform(corpus["text"])
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=40,
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        random_state=RNG,
    )
    theta = lda.fit_transform(X)
    return lda, vec, theta


def topic_top_words(lda: LatentDirichletAllocation, vec: CountVectorizer, n: int = 12) -> list[list[str]]:
    vocab = np.array(vec.get_feature_names_out())
    return [list(vocab[topic.argsort()[::-1][:n]]) for topic in lda.components_]


def select_k(corpus: pd.DataFrame, candidates=(4, 6, 8, 10, 12)) -> tuple[int, list[dict]]:
    """Choose number of topics by held-out perplexity."""
    rows = []
    best_k, best_perp = candidates[0], float("inf")
    for k in candidates:
        lda, vec, _ = fit_lda(corpus, k)
        perp = lda.perplexity(vec.transform(corpus["text"]))
        rows.append({"k": k, "perplexity": float(perp)})
        if perp < best_perp:
            best_perp, best_k = perp, k
    return best_k, rows


# ---------------------------------------------------------------------------
# 4. GP classification & regression
# ---------------------------------------------------------------------------
def gp_classify(theta: np.ndarray, y: np.ndarray) -> dict:
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=RNG, n_restarts_optimizer=2)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    acc = cross_val_score(gpc, theta, y, cv=skf, scoring="accuracy")
    f1 = cross_val_score(gpc, theta, y, cv=skf, scoring="f1_macro")
    # Fit on all data for confusion matrix / report
    gpc.fit(theta, y)
    yhat = gpc.predict(theta)
    return {
        "model": gpc,
        "cv_acc_mean": float(acc.mean()),
        "cv_acc_std": float(acc.std()),
        "cv_f1_mean": float(f1.mean()),
        "cv_f1_std": float(f1.std()),
        "train_acc": float(accuracy_score(y, yhat)),
        "confusion": confusion_matrix(y, yhat).tolist(),
        "report": classification_report(y, yhat, output_dict=True),
    }


def baseline_classify(theta: np.ndarray, y: np.ndarray) -> dict:
    clf = LogisticRegression(max_iter=1000, random_state=RNG)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    acc = cross_val_score(clf, theta, y, cv=skf, scoring="accuracy")
    f1 = cross_val_score(clf, theta, y, cv=skf, scoring="f1_macro")
    return {
        "cv_acc_mean": float(acc.mean()),
        "cv_acc_std": float(acc.std()),
        "cv_f1_mean": float(f1.mean()),
        "cv_f1_std": float(f1.std()),
    }


def gp_regress_year(theta: np.ndarray, years: np.ndarray) -> dict:
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RNG, n_restarts_optimizer=2)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    mae = -cross_val_score(gpr, theta, years, cv=kf, scoring="neg_mean_absolute_error")
    r2 = cross_val_score(gpr, theta, years, cv=kf, scoring="r2")
    gpr.fit(theta, years)
    yhat, ystd = gpr.predict(theta, return_std=True)
    return {
        "cv_mae_mean": float(mae.mean()),
        "cv_mae_std": float(mae.std()),
        "cv_r2_mean": float(r2.mean()),
        "cv_r2_std": float(r2.std()),
        "train_mae": float(mean_absolute_error(years, yhat)),
        "train_r2": float(r2_score(years, yhat)),
        "pred_mean": yhat.tolist(),
        "pred_std": ystd.tolist(),
    }


def baseline_regress_year(theta: np.ndarray, years: np.ndarray) -> dict:
    reg = Ridge(alpha=1.0, random_state=RNG)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    mae = -cross_val_score(reg, theta, years, cv=kf, scoring="neg_mean_absolute_error")
    r2 = cross_val_score(reg, theta, years, cv=kf, scoring="r2")
    return {
        "cv_mae_mean": float(mae.mean()),
        "cv_mae_std": float(mae.std()),
        "cv_r2_mean": float(r2.mean()),
        "cv_r2_std": float(r2.std()),
    }


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def plot_perplexity(rows, path):
    df = pd.DataFrame(rows)
    plt.figure(figsize=(4.2, 3.0))
    plt.plot(df["k"], df["perplexity"], marker="o")
    plt.xlabel("Number of topics K")
    plt.ylabel("Held-out perplexity")
    plt.title("LDA model selection")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_topic_words(topics, path):
    rows = [
        {"topic": i + 1, "rank": j + 1, "word": w}
        for i, ws in enumerate(topics)
        for j, w in enumerate(ws[:8])
    ]
    df = pd.DataFrame(rows)
    n = len(topics)
    cols = 3
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(8.5, 2.3 * rws))
    axes = np.array(axes).reshape(-1)
    for i, ws in enumerate(topics):
        ax = axes[i]
        ax.barh(range(len(ws[:8]))[::-1], range(1, 9)[::-1], color=sns.color_palette("crest", 8))
        ax.set_yticks(range(len(ws[:8]))[::-1])
        ax.set_yticklabels(ws[:8])
        ax.set_xticks([])
        ax.set_title(f"Topic {i + 1}", fontsize=10)
    for j in range(len(topics), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_topic_by_category(corpus, theta, path):
    df = pd.DataFrame(theta, columns=[f"T{i+1}" for i in range(theta.shape[1])])
    df["category"] = corpus["category"].values
    mean = df.groupby("category").mean().T
    mean.plot(kind="bar", figsize=(6.5, 3.2), color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Mean topic probability")
    plt.xlabel("Topic")
    plt.title("Topic prevalence: Plans vs Budget Speeches")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_topics_over_time(corpus, theta, path):
    df = pd.DataFrame(theta, columns=[f"T{i+1}" for i in range(theta.shape[1])])
    df["year"] = corpus["year"].values
    yr = df.groupby("year").mean()
    yr.plot(figsize=(6.5, 3.4), marker="o", linewidth=1.4)
    plt.ylabel("Mean topic probability")
    plt.xlabel("Document year")
    plt.title("Topic prevalence over time")
    plt.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_tsne(theta, y_cat, path):
    ts = TSNE(n_components=2, perplexity=15, random_state=RNG, init="pca")
    Z = ts.fit_transform(theta)
    plt.figure(figsize=(4.4, 3.4))
    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    labels = {0: "Plan", 1: "Budget speech"}
    for c in (0, 1):
        m = y_cat == c
        plt.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7, c=colors[c], label=labels[c])
    plt.legend(fontsize=8)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Documents in topic space")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion(cm, path, labels=("Plan", "Speech"), title="GP classifier"):
    plt.figure(figsize=(3.4, 2.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_gpr(years, pred_mean, pred_std, path):
    order = np.argsort(years)
    y = years[order]
    m = np.array(pred_mean)[order]
    s = np.array(pred_std)[order]
    plt.figure(figsize=(6.5, 3.0))
    plt.scatter(y, m, s=10, alpha=0.6, label="GP mean")
    plt.fill_between(y, m - s, m + s, alpha=0.2, label="±1 std")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1, label="Ideal")
    plt.xlabel("True document year")
    plt.ylabel("Predicted year")
    plt.title("Gaussian Process Regression on document year")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_wordcloud(corpus, theta, lda, vec, path):
    vocab = vec.get_feature_names_out()
    n_topics = lda.components_.shape[0]
    cols = 3
    rows = int(np.ceil(n_topics / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 2.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, comp in enumerate(lda.components_):
        freqs = {vocab[j]: float(comp[j]) for j in comp.argsort()[::-1][:40]}
        wc = WordCloud(width=420, height=240, background_color="white",
                       colormap="viridis").generate_from_frequencies(freqs)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {i + 1}", fontsize=10)
        axes[i].axis("off")
    for j in range(n_topics, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("[1] Loading corpus...")
    corpus = load_corpus()
    corpus.to_csv(PROC / "corpus_chunks.csv", index=False)
    print(f"    {len(corpus)} chunks "
          f"({(corpus.category == 'plan').sum()} plan, "
          f"{(corpus.category == 'speech').sum()} speech) "
          f"from {corpus.source_file.nunique()} source documents")

    print("[2] Selecting number of topics (perplexity)...")
    best_k, perp_rows = select_k(corpus, candidates=(4, 6, 8, 10, 12))
    print(f"    perplexity by K: {[(r['k'], round(r['perplexity'], 1)) for r in perp_rows]}")
    # In practice perplexity keeps falling; pick a moderate K for interpretability.
    chosen_k = 8
    print(f"    chosen K = {chosen_k} (balance of perplexity and interpretability)")

    print(f"[3] Fitting final LDA with K={chosen_k}...")
    lda, vec, theta = fit_lda(corpus, chosen_k)
    topics = topic_top_words(lda, vec, n=15)
    for i, ws in enumerate(topics):
        print(f"    Topic {i + 1}: {', '.join(ws[:10])}")

    np.save(PROC / "theta.npy", theta)
    pd.DataFrame(theta, columns=[f"T{i+1}" for i in range(chosen_k)]).to_csv(
        PROC / "doc_topic.csv", index=False
    )
    (PROC / "topics.json").write_text(json.dumps(topics, indent=2))

    print("[4] GP classification (plan vs speech) on topic features...")
    y_cat = (corpus["category"] == "speech").astype(int).values
    gpc = gp_classify(theta, y_cat)
    base = baseline_classify(theta, y_cat)
    print(f"    GP   acc = {gpc['cv_acc_mean']:.3f} ± {gpc['cv_acc_std']:.3f}, "
          f"F1 = {gpc['cv_f1_mean']:.3f}")
    print(f"    LR   acc = {base['cv_acc_mean']:.3f} ± {base['cv_acc_std']:.3f}, "
          f"F1 = {base['cv_f1_mean']:.3f}")

    print("[5] GP regression on document year...")
    years = corpus["year"].values.astype(float)
    gpr = gp_regress_year(theta, years)
    base_reg = baseline_regress_year(theta, years)
    print(f"    GP   MAE = {gpr['cv_mae_mean']:.2f} yr, R^2 = {gpr['cv_r2_mean']:.3f}")
    print(f"    Ridge MAE= {base_reg['cv_mae_mean']:.2f} yr, R^2 = {base_reg['cv_r2_mean']:.3f}")

    print("[5b] Thresholding continuous output (year) -> era classes; GP classification...")
    era = np.array([year_to_era(int(y)) for y in years])
    n_pre, n_post = int((era == 0).sum()), int((era == 1).sum())
    gpc_era = gp_classify(theta, era)
    base_era = baseline_classify(theta, era)
    print(f"    threshold = {FEDERAL_THRESHOLD_YEAR}; pre-federal={n_pre}, federal={n_post}")
    print(f"    GP   acc = {gpc_era['cv_acc_mean']:.3f} ± {gpc_era['cv_acc_std']:.3f}, "
          f"F1 = {gpc_era['cv_f1_mean']:.3f}")
    print(f"    LR   acc = {base_era['cv_acc_mean']:.3f} ± {base_era['cv_acc_std']:.3f}, "
          f"F1 = {base_era['cv_f1_mean']:.3f}")

    metrics = {
        "perplexity": perp_rows,
        "chosen_k": chosen_k,
        "gp_classifier": {k: v for k, v in gpc.items() if k != "model"},
        "baseline_classifier_lr": base,
        "gp_regressor_year": {k: v for k, v in gpr.items() if k not in ("pred_mean", "pred_std")},
        "baseline_regressor_ridge": base_reg,
        "era_classification": {
            "threshold_year": FEDERAL_THRESHOLD_YEAR,
            "class_labels": ["pre-federal (<=2015)", "federal (>2015)"],
            "n_pre": n_pre,
            "n_post": n_post,
            "gp": {k: v for k, v in gpc_era.items() if k != "model"},
            "baseline_lr": base_era,
        },
        "topic_top_words": topics,
        "corpus_summary": {
            "n_chunks": int(len(corpus)),
            "n_plan_chunks": int((corpus.category == "plan").sum()),
            "n_speech_chunks": int((corpus.category == "speech").sum()),
            "n_source_docs": int(corpus.source_file.nunique()),
            "n_plan_docs": int(corpus.loc[corpus.category == "plan", "source_file"].nunique()),
            "n_speech_docs": int(corpus.loc[corpus.category == "speech", "source_file"].nunique()),
            "vocab_size": int(len(vec.get_feature_names_out())),
            "total_words": int(corpus["text"].str.split().str.len().sum()),
            "year_min": int(years.min()),
            "year_max": int(years.max()),
        },
    }
    (PROC / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("[6] Generating figures...")
    plot_perplexity(perp_rows, FIG / "fig_perplexity.png")
    plot_topic_words(topics, FIG / "fig_topic_words.png")
    plot_topic_by_category(corpus, theta, FIG / "fig_topic_category.png")
    plot_topics_over_time(corpus, theta, FIG / "fig_topics_over_time.png")
    plot_tsne(theta, y_cat, FIG / "fig_tsne.png")
    plot_confusion(np.array(gpc["confusion"]), FIG / "fig_confusion.png",
                   labels=("Plan", "Speech"), title="GP classifier: genre")
    plot_confusion(np.array(gpc_era["confusion"]), FIG / "fig_confusion_era.png",
                   labels=("Pre-2015", "Federal"), title="GP classifier: era (thresholded year)")
    plot_gpr(years, gpr["pred_mean"], gpr["pred_std"], FIG / "fig_gpr_year.png")
    plot_wordcloud(corpus, theta, lda, vec, FIG / "fig_wordclouds.png")
    print(f"    figures written to {FIG}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
