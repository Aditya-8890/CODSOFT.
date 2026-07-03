import os
import warnings
import numpy as np
import pandas as pd
import argparse
import matplotlib

# Add CLI flag to optionally show plots interactively when running the script
parser = argparse.ArgumentParser()
# Show plots by default; provide `--no-show` to disable interactive display
parser.add_argument("--no-show", action="store_false", dest="show",
                    default=True,
                    help="Do not display plots interactively (useful for headless runs)")
args, _ = parser.parse_known_args()

# Use non-interactive backend when not showing plots (e.g., headless environments)
if not args.show:
    matplotlib.use("Agg")

import subprocess
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.utils import resample

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("creditcard.csv")

df.info()


print(df.head(10) )

print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum().sum())
print("\nClass distribution:\n", df["Class"].value_counts())
print("\nAmount stats by class:\n", df.groupby("Class")["Amount"].describe())


genuine = df[df["Class"] == 0]
fraud   = df[df["Class"] == 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["Genuine", "Fraud"],
            [len(genuine), len(fraud)],
            color=["steelblue", "crimson"])
axes[0].set_title("Class Count")
axes[0].set_ylabel("Count")
axes[1].pie([len(genuine), len(fraud)],
            labels=["Genuine", "Fraud"],
            colors=["steelblue", "crimson"],
            autopct="%1.2f%%", startangle=90)
axes[1].set_title("Class Split")
plt.tight_layout()
path = "plots/01_class_distribution.png"
plt.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
genuine["Amount"].plot.kde(ax=axes[0], label="Genuine", color="steelblue")
fraud["Amount"].plot.kde(ax=axes[0], label="Fraud", color="crimson")
axes[0].set_xlim(-100, 2500)
axes[0].set_title("Amount KDE")
axes[0].legend()
axes[1].boxplot([genuine["Amount"].values, fraud["Amount"].values],
                labels=["Genuine", "Fraud"], patch_artist=True)
axes[1].set_title("Amount Boxplot")
plt.tight_layout()
path = "plots/02_amount_distribution.png"
plt.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(genuine["Time"], bins=50, color="steelblue", alpha=0.7, label="Genuine")
axes[0].set_title("Genuine - Time")
axes[1].hist(fraud["Time"], bins=40, color="crimson", alpha=0.7, label="Fraud")
axes[1].set_title("Fraud - Time")
plt.tight_layout()
path = "plots/03_time_distribution.png"
plt.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close()

plt.figure(figsize=(20, 16))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, linewidths=0.3, annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
path = "plots/04_correlation_heatmap.png"
plt.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close()

corr_with_class = df.corr()["Class"].drop("Class").sort_values()
top = pd.concat([corr_with_class.head(15), corr_with_class.tail(15)])
colors = ["crimson" if v < 0 else "steelblue" for v in top.values]
plt.figure(figsize=(10, 7))
plt.barh(top.index, top.values, color=colors)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Top Feature Correlations with Fraud")
plt.tight_layout()
path = "plots/05_feature_correlations.png"
plt.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close()

scaler = RobustScaler()
df["scaled_Time"]   = scaler.fit_transform(df[["Time"]])
df["scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
df.drop(columns=["Time", "Amount"], inplace=True)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df   = pd.concat([X_train, y_train], axis=1)
gen_train  = train_df[train_df["Class"] == 0]
frau_train = train_df[train_df["Class"] == 1]

gen_under     = resample(gen_train, replace=False, n_samples=len(frau_train), random_state=42)
under_df      = pd.concat([gen_under, frau_train])
X_train_under = under_df.drop(columns=["Class"])
y_train_under = under_df["Class"]

frau_over    = resample(frau_train, replace=True, n_samples=len(gen_train), random_state=42)
over_df      = pd.concat([gen_train, frau_over])
X_train_over = over_df.drop(columns=["Class"])
y_train_over = over_df["Class"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (title, y_s) in zip(axes, [
    ("Imbalanced",   y_train),
    ("Undersample",  y_train_under),
    ("Oversample",   y_train_over),
]):
    c = y_s.value_counts().sort_index()
    ax.bar(["Genuine", "Fraud"], c.values, color=["steelblue", "crimson"])
    ax.set_title(title)
plt.tight_layout()
path = "plots/06_resampling_comparison.png"
plt.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close()

configs = {
    "Imbalanced":  (X_train,       y_train),
    "Undersample": (X_train_under, y_train_under),
    "Oversample":  (X_train_over,  y_train_over),
}
colors = ["steelblue", "crimson", "mediumpurple"]
results = {}

cm_fig, cm_axes = plt.subplots(1, 3, figsize=(16, 5))
pr_fig, pr_axes = plt.subplots(1, 2, figsize=(13, 5))

for idx, (name, (X_tr, y_tr)) in enumerate(configs.items()):
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "Precision": report["1"]["precision"],
        "Recall":    report["1"]["recall"],
        "F1-Score":  report["1"]["f1-score"],
        "PR-AUC":    average_precision_score(y_test, y_prob),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=cm_axes[idx],
                xticklabels=["Genuine", "Fraud"],
                yticklabels=["Genuine", "Fraud"])
    cm_axes[idx].set_title(name)
    cm_axes[idx].set_xlabel("Predicted")
    cm_axes[idx].set_ylabel("Actual")

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_axes[0].plot(rec_arr, prec_arr, color=colors[idx], linewidth=2,
                    label=f"{name} (AUC={results[name]['PR-AUC']:.3f})")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr_axes[1].plot(fpr, tpr, color=colors[idx], linewidth=2,
                    label=f"{name} (AUC={results[name]['ROC-AUC']:.3f})")

cm_fig.suptitle("Confusion Matrices - Logistic Regression")
cm_fig.tight_layout()
path = "plots/07_confusion_matrices.png"
cm_fig.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close(cm_fig)

pr_axes[0].set_xlabel("Recall"); pr_axes[0].set_ylabel("Precision")
pr_axes[0].set_title("Precision-Recall Curve"); pr_axes[0].legend()
pr_axes[1].plot([0,1],[0,1],"k--")
pr_axes[1].set_xlabel("False Positive Rate"); pr_axes[1].set_ylabel("True Positive Rate")
pr_axes[1].set_title("ROC Curve"); pr_axes[1].legend()
pr_fig.tight_layout()
path = "plots/08_pr_roc_curves.png"
pr_fig.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close(pr_fig)

results_df = pd.DataFrame(results).T
print("\nSummary:\n", results_df)

metrics = ["Precision", "Recall", "F1-Score", "PR-AUC", "ROC-AUC"]
x = np.arange(len(metrics))
w = 0.25

plt.figure(figsize=(13, 5))
for i, (name, col) in enumerate(zip(configs.keys(), colors)):
    vals = [results[name][m] for m in metrics]
    plt.bar(x + (i - 1) * w, vals, w, label=name, color=col, alpha=0.85)
plt.xticks(x, metrics)
plt.ylim(0, 1.2)
plt.title("Metrics Comparison - Logistic Regression")
plt.legend()
plt.tight_layout()
path = "plots/09_metrics_comparison.png"
plt.savefig(path)
if args.show:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass
plt.close()

print("\nAll plots saved to plots/")
