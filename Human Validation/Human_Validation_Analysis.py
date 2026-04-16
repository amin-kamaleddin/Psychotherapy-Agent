# therapy_agent_analysis_updated.py
# ------------------------------------------------------------
# Publication-ready analysis of:
# "Evaluation of Therapy Conversations.xlsx"
# Sheet: "Renamed for Analysis"
#
# What this script does
# - Loads and cleans the data
# - Removes invalid ratings outside expected ranges
# - Produces:
#     1) Violin plot for Overall ratings (1-10), with mean ± SD shown above each violin
#     2) Heatmap for mean criteria ratings
#     3) Grouped bar plot for criteria means, with mean ± SD shown above each bar
# - Runs repeated-measures nonparametric statistics:
#     * Friedman omnibus tests
#     * Wilcoxon signed-rank pairwise post hoc tests
#     * Holm correction for multiple comparisons
#     * Kendall's W and rank-biserial effect sizes
# - Exports descriptive and inferential tables
#
# Install dependencies:
#   pip install pandas numpy matplotlib scipy openpyxl
#
# Run:
#   python therapy_agent_analysis_updated.py
# ------------------------------------------------------------

from pathlib import Path
import math
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# CONFIG
# ============================================================
FILE_PATH = "Evaluation of Therapy Conversations.xlsx"
SHEET_NAME = "Renamed for Analysis"
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AGENTS = ["Agent-1", "Agent-2", "Agent-3", "Agent-4"]
CRITERIA = ["Empathy", "Understanding", "Helpfulness", "Supportiveness"]

# Consistent agent colors requested by user
AGENT_COLORS = {
    "Agent-1": "#F28E2B",  # orange
    "Agent-2": "#59A14F",  # green
    "Agent-3": "#1F4E79",  # dark blue
    "Agent-4": "#76B7EB",  # light blue
}

OVERALL_RANGE = (1, 10)
CRITERIA_RANGE = (1, 5)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def mean_sd_text(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    mean_ = x.mean()
    sd_ = x.std(ddof=1)
    return f"{mean_:.2f} ± {sd_:.2f}"


def ci95(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    mean_ = x.mean()
    sd_ = x.std(ddof=1)
    se_ = sd_ / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    lower = mean_ - t_crit * se_
    upper = mean_ + t_crit * se_
    return lower, upper


def iqr(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return x.quantile(0.75) - x.quantile(0.25)


def kendalls_w_from_friedman(chi2_stat: float, n_subjects: int, k_conditions: int) -> float:
    if n_subjects <= 0 or k_conditions <= 1:
        return np.nan
    return chi2_stat / (n_subjects * (k_conditions - 1))


def holm_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def rank_biserial_from_wilcoxon(x, y):
    """
    Rank-biserial effect size for paired samples.
    Computed from absolute non-zero paired differences ranked with average ties.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    paired = pd.concat([x, y], axis=1).dropna()
    if paired.empty:
        return np.nan, 0

    d = paired.iloc[:, 0] - paired.iloc[:, 1]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 0

    abs_d = np.abs(d)
    ranks = stats.rankdata(abs_d, method="average")
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    rbc = (w_pos - w_neg) / (w_pos + w_neg)
    return rbc, n


def descriptive_row(outcome_name: str, agent_name: str, values: pd.Series):
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)
    mean_ = values.mean() if n > 0 else np.nan
    sd_ = values.std(ddof=1) if n > 1 else np.nan
    median_ = values.median() if n > 0 else np.nan
    iqr_ = iqr(values) if n > 0 else np.nan
    ci_low, ci_high = ci95(values) if n > 1 else (np.nan, np.nan)

    return {
        "Outcome": outcome_name,
        "Agent": agent_name,
        "N": n,
        "Mean": mean_,
        "SD": sd_,
        "Mean ± SD": f"{mean_:.2f} ± {sd_:.2f}" if n > 1 else "NA",
        "95% CI Lower": ci_low,
        "95% CI Upper": ci_high,
        "95% CI": f"[{ci_low:.2f}, {ci_high:.2f}]" if n > 1 else "NA",
        "Median": median_,
        "IQR": iqr_,
    }


def friedman_test_for_columns(df_: pd.DataFrame, columns, outcome_label):
    complete = df_[columns].dropna()
    n = len(complete)

    if n < 2:
        return {
            "Outcome": outcome_label,
            "N_complete_cases": n,
            "Statistic": np.nan,
            "df": len(columns) - 1,
            "p_value": np.nan,
            "Kendalls_W": np.nan,
        }

    arrays = [complete[col].values for col in columns]
    res = stats.friedmanchisquare(*arrays)
    w = kendalls_w_from_friedman(res.statistic, n, len(columns))

    return {
        "Outcome": outcome_label,
        "N_complete_cases": n,
        "Statistic": res.statistic,
        "df": len(columns) - 1,
        "p_value": res.pvalue,
        "Kendalls_W": w,
    }


def pairwise_wilcoxon_table(df_: pd.DataFrame, columns, outcome_label):
    rows = []
    pairs = list(itertools.combinations(columns, 2))
    raw_pvals = []

    for c1, c2 in pairs:
        sub = df_[[c1, c2]].dropna()
        n = len(sub)

        if n == 0:
            stat = np.nan
            pval = np.nan
            rbc = np.nan
            n_nonzero = 0
        else:
            diffs = sub[c1] - sub[c2]
            if np.all(diffs == 0):
                stat = 0.0
                pval = 1.0
                rbc = 0.0
                n_nonzero = 0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wil = stats.wilcoxon(
                        sub[c1],
                        sub[c2],
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                        mode="auto",
                    )
                stat = wil.statistic
                pval = wil.pvalue
                rbc, n_nonzero = rank_biserial_from_wilcoxon(sub[c1], sub[c2])

        raw_pvals.append(pval)

        rows.append({
            "Outcome": outcome_label,
            "Comparison": f"{c1} vs {c2}",
            "N_complete_pairs": n,
            "N_nonzero_differences": n_nonzero,
            "Wilcoxon_W": stat,
            "p_value_raw": pval,
            "Rank_biserial_r": rbc,
        })

    adjusted = holm_correction([1.0 if pd.isna(p) else p for p in raw_pvals])
    for row, adj in zip(rows, adjusted):
        row["p_value_holm"] = adj
        row["Significant_holm_0.05"] = bool(adj < 0.05)

    return pd.DataFrame(rows)


def add_bar_labels(ax, bars, means, sds, fontsize=9, offset_frac=0.03):
    ymax = ax.get_ylim()[1]
    offset = ymax * offset_frac
    for bar, m, s in zip(bars, means, sds):
        if pd.isna(m):
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = m + (0 if pd.isna(s) else s) + offset
        ax.text(
            x,
            y,
            f"{m:.2f}±{s:.2f}" if not pd.isna(s) else f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]

overall_cols = [f"{agent} Overall" for agent in AGENTS]
criteria_cols = [f"{agent} {criterion}" for agent in AGENTS for criterion in CRITERIA]
expected_cols = overall_cols + criteria_cols

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(
        "The following expected columns were not found in the worksheet:\n"
        + "\n".join(missing)
    )

# Keep a copy before cleaning for audit trail
df_original = df.copy()

# ============================================================
# CLEAN DATA
# ============================================================
for col in expected_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

invalid_rows = []

# Overall columns: valid 1-10
for col in overall_cols:
    invalid_mask = df[col].notna() & ~df[col].between(*OVERALL_RANGE)
    if invalid_mask.any():
        tmp = df.loc[invalid_mask, [col]].copy()
        tmp["Column"] = col
        tmp["Invalid_Value"] = df.loc[invalid_mask, col].values
        invalid_rows.append(tmp)
        df.loc[invalid_mask, col] = np.nan

# Criteria columns: valid 1-5
for col in criteria_cols:
    invalid_mask = df[col].notna() & ~df[col].between(*CRITERIA_RANGE)
    if invalid_mask.any():
        tmp = df.loc[invalid_mask, [col]].copy()
        tmp["Column"] = col
        tmp["Invalid_Value"] = df.loc[invalid_mask, col].values
        invalid_rows.append(tmp)
        df.loc[invalid_mask, col] = np.nan

if invalid_rows:
    invalid_df = pd.concat(invalid_rows, axis=0)
else:
    invalid_df = pd.DataFrame(columns=["Column", "Invalid_Value"])

invalid_df.to_csv(OUTPUT_DIR / "invalid_values_removed.csv", index=False)


# ============================================================
# DESCRIPTIVE STATISTICS TABLE
# ============================================================
desc_rows = []

# Overall
for agent in AGENTS:
    desc_rows.append(
        descriptive_row(
            outcome_name="Overall",
            agent_name=agent,
            values=df[f"{agent} Overall"]
        )
    )

# Each criterion
for criterion in CRITERIA:
    for agent in AGENTS:
        desc_rows.append(
            descriptive_row(
                outcome_name=criterion,
                agent_name=agent,
                values=df[f"{agent} {criterion}"]
            )
        )

descriptive_df = pd.DataFrame(desc_rows)
descriptive_df.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index=False)


# ============================================================
# OMNIBUS FRIEDMAN TESTS
# ============================================================
omnibus_rows = []

# Overall
omnibus_rows.append(
    friedman_test_for_columns(df, overall_cols, "Overall")
)

# Criteria
for criterion in CRITERIA:
    cols = [f"{agent} {criterion}" for agent in AGENTS]
    omnibus_rows.append(
        friedman_test_for_columns(df, cols, criterion)
    )

omnibus_df = pd.DataFrame(omnibus_rows)
omnibus_df.to_csv(OUTPUT_DIR / "omnibus_friedman_tests.csv", index=False)


# ============================================================
# PAIRWISE WILCOXON POST HOC TESTS
# ============================================================
pairwise_tables = []

# Overall
pairwise_tables.append(
    pairwise_wilcoxon_table(df, overall_cols, "Overall")
)

# Criteria
for criterion in CRITERIA:
    cols = [f"{agent} {criterion}" for agent in AGENTS]
    pairwise_tables.append(
        pairwise_wilcoxon_table(df, cols, criterion)
    )

pairwise_df = pd.concat(pairwise_tables, axis=0, ignore_index=True)
pairwise_df.to_csv(OUTPUT_DIR / "pairwise_wilcoxon_posthoc.csv", index=False)


# ============================================================
# FIGURE 1: VIOLIN PLOT FOR OVERALL SCORES
# - requested instead of boxplot
# - with mean ± SD on top of each violin
# - with requested colors
# ============================================================
overall_data = [df[f"{agent} Overall"].dropna().values for agent in AGENTS]
overall_means = [np.nanmean(x) if len(x) > 0 else np.nan for x in overall_data]
overall_sds = [np.nanstd(x, ddof=1) if len(x) > 1 else np.nan for x in overall_data]

fig, ax = plt.subplots(figsize=(10, 6))
parts = ax.violinplot(
    overall_data,
    positions=np.arange(1, len(AGENTS) + 1),
    showmeans=False,
    showmedians=True,
    showextrema=False,
)

for body, agent in zip(parts["bodies"], AGENTS):
    body.set_facecolor(AGENT_COLORS[agent])
    body.set_edgecolor("black")
    body.set_alpha(0.75)
    body.set_linewidth(1.0)

if "cmedians" in parts:
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.2)

# Overlay mean points and SD error bars
for i, agent in enumerate(AGENTS, start=1):
    m = overall_means[i - 1]
    s = overall_sds[i - 1]
    if pd.isna(m):
        continue

    ax.errorbar(
        i,
        m,
        yerr=s if not pd.isna(s) else None,
        fmt="o",
        color="black",
        capsize=5,
        markersize=6,
        linewidth=1.5,
        zorder=3,
    )

    y_text = min(OVERALL_RANGE[1] + 0.15, m + (0 if pd.isna(s) else s) + 0.25)
    ax.text(
        i,
        y_text,
        f"{m:.2f}±{s:.2f}" if not pd.isna(s) else f"{m:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_xticks(np.arange(1, len(AGENTS) + 1))
ax.set_xticklabels(AGENTS)
ax.set_ylabel("Overall rating (1–10)")
ax.set_title("Overall Ratings by Agent")
ax.set_ylim(1, 10.6)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_1_overall_violin.svg", format="svg", bbox_inches="tight", dpi=300)
plt.close()


# ============================================================
# FIGURE 2: HEATMAP OF MEAN CRITERIA SCORES
# ============================================================
criteria_means = pd.DataFrame(index=AGENTS, columns=CRITERIA, dtype=float)

for agent in AGENTS:
    for criterion in CRITERIA:
        criteria_means.loc[agent, criterion] = df[f"{agent} {criterion}"].mean()

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(criteria_means.values, aspect="auto")

ax.set_xticks(range(len(CRITERIA)))
ax.set_xticklabels(CRITERIA, rotation=30, ha="right")
ax.set_yticks(range(len(AGENTS)))
ax.set_yticklabels(AGENTS)
ax.set_title("Mean Criteria Ratings by Agent")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Mean rating (1–5)")

for i in range(len(AGENTS)):
    for j in range(len(CRITERIA)):
        value = criteria_means.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_2_criteria_heatmap.svg", format="svg", bbox_inches="tight", dpi=300)
plt.close()


# ============================================================
# FIGURE 3: GROUPED BAR PLOT FOR CRITERIA SCORES
# - with mean ± SD on top of each bar
# - with requested colors
# ============================================================
criteria_means_bar = pd.DataFrame(index=AGENTS, columns=CRITERIA, dtype=float)
criteria_sds_bar = pd.DataFrame(index=AGENTS, columns=CRITERIA, dtype=float)

for agent in AGENTS:
    for criterion in CRITERIA:
        vals = df[f"{agent} {criterion}"].dropna()
        criteria_means_bar.loc[agent, criterion] = vals.mean()
        criteria_sds_bar.loc[agent, criterion] = vals.std(ddof=1) if len(vals) > 1 else np.nan

x = np.arange(len(CRITERIA))
width = 0.18

fig, ax = plt.subplots(figsize=(11, 6))

all_bar_tops = []

for i, agent in enumerate(AGENTS):
    means = criteria_means_bar.loc[agent].values.astype(float)
    sds = criteria_sds_bar.loc[agent].values.astype(float)

    bars = ax.bar(
        x + i * width,
        means,
        width=width,
        yerr=sds,
        capsize=4,
        label=agent,
        color=AGENT_COLORS[agent],
        edgecolor="black",
        linewidth=0.8,
    )

    # Track tops for y-limit setting
    for m, s in zip(means, sds):
        if pd.isna(m):
            continue
        all_bar_tops.append(m + (0 if pd.isna(s) else s))

    add_bar_labels(ax, bars, means, sds, fontsize=8, offset_frac=0.02)

center_shift = width * (len(AGENTS) - 1) / 2
ax.set_xticks(x + center_shift)
ax.set_xticklabels(CRITERIA, rotation=20)
ax.set_ylabel("Mean rating (1–5)")
ax.set_title("Mean Criteria Ratings by Agent")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)

if all_bar_tops:
    ymax = min(5.45, max(all_bar_tops) + 0.45)
else:
    ymax = 5.2
ax.set_ylim(0, ymax)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_3_criteria_grouped_bars.svg", format="svg", bbox_inches="tight", dpi=300)
plt.close()


# ============================================================
# EXCEL REPORT
# ============================================================
excel_path = OUTPUT_DIR / "analysis_tables.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    descriptive_df.to_excel(writer, sheet_name="Descriptive_Stats", index=False)
    omnibus_df.to_excel(writer, sheet_name="Friedman_Tests", index=False)
    pairwise_df.to_excel(writer, sheet_name="Pairwise_Wilcoxon", index=False)
    invalid_df.to_excel(writer, sheet_name="Invalid_Values_Removed", index=False)

# ============================================================
# REPORTING NOTE
# ============================================================
report_text = []
report_text.append("Publication-oriented analysis completed.\n")
report_text.append("Design: repeated-measures / within-subject ratings across four agents.\n")
report_text.append("Primary omnibus test: Friedman test.\n")
report_text.append("Post hoc comparisons: paired Wilcoxon signed-rank tests with Holm correction.\n")
report_text.append("Effect sizes: Kendall's W (omnibus) and rank-biserial correlation r (pairwise).\n")
report_text.append("Figures saved as SVG.\n")
report_text.append("\nFiles created:\n")
report_text.append("- figure_1_overall_violin.svg\n")
report_text.append("- figure_2_criteria_heatmap.svg\n")
report_text.append("- figure_3_criteria_grouped_bars.svg\n")
report_text.append("- descriptive_statistics.csv\n")
report_text.append("- omnibus_friedman_tests.csv\n")
report_text.append("- pairwise_wilcoxon_posthoc.csv\n")
report_text.append("- analysis_tables.xlsx\n")
report_text.append("- invalid_values_removed.csv\n")

with open(OUTPUT_DIR / "reporting_note.txt", "w", encoding="utf-8") as f:
    f.writelines(report_text)


# ============================================================
# CONSOLE SUMMARY
# ============================================================
print("\nAnalysis completed successfully.")
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}\n")

print("Descriptive statistics preview:")
print(descriptive_df.round(3).head(12).to_string(index=False))

print("\nOmnibus Friedman tests:")
print(omnibus_df.round(4).to_string(index=False))

print("\nPairwise Wilcoxon post hoc preview:")
print(pairwise_df.round(4).head(12).to_string(index=False))