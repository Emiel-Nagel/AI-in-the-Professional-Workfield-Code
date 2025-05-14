import numpy as np
import pandas as pd
from rpca.rpca import RobustPCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pyrpca


def fetch_data():
    # Load the dataset
    dataset = pd.read_csv('Code_Emiel/Dataset/refData_obf.csv')

    cols = dataset.columns
    features = dataset.loc[:, cols[1:]].values              # Exclude the sample names (index 0) from features
    mask = ~np.isnan(features).any(axis=1)
    features = features[mask]                               # Drop samples with some missing values

    return dataset, features

def plot(rpca_data, pyrpca_data, title, xlabel, ylabel, extra_plots=[]):
    if extra_plots is None:
        extra_plots = []

    fig, ax1 = plt.subplots(figsize=(10, 6))
    idx = np.arange(len(rpca_data))

    # ── RPCA on ax1 (blue) ────────────────────────────────────────────────────
    sns.scatterplot(x=idx, y=rpca_data, ax=ax1, label="RPCA", color="tab:blue")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel + " (RPCA)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # ── pyrpca on a twin axis (red) ──────────────────────────────────────────
    ax2 = ax1.twinx()
    sns.scatterplot(
        x=idx,
        y=pyrpca_data,
        ax=ax2,
        label="pyrpca",
        marker="x",
        color="tab:red",
    )
    ax2.set_ylabel(ylabel + " (pyrpca)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # ── Extra decorations provided by caller ────────────────────────────────
    for fn in extra_plots:
        fn(ax1)          # pass the primary axis so callers can draw on it

    # Combine the two legends
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1 + h2, l1 + l2, loc="best")

    ax1.set_title(title)
    fig.tight_layout()
    plt.show()

def norm_to_prob(x: np.ndarray) -> np.ndarray:
    """Convert a non‑negative 1‑D array to a probability distribution."""
    total = x.sum()
    if total == 0:
        return np.full_like(x, 1 / len(x), dtype=float)
    return x / total

def rpca_pipeline(alpha, x):
    rpca = RobustPCA(max_iter=10000, tol=1e-6).fit(x)           # robust decomposition
    L, S = rpca.L_, rpca.S_           # robust decomposition
    
    print("Estimated rank(L):", np.linalg.matrix_rank(L))
    print(L.shape)
    print(S.shape)

    row_outlier_score = np.abs(S).sum(axis=1)
    print("Row outlier scores:", row_outlier_score)
    row_probs = norm_to_prob(row_outlier_score)

    cutoff = np.percentile(row_outlier_score, 100 * (1 - alpha))
    print(f"{100*(1-alpha):.1f}th percentile (cutoff):", cutoff)

    suspect_rows = np.where(row_outlier_score > cutoff)[0]
    print(f"Top-{alpha} % outlier rows:", suspect_rows)

    return row_outlier_score, row_probs, suspect_rows

    plot(row_outlier_score, title="Rpca Row Outlier Scores", xlabel="Row index", ylabel="‖S‖₁ per row",
        extra_plots=[lambda: plt.axhline(cutoff, linestyle="--", label=f"Cutoff={cutoff:.2f}")]
    )

def pyrpca_pipeline(alpha, x):
    sparsity = lambda x: 1/np.sqrt(max(x.shape))
    print("Sparsity:", sparsity(x))
    L_hat, S_hat = pyrpca.rpca_pcp_ialm(x, sparsity(x))

    print("Estimated rank(L):", np.linalg.matrix_rank(L_hat))
    print(L_hat.shape)
    print(S_hat.shape)
    
    row_outlier_score = np.abs(S_hat).sum(axis=1)
    print("Row outlier scores:", row_outlier_score)
    row_probs = norm_to_prob(row_outlier_score)

    cutoff = np.percentile(row_outlier_score, 100 * (1 - alpha))
    print(f"{100*(1-alpha):.1f}th percentile (cutoff):", cutoff)

    suspect_rows = np.where(row_outlier_score > cutoff)[0]
    print(f"Top-{alpha} % outlier rows:", suspect_rows)

    return row_outlier_score, row_probs, suspect_rows

    plot(row_outlier_score, title="Pyrpca Row Outlier Scores", xlabel="Row index", ylabel="‖S‖₁ per row",
        extra_plots=[lambda: plt.axhline(cutoff, linestyle="--", label=f"Cutoff={cutoff:.2f}")]
    )

def export_suspects_to_excel(
    dataset: pd.DataFrame,
    rpca_suspects: np.ndarray,
    pyrpca_suspects: np.ndarray,
    file_path: str = "Code_Emiel/Output/RPCA suspect_samples.xlsx",
):
    """Write the suspect rows into an Excel workbook.

    Parameters
    ----------
    dataset : DataFrame
        The full, clean dataset (after NaN removal).
    rpca_suspects : array-like of int
        Row indices flagged by the RPCA pipeline.
    pyrpca_suspects : array-like of int
        Row indices flagged by the pyrpca pipeline.
    file_path : str, optional
        Destination Excel file (default: ``suspect_rows.xlsx``).
    """
    rpca_df = dataset.iloc[rpca_suspects].copy()
    pyrpca_df = dataset.iloc[pyrpca_suspects].copy()
    combined_df = dataset.iloc[np.union1d(rpca_suspects, pyrpca_suspects)].copy()

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        rpca_df.to_excel(writer, sheet_name="RPCA", index=False)
        pyrpca_df.to_excel(writer, sheet_name="pyrpca", index=False)
        combined_df.to_excel(writer, sheet_name="Combined", index=False)

    print(f"✓ Suspect rows exported to '{file_path}'.")

def main():
    alpha = 0.05
    dataset, features = fetch_data()
    print(features.shape)

    x = StandardScaler(with_mean=True, with_std=True).fit_transform(features)
    rpca_values = rpca_pipeline(alpha, x)
    pyrpca_values = pyrpca_pipeline(alpha, x)
    plot(rpca_values[0], pyrpca_values[0],
        title="Row Outlier Scores",
        xlabel="Row index",
        ylabel="‖S‖₁ per row",
        # extra_plots=[lambda ax: ax.axhline(cutoff, ls="--", color="k")]
    )
    export_suspects_to_excel(dataset, rpca_values[2], pyrpca_values[2])

    # rpca = RobustPCA(max_iter=10000, tol=1e-6).fit(x)           # robust decomposition
    # L, S = rpca.L_, rpca.S_           # robust decomposition

    # print("Estimated rank(L):", np.linalg.matrix_rank(L))
    # print(L.shape)
    # print(S.shape)

    # row_outlier_score = np.abs(S).sum(axis=1)
    # print(row_outlier_score)

    # cutoff = np.percentile(row_outlier_score, 100 * (1 - alpha))
    # print(f"{100*(1-alpha):.1f}th percentile (cutoff):", cutoff)

    # suspect_rows = np.where(row_outlier_score > cutoff)[0]
    # print(f"Top-{alpha} % outlier rows:", suspect_rows)

    # plot(row_outlier_score, title="Row Outlier Scores", xlabel="Row index", ylabel="‖S‖₁ per row",
    #     extra_plots=[lambda: plt.axhline(cutoff, linestyle="--", label=f"Cutoff={cutoff:.2f}")]
    # )
    return "Hi"
