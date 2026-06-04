"""
🔍 PREPROCESSING EXPLAINABILITY MODULE
========================================
Answers "WHY" at the data level — before any model is trained.

Techniques:
  1. Mutual Information (MI) between each factor and anomaly labels, per regime
  2. Correlation-based factor clustering (redundancy analysis)
  3. Factor discriminability heatmap (KS-statistic per factor per regime)

This module is the first stage of the End-to-End XAI pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ks_2samp
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def compute_mutual_information_per_regime(
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    feature_names: list = None,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute mutual information between each factor and anomaly label,
    stratified by volatility regime.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Factor matrix.
    y : np.ndarray, shape (N,)
        Binary anomaly labels (0/1).
    regimes : np.ndarray, shape (N,)
        Regime labels (0=Low, 1=Med, 2=High).
    feature_names : list, optional
        Names of the D features.
    n_neighbors : int
        KNN parameter for MI estimation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Rows = features, Columns = ['MI_Low', 'MI_Med', 'MI_High', 'MI_Global'].
    """
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(X.shape[1])]

    regime_names = {0: 'MI_Low', 1: 'MI_Med', 2: 'MI_High'}
    results = {}

    # Global MI
    mi_global = mutual_info_classif(
        X, y, n_neighbors=n_neighbors, random_state=random_state
    )
    results['MI_Global'] = mi_global

    # Per-regime MI
    for r, col_name in regime_names.items():
        mask = regimes == r
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            results[col_name] = np.zeros(X.shape[1])
            continue

        mi_r = mutual_info_classif(
            X[mask], y[mask],
            n_neighbors=min(n_neighbors, mask.sum() // 3),
            random_state=random_state
        )
        results[col_name] = mi_r

    df = pd.DataFrame(results, index=feature_names)
    df = df[['MI_Low', 'MI_Med', 'MI_High', 'MI_Global']]

    # Add rank columns
    for col in df.columns:
        df[f'{col}_rank'] = df[col].rank(ascending=False).astype(int)

    return df


def compute_factor_discriminability(
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    feature_names: list = None,
) -> pd.DataFrame:
    """Compute KS-statistic measuring distributional separation between
    normal and anomaly samples for each factor, per regime.

    A high KS-statistic means the factor strongly distinguishes
    anomalies from normal observations in that regime.

    Returns
    -------
    pd.DataFrame
        Rows = features, Columns = ['KS_Low', 'KS_Med', 'KS_High', 'KS_Global',
                                     'pval_Low', 'pval_Med', 'pval_High', 'pval_Global'].
    """
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(X.shape[1])]

    regime_names = {0: 'Low', 1: 'Med', 2: 'High'}
    results = {}

    # Global KS
    for j, fname in enumerate(feature_names):
        normal = X[y == 0, j]
        anomaly = X[y == 1, j]
        if len(anomaly) < 2:
            results.setdefault('KS_Global', []).append(0.0)
            results.setdefault('pval_Global', []).append(1.0)
        else:
            stat, pval = ks_2samp(normal, anomaly)
            results.setdefault('KS_Global', []).append(stat)
            results.setdefault('pval_Global', []).append(pval)

    # Per-regime KS
    for r, rname in regime_names.items():
        mask = regimes == r
        for j, fname in enumerate(feature_names):
            normal = X[(y == 0) & mask, j]
            anomaly = X[(y == 1) & mask, j]
            if len(anomaly) < 2 or len(normal) < 2:
                results.setdefault(f'KS_{rname}', []).append(0.0)
                results.setdefault(f'pval_{rname}', []).append(1.0)
            else:
                stat, pval = ks_2samp(normal, anomaly)
                results.setdefault(f'KS_{rname}', []).append(stat)
                results.setdefault(f'pval_{rname}', []).append(pval)

    df = pd.DataFrame(results, index=feature_names)
    return df


def compute_factor_correlation_clusters(
    X: np.ndarray,
    feature_names: list = None,
    n_clusters: int = 5,
    method: str = 'average',
) -> dict:
    """Hierarchical clustering of factors based on absolute correlation.

    Identifies redundant factor groups to verify that the factor set
    has low redundancy (a preprocessing explainability artifact).

    Returns
    -------
    dict with keys:
        'correlation_matrix': pd.DataFrame
        'cluster_labels': dict mapping feature_name -> cluster_id
        'linkage_matrix': np.ndarray (for dendrogram plotting)
        'cluster_groups': dict mapping cluster_id -> [feature_names]
    """
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(X.shape[1])]

    # Correlation matrix
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)
    corr_df = pd.DataFrame(corr, index=feature_names, columns=feature_names)

    # Distance matrix: 1 - |correlation|
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)  # Ensure non-negative

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    cluster_labels = dict(zip(feature_names, labels))
    cluster_groups = {}
    for fname, cl in cluster_labels.items():
        cluster_groups.setdefault(cl, []).append(fname)

    return {
        'correlation_matrix': corr_df,
        'cluster_labels': cluster_labels,
        'linkage_matrix': Z,
        'cluster_groups': cluster_groups,
    }


def generate_preprocessing_report(
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    feature_names: list = None,
    output_dir: str = None,
) -> dict:
    """Generate comprehensive preprocessing explainability report.

    Combines MI analysis, KS discriminability, and correlation clustering
    into a unified report answering "why these features, in these regimes?"

    Returns
    -------
    dict with keys 'mi_table', 'ks_table', 'clusters', 'summary'
    """
    print("🔍 GENERATING PREPROCESSING EXPLAINABILITY REPORT")
    print("=" * 55)

    # 1. Mutual Information
    print("\n📊 Computing Mutual Information per regime...")
    mi_df = compute_mutual_information_per_regime(
        X, y, regimes, feature_names
    )
    print(mi_df[['MI_Low', 'MI_Med', 'MI_High', 'MI_Global']].round(4).to_string())

    # 2. KS Discriminability
    print("\n📊 Computing KS discriminability per regime...")
    ks_df = compute_factor_discriminability(
        X, y, regimes, feature_names
    )
    ks_cols = [c for c in ks_df.columns if c.startswith('KS_')]
    print(ks_df[ks_cols].round(4).to_string())

    # 3. Correlation Clustering
    print("\n📊 Computing factor correlation clusters...")
    clusters = compute_factor_correlation_clusters(
        X, feature_names
    )
    print("\n  Cluster Groups:")
    for cl_id, members in sorted(clusters['cluster_groups'].items()):
        print(f"    Cluster {cl_id}: {members}")

    # 4. Summary: Top-3 most discriminative factors per regime
    summary = {}
    for regime_col in ['MI_Low', 'MI_Med', 'MI_High']:
        top3 = mi_df[regime_col].nlargest(3).index.tolist()
        regime_name = regime_col.replace('MI_', '')
        summary[regime_name] = top3
        print(f"\n  🏆 Top-3 discriminative factors ({regime_name} Vol):")
        for i, f in enumerate(top3, 1):
            print(f"    {i}. {f} (MI={mi_df.loc[f, regime_col]:.4f})")

    # Save if output_dir provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        mi_df.to_csv(os.path.join(output_dir, 'preprocessing_mi_report.csv'))
        ks_df.to_csv(os.path.join(output_dir, 'preprocessing_ks_report.csv'))
        clusters['correlation_matrix'].to_csv(
            os.path.join(output_dir, 'factor_correlation_matrix.csv')
        )
        print(f"\n💾 Reports saved to {output_dir}")

    return {
        'mi_table': mi_df,
        'ks_table': ks_df,
        'clusters': clusters,
        'summary': summary,
    }


if __name__ == '__main__':
    # Example usage with project data
    X = np.load('artifacts/X_factors.npy')
    y = np.load('artifacts/y_factors.npy')
    df = pd.read_csv('artifacts/journal_factors.csv')

    # Reconstruct regimes from Composite_Volatility
    vol = df['Composite_Volatility'].values
    q33, q66 = np.quantile(vol, [0.33, 0.66])
    regimes = np.zeros(len(vol), dtype=int)
    regimes[vol <= q33] = 0
    regimes[(vol > q33) & (vol <= q66)] = 1
    regimes[vol > q66] = 2

    report = generate_preprocessing_report(
        X, y, regimes,
        feature_names=df.columns.tolist(),
        output_dir='artifacts'
    )
