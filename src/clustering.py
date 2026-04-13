import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from k_selection import find_optimal_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

FEATURE_COLS = ["modal_kerja", "investasi", "konsumsi", "ekspor", "impor", "lainnya"]


def segment_and_cluster(
    input_path=PROJECT_ROOT / "data/processed/kredit_jp_op_features.csv",
    output_path=PROJECT_ROOT / "data/processed/kredit_jp_op_clustered_dualsegment.csv",
    high_plot=FIGURES_DIR / "high_volume_pca.png",
    normal_plot=FIGURES_DIR / "normal_volume_pca.png"
):
    """
    Lakukan segmentasi dua tingkat lalu clustering pada tiap segmen.

    Segmentasi:
    - High-volume : 25% provinsi dengan total_kredit tertinggi (Q75)
    - Normal-volume: provinsi lainnya

    Clustering per segmen menggunakan K-Means dengan inisialisasi centroid
    dari ABC Optimizer, jumlah k optimal ditentukan oleh Silhouette Score.
    """
    df = pd.read_csv(input_path)

    # Gunakan total_kredit dari feature engineering sebagai dasar segmentasi
    if "total_kredit" not in df.columns:
        df["total_kredit"] = df[FEATURE_COLS].sum(axis=1)

    # Segmentasi dua tingkat berdasarkan Q75 total kredit
    threshold = df["total_kredit"].quantile(0.75)
    df_high = df[df["total_kredit"] > threshold].copy().reset_index(drop=True)
    df_normal = df[df["total_kredit"] <= threshold].copy().reset_index(drop=True)

    print(f"Threshold Q75: {threshold:,.2f}")
    print(f"Segmen high-volume : {len(df_high)} provinsi")
    print(f"Segmen normal-volume: {len(df_normal)} provinsi\n")

    def run_clustering_segment(df_segment, plot_path, segment_name):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_segment[FEATURE_COLS])

        # Tentukan k optimal menggunakan ABC + Silhouette Score
        k_eval_path = FIGURES_DIR / f"k_evaluation_{segment_name}.png"
        k_optimal, best_centroids = find_optimal_k(
            X_scaled,
            k_range=range(2, 7),
            plot_path=k_eval_path
        )

        # Clustering final dengan centroid inisialisasi terbaik dari ABC
        kmeans = KMeans(n_clusters=k_optimal, init=best_centroids, n_init=1, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        df_segment = df_segment.copy()
        df_segment["cluster"] = labels
        df_segment["segment"] = segment_name

        # Visualisasi dengan proyeksi PCA 2D
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        variance_explained = pca.explained_variance_ratio_

        df_pca = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        df_pca["cluster"] = labels
        df_pca["provinsi"] = df_segment["provinsi"].values

        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="cluster", palette="Set2", s=120)
        plt.title(
            f"Segment: {segment_name} (k={k_optimal})\n"
            f"Var. explained: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}"
        )
        for i in range(len(df_pca)):
            plt.text(
                df_pca["PC1"].iloc[i] + 0.1,
                df_pca["PC2"].iloc[i],
                df_pca["provinsi"].iloc[i],
                fontsize=8
            )
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return df_segment

    # Clustering untuk masing-masing segmen
    print("=== Segmen High-Volume ===")
    df_high_clustered = run_clustering_segment(df_high, high_plot, "high_volume")

    print("\n=== Segmen Normal-Volume ===")
    df_normal_clustered = run_clustering_segment(df_normal, normal_plot, "normal_volume")

    # Gabungkan dan simpan hasil
    df_final = pd.concat([df_high_clustered, df_normal_clustered], axis=0).reset_index(drop=True)
    os.makedirs(output_path.parent, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"\nClustering 2 level selesai!")
    print(f"Hasil : {output_path}")
    print(f"Plot  : {high_plot}, {normal_plot}")

if __name__ == "__main__":
    segment_and_cluster()
