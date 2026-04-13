import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from abc_optimizer import ABCOptimizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"


def find_optimal_k(data, k_range=range(2, 10), seed=42, save_plot=True, plot_path=None):
    """
    Cari jumlah klaster optimal menggunakan Silhouette Score.

    Untuk setiap k dalam k_range:
    - Inisialisasi centroid dengan ABC Optimizer
    - Jalankan K-Means
    - Hitung Silhouette Score dan Inertia (SSE)

    Parameter
    ----------
    data      : ndarray, data yang sudah di-scaling
    k_range   : range, rentang k yang dievaluasi
    seed      : int, random seed
    save_plot : bool, simpan grafik evaluasi k atau tidak
    plot_path : Path/str, path untuk menyimpan grafik (default: figures/k_evaluation.png)

    Return
    ------
    best_k         : int, jumlah klaster optimal
    best_centroids : ndarray, centroid inisialisasi terbaik dari ABC
    """
    silhouette_scores = []
    inertia_scores = []
    best_k = None
    best_score = -1
    best_centroids = None

    print("Mencari k optimal berdasarkan silhouette score...")

    for k in k_range:
        abc = ABCOptimizer(data=data, num_clusters=k, seed=seed)
        init_centroids = abc.optimize()

        kmeans = KMeans(n_clusters=k, init=init_centroids, n_init=1, random_state=seed)
        labels = kmeans.fit_predict(data)

        sil_score = silhouette_score(data, labels)
        inertia = kmeans.inertia_

        silhouette_scores.append(sil_score)
        inertia_scores.append(inertia)

        print(f"  k={k} | Silhouette Score: {sil_score:.4f} | SSE: {inertia:.2f}")

        if sil_score > best_score:
            best_score = sil_score
            best_k = k
            best_centroids = init_centroids

    if save_plot:
        if plot_path is None:
            plot_path = FIGURES_DIR / "k_evaluation.png"
        plot_path = Path(plot_path)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(list(k_range), silhouette_scores, marker='o')
        plt.title("Silhouette Score vs k")
        plt.xlabel("Jumlah Klaster (k)")
        plt.ylabel("Silhouette Score")

        plt.subplot(1, 2, 2)
        plt.plot(list(k_range), inertia_scores, marker='o', color='orange')
        plt.title("SSE (Inertia) vs k")
        plt.xlabel("Jumlah Klaster (k)")
        plt.ylabel("Inertia")

        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"  Grafik evaluasi disimpan: {plot_path}")

    print(f"\nK optimal: {best_k} (silhouette={best_score:.4f})")
    return best_k, best_centroids
