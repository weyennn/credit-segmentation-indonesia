import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


class ABCOptimizer:
    """
    Artificial Bee Colony (ABC) Optimizer untuk inisialisasi centroid K-Means.

    Mengimplementasikan tiga fase standar ABC:
    - Employed Bees  : eksplorasi lokal di sekitar setiap food source
    - Onlooker Bees  : seleksi food source secara proporsional terhadap fitness
    - Scout Bees     : penggantian food source yang tidak berkembang (abandonment)

    Parameter
    ----------
    data        : ndarray, data yang akan di-cluster (sudah di-scaling)
    num_clusters: int, jumlah klaster (k)
    max_iter    : int, jumlah iterasi maksimum
    colony_size : int, jumlah employed bee (= jumlah food source)
    limit       : int, batas trial sebelum food source diganti oleh scout bee
                  (default: colony_size * num_clusters)
    seed        : int, random seed untuk reproduksibilitas
    """

    def __init__(self, data, num_clusters=3, max_iter=50, colony_size=20, limit=None, seed=42):
        self.data = data
        self.k = num_clusters
        self.max_iter = max_iter
        self.colony_size = colony_size
        self.limit = limit if limit is not None else colony_size * num_clusters
        self.dim = data.shape[1]
        self.rng = np.random.RandomState(seed)

    def _initialize_food_sources(self):
        """Inisialisasi food source: tiap source adalah k centroid acak dari data."""
        return np.array([
            self.data[self.rng.choice(self.data.shape[0], self.k, replace=False)]
            for _ in range(self.colony_size)
        ])

    def _evaluate(self, centroids):
        """Hitung SSE (within-cluster sum of squared errors) sebagai ukuran kualitas."""
        _, dists = pairwise_distances_argmin_min(self.data, centroids)
        return np.sum(dists ** 2)

    def _fitness(self, cost):
        """Konversi cost ke nilai fitness untuk seleksi onlooker bee.
        Semakin kecil cost, semakin besar fitness.
        """
        return 1.0 / (1.0 + cost)

    def _generate_candidate(self, food_sources, i):
        """Hasilkan kandidat food source baru dari food source i dengan tetangga j (j ≠ i)."""
        j = i
        while j == i:
            j = self.rng.randint(0, self.colony_size)
        phi = self.rng.uniform(-1, 1, size=(self.k, self.dim))
        return food_sources[i] + phi * (food_sources[i] - food_sources[j])

    def optimize(self):
        """Jalankan optimasi ABC dan kembalikan centroid terbaik yang ditemukan."""
        food_sources = self._initialize_food_sources()
        costs = np.array([self._evaluate(s) for s in food_sources])
        trial_counts = np.zeros(self.colony_size, dtype=int)

        best_idx = np.argmin(costs)
        best_source = food_sources[best_idx].copy()
        best_cost = costs[best_idx]

        for _ in range(self.max_iter):

            # --- Fase 1: Employed Bees ---
            # Tiap employed bee mengeksplorasi di sekitar food source-nya sendiri
            for i in range(self.colony_size):
                candidate = self._generate_candidate(food_sources, i)
                candidate_cost = self._evaluate(candidate)
                if candidate_cost < costs[i]:
                    food_sources[i] = candidate
                    costs[i] = candidate_cost
                    trial_counts[i] = 0
                    if candidate_cost < best_cost:
                        best_source = candidate.copy()
                        best_cost = candidate_cost
                else:
                    trial_counts[i] += 1

            # --- Fase 2: Onlooker Bees ---
            # Onlooker memilih food source secara probabilistik berdasarkan fitness
            fit_vals = np.array([self._fitness(c) for c in costs])
            probs = fit_vals / fit_vals.sum()
            for _ in range(self.colony_size):
                i = self.rng.choice(self.colony_size, p=probs)
                candidate = self._generate_candidate(food_sources, i)
                candidate_cost = self._evaluate(candidate)
                if candidate_cost < costs[i]:
                    food_sources[i] = candidate
                    costs[i] = candidate_cost
                    trial_counts[i] = 0
                    if candidate_cost < best_cost:
                        best_source = candidate.copy()
                        best_cost = candidate_cost
                else:
                    trial_counts[i] += 1

            # --- Fase 3: Scout Bees ---
            # Food source yang tidak berkembang selama >= limit trial diganti acak
            for i in range(self.colony_size):
                if trial_counts[i] >= self.limit:
                    food_sources[i] = self.data[
                        self.rng.choice(self.data.shape[0], self.k, replace=False)
                    ]
                    costs[i] = self._evaluate(food_sources[i])
                    trial_counts[i] = 0

        return best_source
