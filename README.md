# Segmentasi Provinsi di Indonesia Berdasarkan Kredit JP–OP

## Latar Belakang Masalah
Distribusi kredit antar provinsi di Indonesia menunjukkan perbedaan yang signifikan.
Penerapan kebijakan kredit yang seragam berpotensi mengabaikan karakteristik
keuangan daerah dan menimbulkan ketidakefisienan kebijakan.

Proyek ini bertujuan untuk melakukan segmentasi provinsi di Indonesia berdasarkan
data Kredit menurut Jenis Penggunaan (JP) dan Orientasi Pengguna (OP) guna
mendukung analisis keuangan regional serta perumusan kebijakan yang lebih tepat sasaran.

---

## Data
- **Sumber:** Statistik Perbankan Indonesia (OJK), Februari 2025
- **Unit analisis:** Provinsi
- **Variabel utama:**
  - Kredit modal kerja
  - Kredit investasi
  - Kredit konsumsi
  - Kredit ekspor
  - Kredit impor
  - Kredit lainnya

---

## Pendekatan Analisis

### 1. Persiapan Data
- Menghapus entri non-analitis seperti NPL/NPF, total agregat, dan kategori lainnya
- Normalisasi dan pembersihan penamaan provinsi

### 2. Rekayasa Fitur
- Perhitungan total kredit per provinsi
- Segmentasi dua tingkat untuk menangani ketimpangan skala:
  - **High-volume**: 25% provinsi dengan total kredit tertinggi
  - **Normal-volume**: provinsi lainnya

Pendekatan ini digunakan untuk mencegah dominasi provinsi dengan volume kredit
sangat besar (misalnya DKI Jakarta) dalam proses clustering.

### 3. Clustering
- Penerapan **K-Means** dengan optimasi centroid menggunakan
  **Artificial Bee Colony (ABC)**
- Penentuan jumlah klaster optimal menggunakan **Silhouette Score**

---

## Hasil Segmentasi

> Data mencakup **33 provinsi** (Kalimantan Utara tidak tersedia dalam data OJK Februari 2025).

### Segmen High-Volume (8 provinsi — Q75 tertinggi)
- Jumlah klaster optimal: **2** | Silhouette Score: **0,7906**
- Distribusi klaster:
  - Klaster 0: 1 provinsi — DKI Jakarta (outlier ekstrem)
  - Klaster 1: 7 provinsi — Jawa Barat, Jawa Timur, Jawa Tengah, Banten, Sumatera Utara, Sumatera Selatan, Sulawesi Selatan

### Segmen Normal-Volume (25 provinsi)
- Jumlah klaster optimal: **4** | Silhouette Score: **0,4113**
- Distribusi klaster:
  - Klaster 0: 7 provinsi — Bengkulu, Bangka Belitung, Gorontalo, Sulawesi Barat, Maluku, Maluku Utara, Papua Barat
  - Klaster 1: 14 provinsi — D.I Yogyakarta, Jambi, Aceh, Sumatera Barat, Lampung, Kalimantan Selatan, Kalimantan Barat, Kalimantan Tengah, Sulawesi Tengah, Sulawesi Utara, Sulawesi Tenggara, NTB, NTT, Papua
  - Klaster 2: 3 provinsi — Riau, Kalimantan Timur, Bali
  - Klaster 3: 1 provinsi — Kepulauan Riau

---

## Insight Utama
- Provinsi dengan volume kredit sangat tinggi membentuk klaster tersendiri
  dan memiliki karakteristik yang berbeda secara signifikan.
- Segmentasi dua tingkat efektif dalam mengurangi distorsi akibat outlier
  pada proses clustering.
- Provinsi pada segmen normal menunjukkan variasi struktur kredit yang cukup besar,
  menandakan bahwa perilaku kredit tidak seragam secara nasional.
- Pola kredit memperlihatkan konsentrasi geografis dan ekonomi tertentu.

---

## Rekomendasi
- **Kebijakan Diferensial:**  
  Regulator perlu menerapkan kebijakan kredit yang disesuaikan dengan
  karakteristik klaster provinsi, bukan pendekatan seragam nasional.
- **Pemantauan Risiko:**  
  Provinsi dengan volume kredit ekstrem memerlukan kerangka pemantauan khusus
  karena pengaruhnya yang besar terhadap indikator nasional.
- **Program Keuangan Terarah:**  
  Inisiatif inklusi dan ekspansi kredit dapat diprioritaskan berdasarkan
  karakteristik segmen provinsi.

---

## Visualisasi
- Visualisasi PCA dan pemetaan klaster tersedia pada folder `/figures`:
  - `map_cluster_jp_op_segmented.png` – Peta klaster kredit per segmen
  - `high_volume_pca.png` – Proyeksi PCA segmen high-volume
  - `normal_volume_pca.png` – Proyeksi PCA segmen normal-volume
  - `k_evaluation_high_volume.png` – Grafik evaluasi k segmen high-volume
  - `k_evaluation_normal_volume.png` – Grafik evaluasi k segmen normal-volume

---

## Tools & Library
- Python, Pandas, Scikit-learn
- GeoPandas, Matplotlib, Seaborn
- Artificial Bee Colony (custom optimizer)
- PCA dan visualisasi geospasial

---

## Struktur Proyek

```
src/
├── abc_optimizer.py
├── k_selection.py
├── preprocessing.py
├── feature_engineering.py
├── clustering.py
└── mapping.py

data/
├── raw/
├── processed/
└── geo/

EDA/
└── eksplorasi_data_kredit.ipynb

figures/
└── *.png
```
---

## Penulis
**Yayang Matira**  
Mahasiswa Magister Ilmu Komputer  
Universitas Gadjah Mada
