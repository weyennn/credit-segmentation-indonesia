import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "modal_kerja",
    "investasi",
    "konsumsi",
    "ekspor",
    "impor",
    "lainnya"
]

def generate_features(
    input_path=PROJECT_ROOT / "data/processed/kredit_jp_op_clean.csv",
    output_path=PROJECT_ROOT / "data/processed/kredit_jp_op_features.csv"
):
    print("Membuat fitur dari:", input_path)

    df = pd.read_csv(input_path)

    df_features = df[["provinsi"] + FEATURE_COLS].copy()

    # Pastikan semua fitur numerik
    df_features[FEATURE_COLS] = df_features[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')

    # Drop jika ada NaN setelah konversi
    df_features = df_features.dropna()

    # Hitung total kredit per provinsi sebagai dasar segmentasi dua tingkat
    df_features["total_kredit"] = df_features[FEATURE_COLS].sum(axis=1)

    # Simpan hasil fitur
    os.makedirs(output_path.parent, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"Fitur disimpan ke: {output_path}")
    print(f"Jumlah provinsi: {len(df_features)}")

if __name__ == "__main__":
    generate_features()
