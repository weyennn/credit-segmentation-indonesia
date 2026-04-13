import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Baris-baris di Excel yang bukan nama provinsi
NON_PROVINCE_KEYWORDS = ["total", "lainnya", "luar negeri", "npl", "npf"]

def preprocess_kredit_jp_op(
    input_path=PROJECT_ROOT / "data/raw/STATISTIK_PERBANKAN_FEBRUARI 2025.xlsx",
    sheet_name="Kredit JP-OP per Lok._3.12.a.",
    output_path=PROJECT_ROOT / "data/processed/kredit_jp_op_clean.csv"
):
    print("Membaca sheet:", sheet_name)

    # Load dengan skip baris header multi-baris
    df_raw = pd.read_excel(input_path, sheet_name=sheet_name, skiprows=3)

    # Drop kolom kosong
    df = df_raw.dropna(axis=1, how='all').reset_index(drop=True)

    # Bersihkan nama kolom
    df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)

    # Rename kolom
    df = df.rename(columns={
        "Lokasi / Location": "provinsi",
        "Modal Kerja (Working Capital)": "modal_kerja",
        "Investasi (Investment)": "investasi",
        "Konsumsi (Consumption)": "konsumsi",
        "Ekspor": "ekspor",
        "Impor": "impor",
        "Lainnya": "lainnya"
    })

    df = df[["provinsi", "modal_kerja", "investasi", "konsumsi", "ekspor", "impor", "lainnya"]]

    # Bersihkan spasi pada nama provinsi
    df["provinsi"] = df["provinsi"].str.strip()

    # Buang baris non-provinsi (Total, Lainnya, NPL, dst.)
    mask_non_province = df["provinsi"].str.lower().apply(
        lambda x: any(kw in str(x) for kw in NON_PROVINCE_KEYWORDS)
    )
    df = df[~mask_non_province]

    # Drop jika masih ada NaN
    df = df.dropna()

    # Simpan
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data hasil preprocessing disimpan di {output_path}")
    print(f"Jumlah provinsi: {len(df)}")

if __name__ == "__main__":
    preprocess_kredit_jp_op()
