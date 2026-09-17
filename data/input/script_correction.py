"""
script_correction.py
Corrections à appliquer sur Database_K1_SA.csv après chaque mise à jour.
Usage : python data/script_correction.py
"""

import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent / "Database_K1_SA.csv"


def fix_suparinpei_style(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Pour les athlètes dont plus de 80 % des katas sont en Shotokan,
    passe tous leurs Suparinpei classés en ShitoRyu vers Shotokan.
    """
    athletes_to_fix = []
    athletes_with_supa_shitoRyu = df[
        (df["Kata"] == "Suparinpei") & (df["Style"] == "ShitoRyu")
    ]["Nom"].unique()

    for nom in athletes_with_supa_shitoRyu:
        style_counts = df[df["Nom"] == nom]["Style"].dropna().value_counts()
        total = style_counts.sum()
        if total > 0 and style_counts.get("Shotokan", 0) / total > 0.80:
            athletes_to_fix.append(nom)

    mask = (
        (df["Kata"] == "Suparinpei")
        & (df["Style"] == "ShitoRyu")
        & (df["Nom"].isin(athletes_to_fix))
    )
    count = int(mask.sum())
    df.loc[mask, "Style"] = "Shotokan"

    if count:
        print(f"[fix_suparinpei_style] {count} ligne(s) corrigée(s) pour : {sorted(athletes_to_fix)}")
    else:
        print("[fix_suparinpei_style] Aucune correction nécessaire.")

    return df, count


def main():
    print(f"Lecture de {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH, sep=";")

    df, _ = fix_suparinpei_style(df)

    df.to_csv(CSV_PATH, sep=";", index=False)
    print("Fichier sauvegardé.")


if __name__ == "__main__":
    main()
