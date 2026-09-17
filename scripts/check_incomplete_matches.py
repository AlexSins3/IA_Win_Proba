"""Détecte les matchs Tbilisi avec un seul clip (red ou blue manquant sur disque)."""
import sys
sys.path.insert(0, "src")

from pathlib import Path
import pandas as pd
from kata_pipeline.gnn.data.dataset import _expected_stem

clips_dir = Path("data/clips")
clip_index = {p.stem: p for p in clips_dir.rglob("*.mp4")}

results = []
for csv in sorted(Path("data/output").glob("sa_tbilisi_*_clips.csv")):
    df = pd.read_csv(csv)
    for match_id, grp in df.groupby("id_match"):
        found = []
        missing = []
        for _, row in grp.iterrows():
            stem = _expected_stem(row)
            if stem in clip_index:
                found.append(row["color"])
            else:
                missing.append(f"{row['color']}({row['athlete']})")
        if missing:
            results.append({
                "csv": csv.name,
                "id_match": match_id,
                "found": found,
                "missing": missing,
            })

if results:
    print(f"{len(results)} match(s) incomplet(s) :\n")
    for r in results:
        print(f"  [{r['csv']}]")
        print(f"    id_match : {r['id_match']}")
        print(f"    présent  : {r['found']}")
        print(f"    manquant : {r['missing']}")
        print()
else:
    print("Aucun match incomplet — tous les clips red+blue sont présents.")
