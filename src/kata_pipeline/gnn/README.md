# Phase 2 — Poses + Graph Neural Network (premier jet)

Ce module transforme les clips de kata (phase 1) en un modèle capable de :

1. **Comparer deux katas et prédire le vainqueur** (tâche cible), de façon
   **anonyme** : le modèle juge un geste, jamais une personne.
2. Estimer une **marge de victoire** (nombre de drapeaux WKF, 0–5) et un score
   technique quand il est disponible.
3. Produire des **vidéos de visualisation** des déplacements du corps et de la
   **saillance du modèle** (où il regarde).

> ⚠️ **Premier jet.** Avec ~143 clips (~71 matchs) sous un angle quasi constant,
> l'objectif est de **valider la chaîne complète** (poses → graphe → GNN → viz),
> pas d'obtenir un juge fiable. Voir « Limites » plus bas.

## Chaîne de traitement

```
clip .mp4 ──MediaPipe──► poses (T,J,3) ──features──► (T,J,9) ──fenêtres──► ST-GCN ──► embedding
                                                                                     ├─► P(A bat B)  (siamois)
                                                                                     ├─► marge drapeaux
                                                                                     └─► score technique
```

### Articulations suivies (15)
nez, épaules, coudes, poignets, hanches, genoux, chevilles, pieds
(sous-ensemble stable des 33 landmarks MediaPipe ; les mains/pieds fins sont
trop bruités de loin).

### Features par nœud (9 canaux)
`[x, y, z, vx, vy, vz, |v|, angle_articulaire, confiance]`
→ position, **vitesse** (dynamique épaules/hanches/poignets/chevilles),
**angles** (coude, genou, épaule, hanche), confiance de détection.

Le modèle est un **ST-GCN** (Spatial-Temporal Graph Convolutional Network) en
PyTorch pur (pas de `torch-geometric`, pénible à installer sous Windows).

## Installation

```powershell
pip install -e ".[gnn]"   # mediapipe, torch, scikit-learn, matplotlib, tqdm
```

## Utilisation

```powershell
# 1) Extraire les poses des clips validés liés à un CSV
#    -> data/poses/SA/*.npz ou data/poses/K1/*.npz
kata-pipeline gnn extract-poses
kata-pipeline gnn extract-poses --competition-type K1

# 2) Entraîner le juge (prédiction du vainqueur)
kata-pipeline gnn train-comparator

# 3) (option) Entraîner la tête de qualité (marge de drapeaux + score)
kata-pipeline gnn train-quality

# 4) Visualiser les déplacements du corps (sans modèle)
kata-pipeline gnn viz-motion data/clips/SA/pool_1/UN_CLIP.mp4

# 5) Visualiser la saillance du modèle
kata-pipeline gnn viz-saliency data/clips/SA/pool_1/UN_CLIP.mp4 --model models/quality.pt

# 6) Comparer deux katas
kata-pipeline gnn compare data/clips/.../A.mp4 data/clips/.../B.mp4
```

Configuration surchargée via YAML : `kata-pipeline gnn ... --config mon_gnn.yaml`
(voir `GNNConfig` dans `config.py`).

`extract-poses` parcourt les MP4 (hors `pending` par défaut). Les commandes
`train-*` et `evaluate-outcome` utilisent uniquement les CSV et les `.npz` : les
clips peuvent donc être archivés après extraction et contrôle des poses. Les
anciens dossiers `data/clips/pool_*` et les poses à plat restent reconnus.

Avec `archive.enabled: true`, `extract-poses` complète son manifeste depuis
l'index Google Drive et restaure seulement les clips dont la pose doit être
créée ou recalculée. Le cache local est borné et vérifié par checksum. Les
commandes `kata-pipeline drive auth|index|status|fetch|prune` gèrent cette
archive ; le fichier d'exemple est `config/drive.example.yaml` et la procédure
complète figure dans le `GUIDE.md` à la racine.

## Structure

```
gnn/
├── config.py            # GNNConfig (pose, graphe, modèle, entraînement, chemins)
├── graph/skeleton.py    # articulations, arêtes, matrice d'adjacence
├── pose/
│   ├── extractor.py     # MediaPipe -> poses .npz
│   └── features.py      # angles, vitesses, normalisation, fenêtrage
├── data/dataset.py      # manifestes extraction et train, datasets, split par match
├── models/
│   ├── stgcn.py         # encodeur ST-GCN
│   └── heads.py         # KataQualityNet + KataComparator (siamois anti-symétrique)
├── train.py             # boucles d'entraînement
├── viz/overlay.py       # vidéos mouvement + saillance
└── cli.py               # sous-commandes `kata-pipeline gnn ...`
```

## Choix de conception clés

- **Anonymat** : le comparateur est *siamois anti-symétrique*
  `logit(A>B) = q(B) - q(A)` ; swapper les entrées inverse la prédiction.
  Aucune feature d'identité n'est utilisée.
- **Normalisation** des poses (centrage bassin + échelle torse) → invariance à la
  position et à la distance caméra.
- **Split par match** (jamais rouge et bleu du même match des deux côtés) → pas de
  fuite de données.
- **Cibles masquées** : les scores manquants n'entrent pas dans la perte.

## Limites (à avoir en tête)

- **Très peu de données** : ~71 matchs → un GNN peut mémoriser. Résultats à ne pas
  surinterpréter. Pistes : augmentation temporelle (déjà en place via fenêtres),
  flip horizontal, jitter ; pré-entraînement auto-supervisé sur les poses.
- **Angle unique** : bon pour un premier modèle, mais pas d'invariance de vue
  apprise. Multi-angles/multi-compétitions à collecter pour généraliser.
- **Pose 2D bruitée** : occlusions, profondeur `z` peu fiable. Envisager un
  lisseur temporel (One-Euro) ou un modèle 3D ultérieurement.
- **Labels indirects** : on prédit le *résultat de match* (drapeaux), pas la note
  d'un juge kata. C'est un proxy utile mais imparfait de la qualité technique.
