# Kata Pipeline

Pipeline d'extraction de clips vidéo de kata depuis des lives YouTube de compétitions de karaté.

## Objectif

Transformer des enregistrements longs de lives de compétitions de karaté en clips vidéo individuels de kata, proprement découpés et alignés avec un dataset sportif existant.

Ce projet est la **phase 1** d'un système plus large qui visera à :
- Extraire des features vidéo (poses, keypoints)
- Définir des critères techniques de jugement kata
- Construire un Graph Neural Network pour la prédiction de notes et de résultats

## Architecture

```
src/kata_pipeline/
├── cli.py              # Interface en ligne de commande (Typer)
├── config.py           # Configuration centralisée (Pydantic)
├── schemas/            # Modèles de données
│   ├── competition.py  # Schéma du dataset de matchs
│   ├── live.py         # Schéma des métadonnées de lives
│   ├── passage.py      # Passages attendus (rouge/bleu)
│   └── clip.py         # Clips vidéo extraits
├── loaders/            # Chargement des données d'entrée
│   ├── dataset_loader.py
│   └── live_loader.py
├── video/              # Opérations vidéo
│   ├── downloader.py   # Téléchargement YouTube (optionnel)
│   ├── preparer.py     # Préparation et redimensionnement
│   └── clipper.py      # Extraction des clips
├── detection/          # Détection des katas dans la vidéo
│   ├── motion_score.py # Calcul du score de mouvement
│   ├── segment_detector.py  # Détection de segments candidats
│   └── pairing.py      # Groupement rouge/bleu
├── alignment/          # Alignement vidéo ↔ dataset sportif
│   └── aligner.py
├── validation/         # Interface de validation humaine
│   └── app.py          # Application Streamlit
└── utils/              # Utilitaires
    ├── timecode.py     # Conversions de timecodes
    ├── filenames.py    # Génération de noms de fichiers
    └── logging.py      # Configuration du logging
```

## Installation

```bash
# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer le projet
pip install -e ".[all]"
```

### Dépendances système requises

- **ffmpeg** : pour l'extraction et le traitement vidéo
- **yt-dlp** (optionnel) : pour le téléchargement de vidéos YouTube

## Fichiers d'entrée attendus

### 1. Dataset de compétition (CSV/Excel/Parquet)

Colonnes **requises** :
| Colonne | Description |
|---------|-------------|
| `competition` | Nom de la compétition |
| `category` | Catégorie (Male Kata, Female Kata, etc.) |
| `round` | Tour (Round 1, Quarter-final, etc.) |
| `match_order` | Ordre du match dans le tour |
| `athlete_red` | Nom de l'athlète rouge |
| `athlete_blue` | Nom de l'athlète bleu |
| `kata_red` | Kata exécuté par rouge |
| `kata_blue` | Kata exécuté par bleu |

Colonnes **optionnelles** : `competition_type` (`SA` ou `K1`), `style_red`,
`style_blue`, `score_red`, `score_blue`, `flag_result_red`, `flag_result_blue`,
`winner`, `round_outcome`, `decision_type`

### 2. Table des lives (CSV/Excel/Parquet)

Colonnes **requises** :
| Colonne | Description |
|---------|-------------|
| `id_live` | Identifiant unique du live |
| `competition` | Compétition correspondante |
| `category` | Catégorie correspondante |

Colonnes **optionnelles** : `competition_type` (`SA` ou `K1`), `url`,
`local_path`, `useful_start`, `useful_end`, `analysis_quality`, `clip_quality`,
`status`

## Fichiers de sortie

| Fichier | Description |
|---------|-------------|
| `data/intermediate/{live_id}_range.mp4` | Plage utile extraite du live |
| `data/intermediate/{live_id}_analysis.mp4` | Version basse résolution pour analyse |
| `data/intermediate/{live_id}_motion.json` | Scores de mouvement |
| `data/intermediate/{live_id}_segments.json` | Segments candidats détectés |
| `data/intermediate/{live_id}_pairs.json` | Pairing rouge/bleu |
| `data/output/{live_id}_clips.csv` | Dataset final de clips alignés |
| `data/clips/{SA,K1}/pending/*.mp4` | Clips vidéo à valider, séparés par circuit |
| `data/clips/{SA,K1}/.../*.mp4` | Clips vidéo classés/validés |
| `data/poses/{SA,K1}/*.npz` | Poses compactes utilisées par l'entraînement |

## Utilisation

### Préparer un live

1. Placer la vidéo dans `data/input/` ou renseigner l'URL YouTube dans la table des lives
2. Créer/compléter les fichiers CSV d'entrée

### Lancer la pipeline complète

```bash
kata-pipeline run-pipeline LIVE_ID \
    --dataset data/input/competition_dataset.csv \
    --lives data/input/lives.csv
```

### Lancer les étapes séparément

```bash
# 1. Préparer la vidéo
kata-pipeline prepare-video LIVE_ID --lives data/input/lives.csv

# 2. Calculer le mouvement
kata-pipeline compute-motion LIVE_ID

# 3. Détecter les segments
kata-pipeline detect-segments LIVE_ID

# 4. Pairing et alignement
kata-pipeline align LIVE_ID \
    --dataset data/input/competition_dataset.csv \
    --lives data/input/lives.csv

# 5. Générer les clips
kata-pipeline generate-clips LIVE_ID
```

kata-pipeline run-pipeline sa_acoruna_2026_f_pool5_t1 --dataset data/input/SA_ACoruna_2026_F_pool5.csv --lives data/input/SA_ACoruna_2026_F_lives.csv

### Valider les clips

```bash
kata-pipeline validate
```

Cela ouvre une interface Streamlit locale pour parcourir, valider ou corriger les clips.

## Poses, entraînement et archivage vidéo

```bash
# Les clips dans pending sont exclus par défaut.
kata-pipeline gnn extract-poses
kata-pipeline gnn extract-poses --competition-type K1

# Tous les circuits par défaut, ou un seul avec -t SA / -t K1.
kata-pipeline gnn train-outcome
kata-pipeline gnn train-outcome --competition-type SA
```

L'extraction de poses utilise les clips locaux. En revanche, l'entraînement et
l'évaluation construisent leur manifeste directement depuis `data/output` et
`data/poses` : aucune vidéo locale n'est requise une fois les poses produites.

Les clips archivés sur Google Drive peuvent être restaurés automatiquement
quand une pose manque. Le connecteur utilise un accès OAuth en lecture seule,
un index par `drive_file_id` et un cache local LRU de 10 Go :

```bash
pip install -e ".[gnn,drive]"
# Copier config/drive.example.yaml en config/drive.local.yaml, renseigner
# l'ID du dossier Drive `data`, puis placer le JSON OAuth dans .secrets/.
kata-pipeline drive auth -c config/drive.local.yaml
kata-pipeline drive index -c config/drive.local.yaml
kata-pipeline drive status -c config/drive.local.yaml
kata-pipeline gnn extract-poses -c config/drive.local.yaml
```

Voir [GUIDE.md](GUIDE.md) pour la configuration complète. Le train ne contacte
jamais Drive : il reste fondé uniquement sur les CSV et les fichiers de poses.

### Relancer une étape

Chaque étape sauvegarde ses résultats intermédiaires. Vous pouvez relancer n'importe quelle étape sans refaire les précédentes, tant que les fichiers intermédiaires existent.

## Configuration

La configuration est dans `config/default.yaml`. Tous les seuils sont modifiables :

```yaml
# Exemple : ajuster la détection
segments:
  min_duration: 30.0      # Durée min d'un kata (secondes)
  max_duration: 300.0     # Durée max d'un kata
  min_motion_score: 10.0  # Seuil de mouvement
  merge_gap: 5.0          # Gap max à fusionner

# Ajuster le pairing
pairing:
  max_gap_within_match: 120.0  # Écart max rouge-bleu
```

Vous pouvez passer un fichier de config personnalisé avec `--config custom.yaml`.

## Tests

```bash
pytest
pytest --cov=kata_pipeline
```

## Vision long terme

Une fois les clips extraits et validés, la phase 2 du projet consistera à :

1. **Extraction de poses** : Utiliser MediaPipe ou OpenPose pour extraire les keypoints du corps
2. **Critères techniques** : Définir les critères de jugement kata (stabilité, vitesse, amplitude, etc.)
3. **Graphe temporel** : Transformer chaque kata en graphe spatiotemporel
4. **Graph Neural Network** : Entraîner un GNN pour :
   - Prédire la note d'un kata
   - Prédire l'issue d'un match (rouge gagne / bleu gagne)
   - Analyser la performance sur une vidéo d'entraînement

Le code de la phase 1 est conçu pour faciliter cette extension : les clips sont proprement labellisés avec toutes les métadonnées nécessaires.
