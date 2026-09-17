# Guide d'utilisation — Pipeline Kata

Guide pratique : où changer les infos (lien vidéo, timecodes, pool) et quelles commandes lancer.

Deux circuits sont pris en charge à partir de `Type_Compet` dans
`data/input/Database_K1_SA.csv` :

| Type | Tours de pool | Phases finales |
|---|---|---|
| `SA` | `T1`, `T2`, `T3`, `PW1` | `PW2`, `PW3`, `Final`, `Bronze`, `RP1`…`RP4` |
| `K1` | `Pool_1`, `Pool_2`, `Pool_3` | `R1`, `R2`, `Bronze`, `Final` |

Les nouveaux clips sont séparés sous `data/clips/SA/` et `data/clips/K1/`.
L'ancienne arborescence directement sous `data/clips/` reste lisible pendant la
migration.

---

## 0. Préparer le terminal (à faire à chaque nouveau terminal)

```powershell
$env:PATH = "C:\Users\avincent\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin;$env:PATH"
$env:PATH = "C:\Users\avincent\AppData\Local\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe;$env:PATH"
$env:KATA_YTDLP_COOKIES_BROWSER = "chrome"
.\.venv\Scripts\Activate.ps1
```

- `ffmpeg` (découpe/extraction vidéo) et `yt-dlp` (téléchargement YouTube) doivent être dispo.
- `yt-dlp` manquant → `pip install -U "yt-dlp[default]"` (le groupe `default` embarque les scripts EJS).
- `deno` : runtime JavaScript requis par YouTube pour déchiffrer les flux (challenge EJS). Installé via `winget install DenoLand.Deno`. La ligne `PATH` ci-dessus le rend accessible.
- `KATA_YTDLP_COOKIES_BROWSER` : navigateur d'où `yt-dlp` lit les cookies YouTube (contourne le blocage « Sign in to confirm you're not a bot »). Mets le navigateur où tu es connecté à Google (`chrome`, `firefox`, `edge`…) et **ferme-le** avant de lancer (sinon sa base de cookies est verrouillée). Variante fichier : `$env:KATA_YTDLP_COOKIES_FILE = "D:\chemin\cookies.txt"`.

---

## 1. Ajouter une nouvelle pool (où changer les infos)

Un seul script fait **tout** : il crée le dataset des matchs **et** ajoute les lignes du fichier lives (lien vidéo + timecodes). Tu n'as plus rien à éditer à la main.

### 1.1 — Tout générer en une commande

**Étape 1 — Lister les pools** (pour repérer le bon numéro et vérifier le 1er athlète) :

```powershell
.\.venv\Scripts\python.exe scripts\build_pool_dataset.py --competition SA_ACoruna --year 2026 --sex F --list
```

Sortie (exemple) :

```
8 pool(s) détectée(s) :
  pool  5 | 1er athlète (rouge) : Marchetti_Giulia | matchs T1:8 T2:4 T3:2 PW1:1
```

**Étape 2 — Générer dataset + lives** en passant le **lien vidéo** et les **timecodes** de chaque tour (format `DEBUT-FIN`, en `mm:ss` ou `h:mm:ss` — conversion en secondes automatique) :

```powershell
.\.venv\Scripts\python.exe scripts\build_pool_dataset.py `
  --competition SA_ACoruna --year 2026 --sex F --pool 5 `
  --url "https://www.youtube.com/watch?v=XXXX" `
  --t1 "15:28-1:00:45" --t2 "1:01:46-1:33:01" --t3 "1:36:09-1:49:26" --pw1 "1:55:37-2:02:34"
```

Ça produit :

- `data/input/SA_ACoruna_2026_F_pool5.csv` (résultats des matchs)
- 4 lignes ajoutées à `data/input/SA_ACoruna_2026_F_lives.csv` (lien + timecodes + sections)

> Les `id_live` déjà présents ne sont pas dupliqués : relancer la commande est sans risque.

**Si les premiers matchs d'un tour ne sont pas filmés** (live buggé), ajoute `--start-athlete` pour démarrer au bon match :

```powershell
.\.venv\Scripts\python.exe scripts\build_pool_dataset.py `
  --competition SA_ACoruna --year 2026 --sex F --pool 5 --start-athlete Mrakovic_Mila `
  --url "https://www.youtube.com/watch?v=XXXX" `
  --t1 "15:28-1:00:45" --t2 "1:01:46-1:33:01" --t3 "1:36:09-1:49:26" --pw1 "1:55:37-2:02:34"
```

> `--start-athlete` ne touche que le tour T1 (le décalage du live). Les timecodes, eux, pointent déjà sur le vrai début filmé de chaque tour.

### 1.1 bis — Exemple K1

Le type est normalement déduit automatiquement. `--type-compet K1` permet de
le rendre explicite. Les timecodes K1 utilisent l'option générique répétable
`--round` :

```powershell
.\.venv\Scripts\python.exe scripts\build_pool_dataset.py `
  --competition K1_Paris --type-compet K1 --year 2024 --sex F --list

.\.venv\Scripts\python.exe scripts\build_pool_dataset.py `
  --competition K1_Paris --type-compet K1 --year 2024 --sex F --pool 1 `
  --url "https://www.youtube.com/watch?v=XXXX" `
  --round "Pool_1=10:00-30:00" `
  --round "Pool_2=31:00-50:00" `
  --round "Pool_3=51:00-1:05:00"
```

Pour les finales K1, lance d'abord `--finals --list`. Les clés produites sont
`r1_N`, `r2_N`, `bronze_N` et `final_N`, puis s'utilisent avec `--match` comme
pour SA.

### 1.2 — Options du script

| Option | Rôle |
|---|---|
| `--competition` | ex. `SA_ACoruna` (obligatoire) |
| `--type-compet` | `SA` ou `K1` ; facultatif si déductible de la base ou du nom |
| `--year` | ex. `2026` (obligatoire) |
| `--sex` | `F` (Female Kata) ou `M` (Male Kata) |
| `--pool N` | numéro de la pool (ordre d'apparition dans la base) |
| `--finals` | générer les phases finales au lieu d'une pool (voir 1.4) |
| `--list` | affiche les pools détectées sans rien écrire |
| `--start-athlete` | 1er athlète compté en T1 (saute les matchs non filmés au début) |
| `--end-athlete` | dernier athlète compté (arrête la génération après) |
| `--url` | lien YouTube de la pool (déclenche l'ajout au lives) |
| `--t1` / `--t2` / `--t3` / `--pw1` | timecodes d'un tour, format `DEBUT-FIN` (ex. `15:28-1:00:45`) |
| `--round "TOUR=DEBUT-FIN"` | timecode générique répétable, notamment pour `Pool_1/2/3` en K1 |
| `--match "clé=DEBUT-FIN"` | (avec `--finals`) un match final filmé ; répéter par match. URL différente : `"clé=URL\|DEBUT-FIN"` (voir 1.4) |
| `--video-name` | nom du `.mp4` local (défaut : `<compet>_<year>_<sex>_pool<N>.mp4`) |
| `--lives` | fichier lives à compléter (défaut : `data/input/SA_ACoruna_2026_F_lives.csv`) |
| `--output` | chemin du dataset (défaut : `data/input/<compet>_<year>_<sex>_pool<N>.csv`) |
| `--database` | base source (défaut : `data/input/Database_K1_SA.csv`) |

> **Sans `--url`**, seul le dataset des matchs est généré (le lives n'est pas modifié) — pratique si tu veux ajouter les timecodes plus tard.
> **Comment ça marche** : le script choisit le profil depuis `Type_Compet`,
> découpe la base à chaque retour au premier tour (`T1` ou `Pool_1`), appaire
> les lignes adjacentes Rouge/Bleu et déduit le vainqueur via
> `Victoire=VRAI`. Vérifie toujours avec `--list`. Une incohérence de tour dans
> la source est signalée explicitement et le tour de la ligne rouge est retenu.

### 1.3 — Rien d'autre à toucher

- La vidéo se télécharge toute seule au 1er run (pas besoin de la mettre à la main).
- `config/default.yaml` : à modifier **seulement** si la détection des katas déraille (durées, `merge_gap`, seuils).
- Le dossier de sortie des clips est créé automatiquement.

### 1.4 — Phases finales (quarts, demi, finale, bronze, repêchages)

Les phases finales ne sont **pas** des poules : elles forment un bloc global (les vainqueurs de poule s'affrontent). On utilise donc `--finals` **à la place** de `--pool N`. Deux particularités :

- On travaille **match par match** (pas par tour entier), car un même tour peut être réparti sur **plusieurs tatamis = plusieurs vidéos**.
- Seuls les matchs dont tu donnes un timecode sont écrits (dataset **et** lives).

**Étape 1 — Lister les matchs** pour connaître les clés et vérifier les noms :

```powershell
.\.venv\Scripts\python.exe scripts\build_pool_dataset.py --competition SA_ACoruna --year 2026 --sex F --finals --list
```

Sortie (exemple) :

```
Finales SA_ACoruna 2026 F — 17 match(s) :
  quart_1   | Minamoto_Aira    vs Ferreira_Vielma...  -> Minamoto_Aira
  quart_2   | Ikarashi_Risa    vs Abdalghany_Maryam   -> Ikarashi_Risa
  ...
  demi_1    | Minamoto_Aira    vs Ikarashi_Risa       -> Minamoto_Aira
  rp1_1     | Worreby_Mia      vs Vincent_Kethleen    -> Vincent_Kethleen
  rp1_2     | Ritz_Lila        vs Flaschberger_Sarah  -> Ritz_Lila
```

Clés disponibles : `quart_1`…`quart_4`, `demi_1`/`demi_2`, `final_1`, `bronze_1`/`bronze_2`, `rp1_1`/`rp1_2` … `rp4_1`/`rp4_2` (l'ordre `_1`, `_2` = l'ordre dans la base).

**Étape 2 — Générer** en donnant un `--match` par match filmé. Chaque match = `"clé=DEBUT-FIN"`. Si un match est sur une **autre vidéo** que `--url`, préfixe-le par son URL : `"clé=URL|DEBUT-FIN"`.

```powershell
.\.venv\Scripts\python.exe scripts\build_pool_dataset.py `
  --competition SA_ACoruna --year 2026 --sex F --finals `
  --url "https://www.youtube.com/watch?v=VIDEO_TATAMI_A" `
  --match "quart_1=4:13:25-4:20:00" `
  --match "quart_2=4:21:00-4:27:40" `
  --match "demi_1=4:28:54-4:37:11" `
  --match "rp1_1=4:47:00-4:52:00" `
  --match "quart_3=https://www.youtube.com/watch?v=VIDEO_TATAMI_B|1:00:00-1:05:00"
```

Ça produit :
- `data/input/SA_ACoruna_2026_F_finals.csv` (uniquement les matchs listés)
- une ligne `lives` par match : `sa_acoruna_2026_f_final_quart_1`, `..._final_demi_1`, `..._final_rp1_1`, etc.

> **Vidéos multiples** : chaque URL distincte = un fichier local distinct (`..._final_<idYouTube>.mp4`), donc **un téléchargement par tatami**. Les matchs qui partagent une URL partagent le fichier.
> **Une seule commande** : le dataset `_finals.csv` est **réécrit** à chaque exécution (le lives, lui, est complété sans doublon). Passe donc **tous** les `--match` en une fois. Relancer avec une liste partielle écrase les matchs précédents du dataset.
> **Correspondance clé → match** : `_1`, `_2` suivent l'ordre du fichier (vu avec `--finals --list`). Vérifie les noms affichés pour associer le bon timecode au bon match.

Puis, pour générer les clips, une commande **par match final** (avec le dataset `_finals.csv`) :

```powershell
kata-pipeline run-pipeline sa_acoruna_2026_f_final_quart_1 --dataset data/input/SA_ACoruna_2026_F_finals.csv --lives data/input/SA_ACoruna_2026_F_lives.csv
kata-pipeline run-pipeline sa_acoruna_2026_f_final_demi_1  --dataset data/input/SA_ACoruna_2026_F_finals.csv --lives data/input/SA_ACoruna_2026_F_lives.csv
# … idem pour chaque final_<clé> filmé
```

> Chaque URL distincte se télécharge au premier `run-pipeline` qui l'utilise (les suivants réutilisent le fichier).

---

## 2. Générer les clips (Phase 1) — par tour

Une commande par tour. Elle enchaîne : téléchargement → extraction plage → mouvement → segments → alignement → clips.

```powershell
kata-pipeline run-pipeline sa_acoruna_2026_f_pool5_t1 --dataset data/input/SA_ACoruna_2026_F_pool5.csv --lives data/input/SA_ACoruna_2026_F_lives.csv
kata-pipeline run-pipeline sa_acoruna_2026_f_pool5_t2 --dataset data/input/SA_ACoruna_2026_F_pool5.csv --lives data/input/SA_ACoruna_2026_F_lives.csv
kata-pipeline run-pipeline sa_acoruna_2026_f_pool5_t3 --dataset data/input/SA_ACoruna_2026_F_pool5.csv --lives data/input/SA_ACoruna_2026_F_lives.csv
kata-pipeline run-pipeline sa_acoruna_2026_f_pool5_pw1 --dataset data/input/SA_ACoruna_2026_F_pool5.csv --lives data/input/SA_ACoruna_2026_F_lives.csv
```

> Lance **T1 en premier** : c'est lui qui déclenche le téléchargement de la vidéo.

Les clips sortent dans `data/clips/SA/pending/` ou `data/clips/K1/pending/`.
Range ensuite les clips validés dans le même circuit, par exemple
`data/clips/SA/pool_5/` ou `data/clips/K1/pool_1/`. Le chemin réel est enregistré
dans le CSV de clips lors de la génération.

### Étapes une par une (si besoin de débugger un tour)

```powershell
kata-pipeline prepare-video   sa_acoruna_2026_f_pool5_t1 --lives data/input/SA_ACoruna_2026_F_lives.csv
kata-pipeline compute-motion  sa_acoruna_2026_f_pool5_t1
kata-pipeline detect-segments sa_acoruna_2026_f_pool5_t1
kata-pipeline align           sa_acoruna_2026_f_pool5_t1 --dataset data/input/SA_ACoruna_2026_F_pool5.csv --lives data/input/SA_ACoruna_2026_F_lives.csv
kata-pipeline generate-clips  sa_acoruna_2026_f_pool5_t1
```

### Interface de validation (facultatif)

```powershell
kata-pipeline validate
```

---

## 3. Poses + modèle GNN (Phase 2)

Toutes les commandes sont sous `kata-pipeline gnn ...`.

### 3.1 Extraire les poses (squelettes) de tous les clips

```powershell
kata-pipeline gnn extract-poses
# limiter à un circuit :
kata-pipeline gnn extract-poses --competition-type K1
# recalculer même si déjà fait :
kata-pipeline gnn extract-poses --overwrite
```

Par défaut, `pending` est exclu. Les nouvelles poses sont écrites dans
`data/poses/SA/` ou `data/poses/K1/`; les poses historiques directement sous
`data/poses/` restent compatibles.

Si un clip n'est plus présent localement, `extract-poses` peut le restaurer à
la demande depuis l'archive Google Drive décrite à la section 3.2. Une pose
déjà présente n'entraîne aucun téléchargement, sauf avec `--overwrite`.

### 3.2 Restaurer un clip depuis Google Drive

Le connecteur est volontairement en lecture seule. Il utilise un OAuth de type
« application de bureau », un index local fondé sur les identifiants Drive et
un cache LRU limité par défaut à 10 Go.

1. Dans Google Cloud, activer Google Drive API et télécharger le JSON OAuth
   d'une application de bureau.
2. Enregistrer ce fichier sous `.secrets/google_drive_credentials.json`. Ne pas
   le committer et ne pas copier son contenu dans un ticket ou un message.
3. Copier `config/drive.example.yaml` vers `config/drive.local.yaml`, puis
   remplacer `archive.root_folder_id` par l'identifiant du dossier Drive
   `data`. Cet identifiant est la partie située après `/folders/` dans son URL.
4. Installer les dépendances et initialiser l'accès :

```powershell
pip install -e ".[gnn,drive]"
kata-pipeline drive auth --config config/drive.local.yaml
kata-pipeline drive index --config config/drive.local.yaml
kata-pipeline drive status --config config/drive.local.yaml
```

L'arborescence distante attendue est `data/clips/SA/...` et
`data/clips/K1/...`. Le dossier `data` lui-même est désigné par
`root_folder_id`; `remote_clips_path` reste donc `clips`.

Pour contrôler une restauration avant de relancer le pipeline :

```powershell
kata-pipeline drive fetch NOM_DU_CLIP_SANS_EXTENSION --competition-type SA --config config/drive.local.yaml
kata-pipeline gnn extract-poses --config config/drive.local.yaml
```

Les fichiers sont téléchargés dans `data/cache/videos/{SA|K1}`. Leur taille et
leur MD5 Drive sont contrôlés, puis un SHA-256 et la provenance sont inscrits
dans un fichier `.meta.json`. `drive prune` applique manuellement la limite du
cache. Le token OAuth, les secrets, l'index Drive et le cache sont ignorés par
Git.

### 3.3 Entraîner

```powershell
kata-pipeline gnn train-outcome      # modèle "qui gagne" (principal) -> models/outcome.pt
kata-pipeline gnn train-comparator   # comparateur siamois           -> models/comparator.pt
kata-pipeline gnn train-quality      # tête de qualité/score          -> models/quality.pt
# optionnel : n'entraîner qu'un circuit
kata-pipeline gnn train-outcome --competition-type SA
```

Le train et l'évaluation lisent uniquement les CSV de labels et les `.npz` de
poses. Ils ne parcourent plus `data/clips` et continuent donc à fonctionner si
les MP4 sont archivés hors du poste. Sans `--competition-type`, SA et K1 sont
réunis dans le même entraînement.

### 3.4 Évaluer

```powershell
kata-pipeline gnn evaluate-outcome --model models/outcome.pt
```

### 3.5 Prédire une confrontation A vs B

```powershell
kata-pipeline gnn predict-match cheminA.mp4 cheminB.mp4 --model models/outcome.pt
kata-pipeline gnn compare       cheminA.mp4 cheminB.mp4 --model models/comparator.pt
```

### 3.6 Visualisations (squelette + vitesse)

```powershell
kata-pipeline gnn viz-motion cheminClip.mp4                 # un seul clip
kata-pipeline gnn viz-motion-all --competition-type SA --subdir pool_5
kata-pipeline gnn viz-motion-all                            # tous les clips
kata-pipeline gnn viz-motion-all --competition-type K1 --subdir pool_1 --overwrite
```

---

## 4. Récap rapide

| Je veux… | Fichier / Commande |
|---|---|
| **Ajouter une pool** (dataset + lives, lien vidéo + timecodes) | `python scripts\build_pool_dataset.py --competition ... --year ... --sex F --pool N --url ... --t1 ... --t2 ... --t3 ... --pw1 ...` |
| **Lister les matchs de finales** (clés + noms) | `python scripts\build_pool_dataset.py --competition ... --year ... --sex F --finals --list` |
| **Ajouter les phases finales** (par match, multi-tatami) | `python scripts\build_pool_dataset.py --competition ... --year ... --sex F --finals --url ... --match "quart_1=DEBUT-FIN" --match "demi_1=..." ...` |
| Voir les pools disponibles | même script avec `--list` |
| Sauter des matchs non filmés | ajouter `--start-athlete <Nom>` |
| **Générer les clips** | `kata-pipeline run-pipeline <id_live> --dataset ... --lives ...` |
| **Extraire poses / entraîner / prédire** | `kata-pipeline gnn extract-poses` / `train-outcome` / `predict-match` |
