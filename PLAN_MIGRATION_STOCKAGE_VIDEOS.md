# Plan de migration du stockage vidéo

Dernière mise à jour : 2026-09-03

Statut global : **migration terminée et validée (`CHECK PHASE 7`)**

Ce document sert de mémoire de travail pour déplacer les vidéos hors du disque
local, rendre l'entraînement indépendant des clips, puis permettre une
restauration depuis Google Drive.

## Décisions prises

- Les poses continuent d'être extraites à partir des clips vidéo.
- L'entraînement du GNN doit ensuite fonctionner uniquement avec les poses et
  les labels.
- Les vidéos de `data/clips` seront copiées sur Google Drive.
- Les vidéos de lives présentes dans `data/input` seront supprimées localement
  après autorisation explicite.
- Les vidéos de `data/intermediate` seront supprimées localement après
  autorisation explicite.
- Les vidéos de `data/viz` seront archivées sur Google Drive. Elles ne sont pas
  utilisées par l'entraînement : ce sont des rendus dérivés destinés à la
  visualisation.
- Aucune suppression ne doit être effectuée avant la vérification de l'archive.
- Codex ne supprimera aucun fichier sans une instruction explicite de
  l'utilisateur désignant la phase et les catégories à supprimer.

## État initial constaté

| Emplacement | Contenu | Nombre | Volume approximatif |
|---|---|---:|---:|
| `data/clips` | MP4 | 444 | 19,29 Go |
| `data/viz` | MP4 de visualisation | 132 | 32,28 Go |
| `data/input` | MP4 source | 1 | 6,24 Go |
| `data/intermediate` | MP4 intermédiaires | 28 | 8,31 Go |
| `data/poses` | Fichiers NPZ | 443 | 0,16 Go |

Précisions :

- 443 clips correspondent aux 443 fichiers de poses actuellement reconnus par
  le manifeste.
- Un fichier de visualisation suffixé `_motion.mp4` est actuellement rangé dans
  `data/clips`.
- Les CSV contiennent 460 prestations, mais seulement 443 possèdent actuellement
  un clip et une pose disponibles.
- 217 paires gagnant/perdant complètes sont exploitables par l'entraînement
  hiérarchique actuel.
- Les 460 lignes de résultats possèdent une URL YouTube, mais ces URL ne sont
  pas considérées comme une sauvegarde pérenne.

## Règle de passage entre les phases

Une phase n'est considérée comme terminée que lorsque l'utilisateur envoie le
message `CHECK PHASE N` et que ses contrôles sont validés. En cas d'écart, la
phase reste ouverte et aucune suppression dépendante n'est lancée.

---

## Phase 1 — Archivage de `data/clips`

Responsable initial : utilisateur.

Actions :

- [ ] Confirmer l'emplacement exact du dossier d'archive Google Drive.
- [x] Copier toute l'arborescence de `data/clips` en conservant les sous-dossiers
      (`pool_*`, `quart`, `demi`, `repechage`, etc.).
- [ ] Confirmer que les fichiers n'ont pas été renommés pendant la copie.
- [x] Conserver les clips localement à ce stade.
- [ ] Noter dans ce document ou communiquer l'identifiant/l'emplacement du
      dossier Drive.

Contrôle attendu :

- 444 MP4 présents dans l'archive si le dossier est copié tel quel ;
- environ 19,29 Go ;
- arborescence et noms préservés ;
- téléchargement et ouverture réussis d'au moins un clip de plusieurs
  sous-dossiers.

Checkpoint utilisateur : `CHECK PHASE 1`

Statut : **Copie Drive déclarée terminée — validation d'intégrité en phase 3**

---

## Phase 2 — Archivage de `data/viz`

Responsable initial : utilisateur.

Les fichiers de `data/viz` ne sont consommés ni par l'extraction des poses ni
par l'entraînement. Ils peuvent être régénérés à partir des clips et du code,
mais leur archivage conserve les rendus existants sans nouveau calcul.

Actions :

- [x] Copier `data/viz/*.mp4` vers un dossier Drive distinct de l'archive des
      clips sources.
- [ ] Confirmer que les visualisations sont identifiables séparément des clips
      du dataset.
- [x] Conserver les fichiers localement jusqu'à la phase de suppression
      explicitement autorisée.

Contrôle attendu :

- 132 MP4 présents dans l'archive ;
- environ 32,28 Go ;
- téléchargement et lecture réussis de quelques rendus.

Checkpoint utilisateur : `CHECK PHASE 2`

Statut : **Copie Drive déclarée terminée — validation d'intégrité en phase 3**

---

## Phase 3 — Inventaire et vérification de l'archive

Responsable : utilisateur + Codex.

Actions prévues :

- [x] Produire l'inventaire des vidéos importantes avant toute suppression :
      chemin relatif, taille et MD5 Drive vérifié contre le fichier local ; un
      SHA-256 est aussi produit à chaque restauration dans le cache.
- [x] Associer chaque clip à son `drive_file_id` ou à une référence Drive
      stable.
- [x] Comparer les nombres de fichiers et les volumes local/Drive.
- [x] Vérifier les checksums Drive lorsqu'ils sont disponibles.
- [x] Effectuer un test réel de restauration dans un emplacement temporaire.
- [x] Vérifier que les vidéos restaurées peuvent être lues.

Critère de sortie : l'archive des clips est restaurable et son intégrité est
vérifiée. L'archivage de `data/viz` est contrôlé séparément et ne conditionne pas
la validité du dataset de poses.

Checkpoint utilisateur : `CHECK PHASE 3`

Statut : **Terminée — archive distante et intégrité des 444 vidéos vérifiées**

---

## Phase 4 — Nettoyage des vidéos non utilisées par le train

Responsable : Codex, uniquement après autorisation explicite.

Les cibles devront être recomptées et résolues précisément avant suppression.

Suppression prévue :

- [x] uniquement les fichiers vidéo de `data/input` ;
- [x] uniquement les fichiers vidéo de `data/intermediate` ;
- [x] les MP4 de `data/viz`, après autorisation explicite de l'utilisateur et
      validation de leur archive.

À préserver impérativement :

- les CSV et scripts de `data/input` ;
- les JSON de `data/intermediate` ;
- les fichiers `.gitkeep` ;
- tous les fichiers de `data/poses` ;
- tous les clips de `data/clips` jusqu'à la fin de la refonte des manifestes.

Autorisation attendue, par exemple :

`CHECK PHASE 3 — tu peux supprimer les MP4 de data/input et data/intermediate`

Checkpoint après opération : `CHECK PHASE 4`

Statut : **Terminée — MP4 de `data/input`, `data/intermediate` et `data/viz`
supprimés, 46,820 Go libérés au total**

---

## Phase 5 — Séparation des manifestes extraction/entraînement

Responsable : Codex après demande d'implémentation.

Objectif : supprimer la dépendance artificielle de l'entraînement à la présence
locale de `data/clips/**/*.mp4`.

### Manifeste d'extraction

Il doit relier les labels aux vidéos et aux poses afin de déterminer quelles
poses doivent être produites ou recalculées.

Entrées envisagées :

- répertoire ou stockage des clips ;
- répertoire des poses ;
- CSV de labels ;
- futur résolveur Google Drive facultatif.

### Manifeste d'entraînement

Il doit relier directement les labels aux fichiers de poses, sans parcourir ni
tester l'existence des MP4.

Entrées envisagées :

- `output_dir` ou manifeste de labels ;
- `poses_dir` ;
- éventuellement un manifeste d'archive utilisé uniquement comme métadonnée de
  traçabilité.

Actions techniques :

- [x] introduire deux responsabilités clairement séparées ;
- [x] faire utiliser le manifeste de poses par le train et l'évaluation ;
- [x] réserver le manifeste vidéo à `extract-poses` et aux opérations vidéo ;
- [ ] conserver `drive_file_id`, hash et provenance comme métadonnées, sans les
      rendre nécessaires au train — reporté à la phase 6, avec le connecteur ;
- [x] ajouter les tests de non-régression correspondants.

Extension SA/K1 intégrée à cette phase :

- [x] lire et propager `Type_Compet` sous le nom interne `competition_type` ;
- [x] centraliser les tours SA (`T1/T2/T3/PW1`) et K1
      (`Pool_1/Pool_2/Pool_3`) ainsi que leurs phases finales ;
- [x] écrire les nouveaux clips dans `data/clips/{SA|K1}/pending` ;
- [x] écrire les nouvelles poses dans `data/poses/{SA|K1}` ;
- [x] conserver la lecture des anciens clips et poses à plat ;
- [x] permettre `--competition-type SA|K1` pour extraction, train, évaluation
      et visualisations en lot ;
- [x] exclure `pending` et les MP4 dérivés de l'extraction par défaut.
- [x] déplacer physiquement les dossiers de clips historiques sous
      `data/clips/SA/` et créer `data/clips/K1/` — action annoncée par
      l'utilisateur, suivie d'un contrôle Codex.

Critères de sortie :

- le manifeste d'entraînement retrouve 443 poses même si `clips_dir` est absent ;
- les 217 paires complètes restent exploitables ;
- le split reste reproductible avec le même seed ;
- aucun MP4 n'est ouvert pendant le train ou l'évaluation ;
- `extract-poses` continue d'exiger une vidéo locale ou restaurable.

Checkpoint utilisateur : `CHECK PHASE 5`

Statut : **Terminée — code et déplacement physique SA vérifiés**

---

## Phase 6 — Connecteur Google Drive et cache local

Responsable : Codex après validation de l'architecture et fourniture du mode
d'authentification choisi.

Objectif : restaurer automatiquement une vidéo seulement lorsqu'une opération
vidéo en a besoin.

Interface cible :

```text
video_ref / drive_file_id
          │
          ▼
GoogleDriveVideoStore.resolve()
          │
          ▼
cache local vérifié par checksum
          │
          ▼
OpenCV / FFmpeg / extracteur de poses
```

Actions prévues :

- [x] choisir OAuth utilisateur pour le Drive personnel ;
- [x] stocker les secrets hors du dépôt (`.secrets/`, ignoré par Git) ;
- [x] implémenter l'indexation récursive, le téléchargement et la vérification ;
- [x] utiliser les identifiants Drive plutôt que les noms comme référence
      principale ;
- [x] ajouter un cache local LRU limité par défaut à 10 Go ;
- [x] intégrer la restauration à la demande à `gnn extract-poses`, sans aucun
      accès Drive depuis le train ou l'évaluation ;
- [x] ajouter une CLI `drive auth|index|status|fetch|prune`, la configuration
      d'exemple, la documentation et des tests sans réseau ;
- [x] tester une ré-extraction depuis un clip restauré dans le cache Drive.

Paramètres retenus :

- Drive personnel, dossier racine distant `data` ;
- arborescence distante miroir, notamment `data/clips/SA` et
  `data/clips/K1` ;
- OAuth application de bureau et scope Drive en lecture seule ;
- credentials : `.secrets/google_drive_credentials.json` ;
- token local : `.secrets/google_drive_token.json` ;
- index : `data/archive/drive_videos.csv` ;
- cache : `data/cache/videos`, maximum 10 Go avec éviction LRU ;
- contrôle de la taille et du MD5 Drive au téléchargement, plus SHA-256 local
  et fichier de provenance `.meta.json`.
- garantie budgétaire : utiliser un projet Google Cloud dédié sans compte de
  facturation associé, ne demander aucune hausse de quota et interrompre la
  configuration si Google exige l'activation de la facturation.

Pour terminer le test réel, il manque uniquement :

1. ~~l'identifiant exact (ou le lien) du dossier Drive `data`~~ — reçu et
   configuré : `11mbVlrNKUahtoKXB9lTkIVJcFYoabSKZ` ;
2. le fichier OAuth « application de bureau » placé localement dans
   `.secrets/google_drive_credentials.json` — son contenu ne doit pas être
   communiqué dans ce document ni dans un message ;
3. ~~confirmer dans Google Cloud `Facturation` que le projet affiche qu'aucun
   compte de facturation ne lui est associé~~ — confirmé par l'utilisateur :
   `This project has no billing account`.

Sous-phases de validation réelle :

- [x] **6A — préparation utilisateur** : communiquer le lien ou l'ID du dossier
  `data`, déposer le JSON OAuth au chemin prévu, puis répondre
  `CHECK PHASE 6A` ;
- [x] **6B — authentification et indexation Codex** : écrire la configuration
  locale, lancer OAuth, reconstruire l'index et contrôler les nombres SA/K1,
  les identifiants et les checksums, sans supprimer de clip ;
- [x] **6C — restauration Codex** : télécharger un clip témoin dans le cache,
  vérifier taille/checksums/lecture vidéo, puis valider le chemin de
  ré-extraction d'une pose dans un emplacement isolé ;
- **sortie** : consigner les résultats ci-dessous, puis attendre
  `CHECK PHASE 6` avant d'ouvrir la phase 7.

Checkpoint utilisateur : `CHECK PHASE 6`

Statut : **Terminée — validation réelle réussie et `CHECK PHASE 6` reçu**

---

## Phase 7 — Éviction des clips locaux

Responsable : Codex, uniquement après autorisation explicite.

Préconditions :

- [x] archive Drive vérifiée ;
- [x] manifeste d'entraînement indépendant des MP4 ;
- [x] train et évaluation testés sans `clips_dir` ;
- [x] méthode de restauration documentée et testée.

Actions :

- [x] recomptage final de `data/clips` ;
- [x] suppression locale explicitement autorisée et exécutée ;
- [x] conservation du cache Drive LRU limité à 10 Go, séparé sous
      `data/cache/videos` ;
- [x] dernier contrôle du manifeste d'entraînement avec un `clips_dir`
      inexistant : 443 poses, 217 paires et aucune colonne vidéo.

Recomptage final avant autorisation :

- cible potentielle : uniquement les 444 fichiers vidéo sous
  `data/clips/SA`, soit 20 708 430 779 octets (20,708 Go décimaux / 19,286 Gio) ;
- `data/clips/K1` : 0 vidéo ;
- `pending` : 0 vidéo ;
- contenu cible : 443 clips utiles et un rendu dérivé `_motion`, tous présents
  dans l'index Drive avec une taille et un MD5 identiques ;
- à préserver : toute l'arborescence, les deux `.gitkeep`, `data/poses`, les
  CSV/labels, `data/archive/drive_videos.csv`, les secrets OAuth et le clip
  témoin du cache ;
- hors cible : `data/cache/videos` contient un clip restauré de 15 506 905
  octets (14,79 Mio) et reste soumis à sa limite LRU de 10 Go ;
- `data/input`, `data/intermediate` et `data/viz` contiennent déjà 0 vidéo ;
- espace libre avant suppression : 190 250 917 888 octets (177,18 Gio) ;
- gain théorique : 19,286 Gio, soit environ 196,47 Gio libres après opération.

Autorisation exacte attendue si l'utilisateur décide de poursuivre :

`AUTORISATION PHASE 7 — supprime uniquement les 444 fichiers vidéo de data/clips ; préserve l'arborescence, les .gitkeep, les poses, l'index Drive, les secrets OAuth et le cache`

Checkpoint utilisateur : `CHECK PHASE 7`

Statut : **Terminée — suppression contrôlée et `CHECK PHASE 7` reçu**

---

## État d'avancement synthétique

| Phase | Description | Statut |
|---:|---|---|
| 1 | Archivage de `data/clips` | Copie terminée, validation en phase 3 |
| 2 | Archivage de `data/viz` | Copie terminée, validation en phase 3 |
| 3 | Inventaire et vérification | Terminée, 444 MD5 conformes |
| 4 | Nettoyage input/intermediate/viz | Terminé — 46,820 Go libérés |
| 5 | Refonte des manifestes + séparation SA/K1 | Terminée et vérifiée |
| 6 | Connecteur Drive et cache | Terminée et validée |
| 7 | Suppression locale des clips | Terminée et validée |

## Journal

- 2026-09-03 : audit initial réalisé, stratégie validée et plan créé.
- 2026-09-03 : l'utilisateur confirme la copie de `data/clips` et `data/viz`
  sur Google Drive. Contrôle local après copie : 444 clips (19,286 Go) et 132
  visualisations (32,276 Go) toujours présents. Vérification distante à faire.
- 2026-09-03 : l'utilisateur confirme que l'archive Drive est correcte et que
  les tests manuels de téléchargement et de lecture sont réussis. La génération
  d'un inventaire SHA-256 complet reste prévue avant la suppression locale des
  clips à la phase 7.
- 2026-09-03 : après autorisation explicite, suppression de 29 MP4 : 1 fichier
  dans `data/input` et 28 fichiers dans `data/intermediate`, soit 14,544 Go.
  Contrôle après suppression : 118 JSON intermédiaires, 4 CSV et 1 script dans
  `data/input`, 444 clips, 132 vidéos `data/viz` et 443 poses préservés.
- 2026-09-03 : contrôle fonctionnel après nettoyage : le manifeste retrouve
  toujours 443 prestations, les 443 poses existent et les 217 paires restent
  exploitables.
- 2026-09-03 : après autorisation explicite complémentaire, suppression des 132
  MP4 archivés de `data/viz`, soit 32,276 Go supplémentaires. Contrôle après
  suppression : aucun MP4 dans `data/viz`, 444 clips et 443 poses préservés.
  Le nettoyage de la phase 4 a libéré 46,820 Go au total.
- 2026-09-03 : phase 5 implémentée. Le manifeste d'entraînement fonctionne avec
  un `clips_dir` inexistant et retrouve 443 poses ainsi que 217 paires, sans
  colonne ni test de chemin vidéo. Le manifeste d'extraction retrouve séparément
  443 clips/poses et exclut `pending`/les rendus dérivés. Les profils SA et K1,
  leurs tours, les nouveaux dossiers typés et les filtres CLI ont été ajoutés.
  Test réel de la base : K1 Paris 2024 F produit 8 pools et 9 matchs de phases
  finales ; une incohérence de tour rouge/bleu présente dans la source est
  signalée sans créer une fausse pool.
- 2026-09-03 : `CHECK PHASE 5` reçu après déplacement utilisateur. Contrôle
  réussi : 444 MP4 sous `data/clips/SA`, 0 sous `data/clips/K1`, aucun MP4 ni
  ancien dossier au premier niveau de `data/clips`, et `pending` vide. Le MP4
  `_motion` historique est correctement exclu. Après déplacement, le manifeste
  d'extraction retrouve 443 clips et 443 poses, tous typés SA ; le manifeste
  d'entraînement, testé avec un `clips_dir` inexistant, retrouve toujours 443
  poses et 217 paires sans colonne vidéo.
- 2026-09-03 : phase 6 implémentée pour un Drive personnel. Ajout d'un client
  OAuth en lecture seule, d'un index récursif de `data/clips/SA|K1` conservant
  les `drive_file_id`, tailles, MD5 et chemins distants, ainsi que d'un cache
  LRU local de 10 Go avec SHA-256 et provenance. `extract-poses` restaure à la
  demande uniquement une vidéo dont la pose manque ; le train et l'évaluation
  restent totalement indépendants des vidéos et de Drive. La CLI
  `drive auth|index|status|fetch|prune`, les dépendances optionnelles, les
  exemples de configuration, la documentation et les tests simulés ont été
  ajoutés. Il reste à fournir l'ID du dossier Drive `data`, à déposer le JSON
  OAuth sous `.secrets/google_drive_credentials.json`, puis à exécuter un test
  réel d'authentification, d'indexation et de restauration.
- 2026-09-03 : validation locale de la phase 6 : imports Google opérationnels,
  13 tests ciblés réussis, compilation et lint ciblé réussis, commandes
  `kata-pipeline drive` chargées correctement. La suite complète obtient 98
  tests réussis et conserve un seul échec préexistant, sans rapport avec Drive,
  dans `TestPairSegments.test_extra_segments` (segments supplémentaires non
  reportés comme non associés).
- 2026-09-03 : URL du dossier Drive personnel `data` reçue. Son identifiant
  `11mbVlrNKUahtoKXB9lTkIVJcFYoabSKZ` est enregistré dans la configuration
  locale ignorée par Git `config/drive.local.yaml`. Le dépôt du JSON OAuth et
  la vérification de l'absence de facturation restent à faire avant
  l'authentification et l'indexation réelles.
- 2026-09-03 : à la demande de l'utilisateur, ajout d'un verrou de coût à la
  phase 6. Le projet Google Cloud consacré au connecteur doit rester sans compte
  de facturation associé et aucune hausse de quota ne sera demandée. Si Google
  exige une activation de facturation, la procédure doit être interrompue avant
  de poursuivre.
- 2026-09-03 : verrou budgétaire contrôlé par l'utilisateur ; Google Cloud
  affiche `This project has no billing account`. La configuration OAuth est en
  cours : l'écran `Audience` signale que les informations obligatoires de
  `Branding` doivent d'abord être complétées.
- 2026-09-03 : `CHECK PHASE 6A` reçu. Le fichier OAuth, initialement nommé avec
  une double extension `.json.json`, a été renommé localement sans afficher son
  contenu. Sa structure `Desktop app` a été validée. OAuth a abouti avec le
  scope Drive en lecture seule et Google a confirmé l'accès au dossier `data`
  attendu (`11mbVlrNKUahtoKXB9lTkIVJcFYoabSKZ`).
- 2026-09-03 : phase 6B validée. L'index distant contient 444 vidéos, toutes SA,
  pour 19,286 Go : 443 clips exploitables et un rendu dérivé exclu. Les 444
  `drive_file_id` sont uniques, les 444 MD5 sont présents et chaque taille
  distante correspond à la taille locale. Le recalcul complet des MD5 locaux a
  confirmé 444 correspondances sur 444, sans fichier absent ni différence.
- 2026-09-03 : phase 6C validée avec le clip témoin
  `SA_Tbilisi_Female_Kata_T2_m02_blue_Khamis_Jana_Enpi.mp4` (14,79 Mio).
  Téléchargement Drive vers le cache réussi, MD5 et SHA-256 conformes, fichier
  de provenance présent, lecture OpenCV réussie en 1920x1080 à 60 fps. Une
  extraction MediaPipe isolée a produit 699 frames de pose à 10 fps sur 15
  articulations, avec 86,84 % de points valides. Le NPZ temporaire a ensuite été
  supprimé ; le clip témoin reste dans le cache LRU (0,01/10 Go).
- 2026-09-03 : contrôle final sans réécriture : `gnn extract-poses` retrouve 443
  éléments, ignore correctement les 443 poses déjà présentes, ne restaure rien
  inutilement et ne produit aucun échec. Le manifeste de train conserve 443
  lignes et aucune colonne `clip_path`.
- 2026-09-03 : `CHECK PHASE 6` reçu. La phase 6 est officiellement clôturée.
  La phase 7 devient techniquement disponible, mais aucune suppression de clip
  local n'est autorisée par ce checkpoint ; elle nécessitera une instruction
  explicite distincte après présentation du recomptage final.
- 2026-09-03 : phase 7 ouverte en lecture seule. Recomptage final : 444 MP4 sous
  `data/clips/SA`, 0 sous K1 et `pending`, pour 20 708 430 779 octets. La cible
  contient 443 clips utiles et un rendu `_motion`; les deux `.gitkeep` sont hors
  cible. L'index Drive présente les mêmes 444 tailles et MD5. `data/input`,
  `data/intermediate` et `data/viz` restent sans vidéo. Le cache séparé conserve
  uniquement le clip témoin de 14,79 Mio. Avec un `clips_dir` volontairement
  inexistant, le manifeste de train retrouve 443 poses, 217 paires et aucune
  colonne vidéo. Aucune suppression effectuée ; autorisation explicite attendue.
- 2026-09-03 : autorisation explicite de phase 7 reçue pour supprimer uniquement
  les 444 vidéos sous `data/clips` tout en préservant l'arborescence, les
  `.gitkeep`, poses, index Drive, secrets OAuth et cache. Une première garde a
  refusé l'opération avant suppression à cause d'une normalisation locale
  `data/clips/...` contre distante `clips/...`; aucun fichier n'avait alors été
  touché. Après correction de cette comparaison, la garde a confirmé les 444
  cibles, leurs 20 708 430 779 octets et leurs correspondances exactes dans
  l'index Drive. Les 444 vidéos ont ensuite été supprimées, sans reste.
- 2026-09-03 : contrôles après éviction réussis. `data/clips` contient 0 vidéo
  et conserve ses dossiers ainsi que les deux `.gitkeep`; `data/clips/K1` est
  toujours présent. Les 443 poses, l'index de 444 références, les credentials,
  le token OAuth et le clip témoin du cache sont présents. Le manifeste
  d'extraction expose 442 sources `google_drive` et une source `cache`; la
  commande `gnn extract-poses` ignore les 443 poses existantes, sans
  téléchargement ni échec. Le manifeste de train conserve 443 lignes, 217
  paires et aucune colonne vidéo. L'espace libre atteint 210 960 400 384 octets
  (environ 196,47 Gio).
- 2026-09-03 : `CHECK PHASE 7` reçu. La phase 7 et le plan de migration du
  stockage vidéo sont officiellement clôturés. État final : entraînement fondé
  uniquement sur les poses, extraction capable de restaurer les clips à la
  demande depuis Drive, archive de 444 vidéos vérifiée, cache LRU de 10 Go et
  aucun MP4 conservé sous `data/clips`.
- 2026-09-07 : correction d'un téléchargement YouTube K1 interrompu par HTTP
  403. `yt-dlp` a été mis à jour de `2026.07.04` vers `2026.08.19` et la version
  minimale du profil `download` a été alignée dans `pyproject.toml`. Le fragment
  incomplet de 10 298 722 octets sous `data/input` a été supprimé après contrôle
  de son chemin. Un téléchargement isolé de 10 secondes avec le même format que
  la pipeline a réussi : MP4 de 10,008 s, AV1 1920x1080 à 60 fps + AAC, lisible
  au début et à 5 s. Le dossier de test a ensuite été supprimé. La pipeline
  complète n'a pas été relancée ; le live source dure 10 h 58 min et représente
  environ 9,5 Go avec les formats sélectionnés.
- 2026-09-08 : correction de l'ajout de lignes dans les fichiers
  `*_lives.csv` par `scripts/build_pool_dataset.py`. Si le CSV avait été
  sauvegardé sans saut de ligne final, `csv.writer` concaténait auparavant la
  première nouvelle ligne à la dernière cellule existante. Les ajouts de pools
  et de finales garantissent désormais une fin de ligne avant toute écriture ;
  un fichier vide reçoit aussi correctement son en-tête. Deux tests de
  non-régression ont été ajoutés. Le CSV K1 Istanbul corrigé a été contrôlé :
  13 lignes de données, 12 colonnes par ligne, aucune ligne mal formée. Tests
  ciblés : 4 réussis ; suite complète : 100 réussis et l'unique échec
  préexistant `TestPairSegments.test_extra_segments`, sans rapport avec cette
  correction.
- 2026-09-10 : correction des métadonnées du live K1 Istanbul dans
  `data/input/K1_Istanbul_2026_F_lives.csv`. La ligne `final_r2_2` associait
  l'URL du deuxième live (`quRzO_IqMf4`) au fichier local du troisième live
  (`k1_istanbul_2026_f_pool5.mp4`). L'URL a été remplacée par celle correspondant
  à ce fichier (`5wBc7pA1g14`). Contrôle final : 23 lignes, 23 `id_live`
  uniques, 12 colonnes par ligne, aucune correspondance ambiguë entre URL et
  chemin local. Le fichier `K1_Istanbul_2026_F_finals.csv` n'a pas été modifié.
- 2026-09-10 : rétablissement de
  `data/input/K1_Istanbul_2026_F_finals.csv` depuis la base sportive. Le fichier,
  qui ne conservait plus que `final_r1_3`, contient désormais les cinq matchs
  filmés déclarés dans le fichier lives : `final_r1_1`, `final_r1_2`,
  `final_r1_3`, `final_r2_1` et `final_r2_2`. Les athlètes, katas, styles,
  drapeaux et vainqueurs ont été reconstruits depuis `Database_K1_SA.csv`.
  Contrôle par les chargeurs de la pipeline : 5 lignes valides et aucune section
  manquante ou supplémentaire entre le dataset des finales et le fichier lives.
