"""Génère automatiquement un dataset de pool au format pipeline.

Lit la base historique (``Database_K1_SA.csv``, séparateur ``;``) et produit le
CSV attendu par la pipeline :

    competition,category,round,match_order,athlete_red,athlete_blue,
    kata_red,kata_blue,style_red,style_blue,flag_result_red,flag_result_blue,
    winner,decision_type,section,competition_type

Les pools ne sont pas étiquetées dans la base : elles se suivent dans l'ordre.
Leur premier tour est ``T1`` pour SA et ``Pool_1`` pour K1. Le profil est déduit
de ``Type_Compet`` et le script sélectionne le bloc numéro ``--pool``.

Exemples
--------
Lister les pools détectées (pour vérifier la numérotation) ::

    python scripts/build_pool_dataset.py --competition SA_ACoruna --year 2026 --sex F --list

Générer la pool 5 en excluant les 2 premiers T1 non filmés ::

    python scripts/build_pool_dataset.py --competition SA_ACoruna --year 2026 \
        --sex F --pool 5 --start-athlete Mrakovic_Mila
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import namedtuple
from pathlib import Path

from kata_pipeline.competition_formats import (
    COMPETITION_FORMATS,
    CompetitionFormat,
    CompetitionType,
    get_competition_format,
    normalize_competition_type,
    round_slug,
)

# Alias SA conservés pour les imports et commandes historiques.
SA_FORMAT = COMPETITION_FORMATS["SA"]
POOL_ROUNDS = list(SA_FORMAT.pool_rounds)
POOL_ROUND_SET = set(POOL_ROUNDS)

# Tours des phases finales (hors poule), dans l'ordre chronologique habituel.
# Chaque code de la base est associé à un nom logique (suffixe de section / id_live).
FINALS_ROUNDS = list(SA_FORMAT.finals_rounds)
FINALS_ROUND_SET = set(FINALS_ROUNDS)
FINALS_LABELS = SA_FORMAT.finals_labels

# Un match de phase finale : sa clé logique (``quart_1``, ``rp2_2``…), le tour
# de la base, l'ordre dans le tour (1-based), et les deux athlètes appariés.
FinalsMatch = namedtuple("FinalsMatch", "key label order red blue winner")

SEX_TO_CATEGORY = {"F": "Female Kata", "M": "Male Kata"}
SEX_ALIASES = {"F": {"F"}, "M": {"M", "H"}}

OUTPUT_HEADER = [
    "competition", "category", "round", "match_order",
    "athlete_red", "athlete_blue", "kata_red", "kata_blue",
    "style_red", "style_blue", "flag_result_red", "flag_result_blue",
    "winner", "decision_type", "section", "competition_type",
]

LIVES_HEADER = [
    "id_live", "url", "local_path", "competition", "category", "section",
    "useful_start", "useful_end", "analysis_quality", "clip_quality", "status",
    "competition_type",
]


def _fmt_flag(value: str) -> str:
    """Convertit ``"4.0"`` -> ``"4"`` ; laisse le reste tel quel."""
    value = (value or "").strip()
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return value


def parse_timecode(tc: str) -> int:
    """Convertit ``"mm:ss"`` ou ``"h:mm:ss"`` (ou secondes) en secondes."""
    tc = tc.strip()
    parts = tc.split(":")
    if len(parts) == 1:
        return int(float(parts[0]))
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"Timecode invalide : {tc!r}")


def parse_range(rng: str) -> tuple[int, int]:
    """Convertit ``"15:28-1:00:45"`` en ``(928, 3645)`` (secondes)."""
    if "-" not in rng:
        raise ValueError(f"Plage invalide (attendu DEBUT-FIN) : {rng!r}")
    start, end = rng.split("-", 1)
    return parse_timecode(start), parse_timecode(end)


def parse_round_spec(spec: str) -> tuple[str, tuple[int, int]]:
    """Analyse ``TOUR=DEBUT-FIN`` pour les tours propres à SA ou K1."""

    if "=" not in spec:
        raise ValueError(f"Spec --round invalide (attendu TOUR=PLAGE) : {spec!r}")
    round_name, raw_range = spec.split("=", 1)
    round_name = round_name.strip()
    if not round_name:
        raise ValueError(f"Nom de tour vide dans --round : {spec!r}")
    return round_name, parse_range(raw_range)


def youtube_id(url: str) -> str:
    """Extrait l'identifiant d'une URL YouTube (pour nommer le fichier local)."""
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/)([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_-]", "", url)[-11:] or "video"


def parse_match_spec(spec: str) -> tuple[str, str | None, tuple[int, int]]:
    """Analyse une spec ``--match`` : ``"quart_1=DEBUT-FIN"`` ou
    ``"quart_1=URL|DEBUT-FIN"``.

    Renvoie ``(cle, url_ou_None, (debut, fin))``. L'URL est optionnelle : si
    absente, le match utilise l'``--url`` global.
    """
    if "=" not in spec:
        raise ValueError(f"Spec --match invalide (attendu CLE=PLAGE) : {spec!r}")
    key, rhs = spec.split("=", 1)
    key = key.strip()
    if "|" in rhs:
        url, rng = rhs.split("|", 1)
        url = url.strip() or None
    else:
        url, rng = None, rhs
    return key, url, parse_range(rng)



def load_rows(
    database: Path,
    competition: str,
    year: str,
    sex: str,
    competition_type: CompetitionType | None = None,
) -> list[dict]:
    """Charge les lignes de la base filtrées par compétition / année / sexe."""
    accepted_sex = SEX_ALIASES[sex]
    with database.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        rows = []
        for row in reader:
            row_type = normalize_competition_type(
                row.get("Type_Compet"), competition=row.get("Competition")
            )
            if (row.get("Competition", "").strip() == competition
                    and str(row.get("Year", "")).strip().split(".")[0] == year
                    and row.get("Sexe", "").strip().upper() in accepted_sex
                    and (competition_type is None or row_type == competition_type)):
                rows.append(row)
    return rows


def detect_competition_type(
    rows: list[dict], requested: str | None, competition: str
) -> CompetitionType:
    """Valide le type demandé ou l'infère sans mélanger SA et K1."""

    explicit = normalize_competition_type(requested, competition=competition)
    detected = {
        value
        for row in rows
        if (value := normalize_competition_type(
            row.get("Type_Compet"), competition=row.get("Competition")
        )) is not None
    }
    if len(detected) > 1:
        raise ValueError(f"Plusieurs Type_Compet trouvés dans la sélection : {sorted(detected)}")
    source_type = next(iter(detected), None)
    if explicit is not None and source_type is not None and explicit != source_type:
        raise ValueError(
            f"--type-compet={explicit} contredit Type_Compet={source_type} dans la base"
        )
    result = explicit or source_type
    if result is None:
        raise ValueError("Type de compétition indéterminable ; préciser --type-compet SA|K1")
    return result


def split_into_pools(
    rows: list[dict], rounds: list[str] = POOL_ROUNDS
) -> list[list[dict]]:
    """Découpe les lignes (tours de pool uniquement) en blocs de pool.

    Un nouveau bloc démarre au premier passage rouge du tour initial après un
    autre tour. Le contrôle de la ceinture évite qu'une anomalie isolée sur la
    ligne bleue ne crée artificiellement une nouvelle pool.
    """
    round_set = set(rounds)
    start_round = rounds[0]
    pool_rows = [r for r in rows if r.get("N_Tour", "").strip() in round_set]
    pools: list[list[dict]] = []
    prev_round: str | None = None
    for row in pool_rows:
        current = row["N_Tour"].strip()
        belt = row.get("Ceinture", "").strip().upper()
        if current == start_round and belt == "R" and prev_round != start_round:
            pools.append([])
        if pools:
            pools[-1].append(row)
        prev_round = current
    return pools


def pair_rows(
    pool: list[dict], rounds: list[str] = POOL_ROUNDS
) -> dict[str, list[tuple[dict, dict]]]:
    """Associe les lignes adjacentes Rouge/Bleu, puis les classe par tour."""
    matches: dict[str, list[tuple[dict, dict]]] = {}
    eligible = [row for row in pool if row.get("N_Tour", "").strip() in set(rounds)]
    for i in range(0, len(eligible) - 1, 2):
        a, b = eligible[i], eligible[i + 1]
        red, blue = (a, b) if a["Ceinture"].strip().upper() == "R" else (b, a)
        red_round = red["N_Tour"].strip()
        blue_round = blue["N_Tour"].strip()
        if red_round != blue_round:
            print(
                f"  [!] Tours incohérents pour {red['Nom'].strip()} / "
                f"{blue['Nom'].strip()} ({red_round} vs {blue_round}) ; "
                f"tour rouge {red_round} retenu.",
                file=sys.stderr,
            )
        matches.setdefault(red_round, []).append((red, blue))
    if len(eligible) % 2 != 0:
        print(
            f"  [!] Nombre impair de lignes ({len(eligible)}), dernière ligne ignorée.",
            file=sys.stderr,
        )
    return matches


def build_dataset(matches: dict[str, list[tuple[dict, dict]]], competition: str,
                  category: str, pool_number: int, start_athlete: str | None,
                  end_athlete: str | None, rounds: list[str] = POOL_ROUNDS,
                  competition_type: CompetitionType = "SA") -> list[list[str]]:
    """Construit les lignes du CSV de sortie."""
    out_rows: list[list[str]] = []
    started = start_athlete is None
    for rnd in rounds:
        pairs = matches.get(rnd, [])
        order = 0
        for red, blue in pairs:
            # Filtre optionnel : on ne commence qu'à partir de start_athlete (T1).
            if not started:
                if red["Nom"].strip() == start_athlete or blue["Nom"].strip() == start_athlete:
                    started = True
                else:
                    continue
            order += 1
            winner = red["Nom"].strip() if red["Victoire"].strip().upper() == "VRAI" \
                else blue["Nom"].strip()
            out_rows.append([
                competition,
                category,
                rnd,
                str(order),
                red["Nom"].strip(),
                blue["Nom"].strip(),
                red["Kata"].strip(),
                blue["Kata"].strip(),
                red["Style"].strip(),
                blue["Style"].strip(),
                _fmt_flag(red["Drapeau"]),
                _fmt_flag(blue["Drapeau"]),
                winner,
                "flag",
                f"pool_{pool_number}_{round_slug(rnd)}",
                competition_type,
            ])
            if end_athlete and (red["Nom"].strip() == end_athlete
                                or blue["Nom"].strip() == end_athlete):
                return out_rows
    return out_rows


def enumerate_finals_matches(
    rows: list[dict],
    rounds: list[str] = FINALS_ROUNDS,
    labels: dict[str, str] = FINALS_LABELS,
) -> list[FinalsMatch]:
    """Énumère les matchs de phase finale, dans l'ordre du fichier.

    Chaque match reçoit une clé logique ``<label>_<order>`` (``quart_1``,
    ``demi_2``, ``rp3_1``…) où ``order`` est son rang (1-based) dans son tour.
    L'appariement Rouge/Bleu suit l'ordre du CSV (voir :func:`pair_rows`).
    """
    round_set = set(rounds)
    finals_rows = [r for r in rows if r.get("N_Tour", "").strip() in round_set]
    matches = pair_rows(finals_rows, rounds)

    result: list[FinalsMatch] = []
    for rnd in rounds:
        label = labels[rnd]
        for order, (red, blue) in enumerate(matches.get(rnd, []), start=1):
            winner = red["Nom"].strip() if red["Victoire"].strip().upper() == "VRAI" \
                else blue["Nom"].strip()
            result.append(FinalsMatch(f"{label}_{order}", label, order,
                                      red, blue, winner))
    return result


def finals_dataset_rows(matches: list[FinalsMatch], filmed_keys: set[str],
                        competition: str, category: str,
                        competition_type: CompetitionType = "SA") -> list[list[str]]:
    """Construit les lignes du CSV pour les seuls matchs filmés (``filmed_keys``).

    La section vaut ``final_<key>`` (ex. ``final_quart_1``), ce qui correspond
    à l'``id_live`` créé dans le fichier lives.
    """
    out_rows: list[list[str]] = []
    for m in matches:
        if m.key not in filmed_keys:
            continue
        out_rows.append([
            competition,
            category,
            m.red["N_Tour"].strip(),
            str(m.order),
            m.red["Nom"].strip(),
            m.blue["Nom"].strip(),
            m.red["Kata"].strip(),
            m.blue["Kata"].strip(),
            m.red["Style"].strip(),
            m.blue["Style"].strip(),
            _fmt_flag(m.red["Drapeau"]),
            _fmt_flag(m.blue["Drapeau"]),
            m.winner,
            "flag",
            f"final_{m.key}",
            competition_type,
        ])
    return out_rows


def summarize(pools: list[list[dict]], rounds: list[str] = POOL_ROUNDS) -> None:
    """Affiche un récapitulatif des pools détectées."""
    print(f"{len(pools)} pool(s) détectée(s) :")
    for idx, pool in enumerate(pools, start=1):
        paired = pair_rows(pool, rounds)
        counts = {r: len(paired.get(r, [])) for r in rounds}
        first = pool[0]["Nom"].strip() if pool else "?"
        detail = " ".join(f"{r}:{counts[r]}" for r in rounds)
        print(f"  pool {idx:>2} | 1er athlète (rouge) : {first:<30} | matchs {detail}")


def _lives_has_competition_type(lives_path: Path) -> bool:
    """Indique si une colonne type peut être écrite sans décaler un ancien CSV."""

    if not lives_path.exists() or lives_path.stat().st_size == 0:
        return True
    with lives_path.open("r", encoding="utf-8", newline="") as fh:
        return "competition_type" in (next(csv.reader(fh), []))


def _append_lives_rows(lives_path: Path, rows: list[list[str]]) -> None:
    """Append CSV rows without merging them into an unterminated last line.

    Some spreadsheet applications and editors save CSV files without a final
    newline. Appending directly would then place the first new row in the last
    existing cell. The binary check also avoids newline translation ambiguity
    on Windows.
    """

    lives_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = lives_path.exists() and lives_path.stat().st_size > 0

    if has_content:
        with lives_path.open("rb+") as fh:
            fh.seek(-1, 2)
            if fh.read(1) not in (b"\r", b"\n"):
                fh.seek(0, 2)
                fh.write(b"\r\n")

    with lives_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not has_content:
            writer.writerow(LIVES_HEADER)
        writer.writerows(rows)


def append_lives(lives_path: Path, competition: str, year: str, sex: str,
                 pool_number: int, url: str, video_name: str,
                 ranges: dict[str, tuple[int, int]], rounds_present: list[str],
                 rounds: list[str] = POOL_ROUNDS,
                 competition_type: CompetitionType = "SA") -> None:
    """Ajoute (ou met à jour) les lignes du fichier lives pour la pool.

    Une ligne est écrite par tour présent à la fois dans le dataset et dans les
    timecodes fournis. Les ``id_live`` déjà présents sont ignorés.
    """
    prefix = f"{competition.lower()}_{year}_{sex.lower()}_pool{pool_number}"
    category = SEX_TO_CATEGORY[sex]
    local_path = f"data/input/{video_name}"

    existing_ids: set[str] = set()
    file_exists = lives_path.exists()
    write_type = _lives_has_competition_type(lives_path)
    if file_exists:
        with lives_path.open("r", encoding="utf-8", newline="") as fh:
            existing_ids = {row["id_live"].strip()
                            for row in csv.DictReader(fh) if row.get("id_live")}

    new_rows: list[list[str]] = []
    for rnd in rounds:
        if rnd not in rounds_present or rnd not in ranges:
            continue
        slug = round_slug(rnd)
        id_live = f"{prefix}_{slug}"
        if id_live in existing_ids:
            print(f"     [=] {id_live} déjà présent, ignoré.")
            continue
        start, end = ranges[rnd]
        new_row = [
            id_live, url, local_path, competition, category,
            f"pool_{pool_number}_{slug}", str(start), str(end),
            "low", "high", "pending",
        ]
        if write_type:
            new_row.append(competition_type)
        new_rows.append(new_row)

    if not new_rows:
        return

    _append_lives_rows(lives_path, new_rows)
    print(f"     [ok] {len(new_rows)} ligne(s) ajoutée(s) au lives -> {lives_path}")


def append_finals_lives(lives_path: Path, competition: str, year: str, sex: str,
                        matches_info: list[tuple[str, str, tuple[int, int]]],
                        video_name_override: str | None = None,
                        competition_type: CompetitionType = "SA") -> None:
    """Ajoute une ligne lives par match filmé des phases finales.

    ``matches_info`` = liste de ``(cle, url, (debut, fin))``. Chaque match peut
    avoir sa propre URL (tatamis / lives différents). Le fichier local est
    déduit de l'identifiant YouTube (une seule URL ⇒ un seul téléchargement).
    ``video_name_override`` ne s'applique que si tous les matchs partagent la
    même URL. Les ``id_live`` déjà présents sont ignorés.
    """
    prefix = f"{competition.lower()}_{year}_{sex.lower()}_final"
    category = SEX_TO_CATEGORY[sex]
    distinct_urls = {url for _, url, _ in matches_info}
    single_url = len(distinct_urls) == 1

    existing_ids: set[str] = set()
    file_exists = lives_path.exists()
    write_type = _lives_has_competition_type(lives_path)
    if file_exists:
        with lives_path.open("r", encoding="utf-8", newline="") as fh:
            existing_ids = {row["id_live"].strip()
                            for row in csv.DictReader(fh) if row.get("id_live")}

    new_rows: list[list[str]] = []
    for key, url, (start, end) in matches_info:
        id_live = f"{prefix}_{key}"
        if id_live in existing_ids:
            print(f"     [=] {id_live} déjà présent, ignoré.")
            continue
        if video_name_override and single_url:
            local_path = f"data/input/{video_name_override}"
        else:
            local_path = (f"data/input/{competition.lower()}_{year}_{sex.lower()}"
                          f"_final_{youtube_id(url)}.mp4")
        new_row = [
            id_live, url, local_path, competition, category,
            f"final_{key}", str(start), str(end),
            "low", "high", "pending",
        ]
        if write_type:
            new_row.append(competition_type)
        new_rows.append(new_row)

    if not new_rows:
        return

    _append_lives_rows(lives_path, new_rows)
    print(f"     [ok] {len(new_rows)} ligne(s) ajoutée(s) au lives -> {lives_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path,
                        default=Path("data/input/Database_K1_SA.csv"),
                        help="Base historique (défaut : data/input/Database_K1_SA.csv)")
    parser.add_argument("--competition", required=True,
                        help="Nom de la compétition, ex. SA_ACoruna")
    parser.add_argument(
        "--type-compet",
        choices=["SA", "K1"],
        default=None,
        help="Format SA ou K1 (déduit de Type_Compet / du nom si absent).",
    )
    parser.add_argument("--year", required=True, help="Année, ex. 2026")
    parser.add_argument("--sex", required=True, choices=["F", "M"],
                        help="Sexe : F (Female Kata) ou M (Male Kata)")
    parser.add_argument("--pool", type=int,
                        help="Numéro de la pool à générer (ordre d'apparition)")
    parser.add_argument("--finals", action="store_true",
                        help="Générer les phases finales du format SA/K1 au lieu "
                             "d'une pool.")
    parser.add_argument("--start-athlete", default=None,
                        help="Athlète du 1er match compté au tour initial (ignore "
                             "les matchs précédents non filmés)")
    parser.add_argument("--end-athlete", default=None,
                        help="Athlète du dernier match compté (arrête la génération après)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Chemin du CSV de sortie (défaut : "
                             "data/input/<compet>_<year>_<sex>_pool<N>.csv)")
    parser.add_argument("--list", action="store_true",
                        help="Lister les pools détectées puis quitter (pas d'écriture)")
    # --- Ajout automatique des lignes du fichier lives (lien vidéo + timecodes) ---
    lives_grp = parser.add_argument_group(
        "lives", "Ajout des lignes lives (lien vidéo + timecodes). Fournir --url + "
        "les timecodes des tours (format DEBUT-FIN, ex. 15:28-1:00:45).")
    lives_grp.add_argument("--url", default=None, help="Lien YouTube de la pool")
    lives_grp.add_argument("--t1", default=None, help="Plage T1, ex. 15:28-1:00:45")
    lives_grp.add_argument("--t2", default=None, help="Plage T2")
    lives_grp.add_argument("--t3", default=None, help="Plage T3")
    lives_grp.add_argument("--pw1", default=None, help="Plage PW1")
    lives_grp.add_argument(
        "--round",
        dest="round_specs",
        action="append",
        default=None,
        metavar="TOUR=PLAGE",
        help="Timecode générique, ex. --round Pool_1=15:28-1:00:45. Répétable.",
    )
    # Phases finales (--finals) : un match à la fois, chacun pouvant avoir sa
    # propre vidéo (tatamis différents). Répéter --match autant que de matchs.
    lives_grp.add_argument(
        "--match", action="append", default=None, metavar="CLE=PLAGE",
        help="(--finals) un match filmé : 'quart_1=15:28-16:05' ou, si la vidéo "
             "diffère de --url, 'quart_1=URL|15:28-16:05'. Répéter par match. "
             "Les clés SA/K1 exactes sont affichées par --finals --list.")
    lives_grp.add_argument("--lives", type=Path,
                           default=None,
                           help="Fichier lives à compléter (défaut : "
                                "data/input/<Competition>_<Year>_<Sex>_lives.csv)")
    lives_grp.add_argument("--video-name", default=None,
                           help="Nom du .mp4 local (défaut : "
                                "<compet>_<year>_<sex>_pool<N>.mp4)")
    args = parser.parse_args(argv)

    # Défaut du fichier lives : auto-dérivé de competition/year/sex.
    if args.lives is None:
        args.lives = Path(
            f"data/input/{args.competition}_{args.year}_{args.sex}_lives.csv"
        )

    if not args.database.exists():
        print(f"[erreur] Base introuvable : {args.database}", file=sys.stderr)
        return 1

    rows = load_rows(args.database, args.competition, args.year, args.sex)
    if not rows:
        print("[erreur] Aucune ligne pour ces filtres "
              f"(competition={args.competition}, year={args.year}, sex={args.sex}).",
              file=sys.stderr)
        return 1

    try:
        competition_type = detect_competition_type(
            rows, args.type_compet, args.competition
        )
    except ValueError as exc:
        print(f"[erreur] {exc}", file=sys.stderr)
        return 1
    competition_format: CompetitionFormat = get_competition_format(competition_type)
    pool_rounds = list(competition_format.pool_rounds)
    finals_rounds = list(competition_format.finals_rounds)

    # --- Mode phases finales -------------------------------------------------
    if args.finals:
        if args.pool is not None:
            print("[erreur] --finals et --pool sont mutuellement exclusifs.",
                  file=sys.stderr)
            return 1

        category = SEX_TO_CATEGORY[args.sex]
        matches = enumerate_finals_matches(
            rows, finals_rounds, competition_format.finals_labels
        )
        if not matches:
            print(f"[erreur] Aucun tour final ({'/'.join(finals_rounds)}) trouvé "
                  "dans la base pour ces filtres.", file=sys.stderr)
            return 1

        # --finals --list : afficher les matchs (clé + athlètes) puis quitter.
        if args.list:
            print(f"Finales {args.competition} {args.year} {args.sex} — "
                  f"{len(matches)} match(s) :")
            for m in matches:
                print(f"  {m.key:<9} | {m.red['Nom'].strip():<28} "
                      f"vs {m.blue['Nom'].strip():<28} -> {m.winner}")
            print("\nPour chaque match filmé : "
                  "--match \"<clé>=DEBUT-FIN\" (ou \"<clé>=URL|DEBUT-FIN\").")
            return 0

        valid_keys = {m.key for m in matches}
        if not args.match:
            print("[erreur] Aucun match fourni. Utilise --match \"<clé>=DEBUT-FIN\" "
                  "(répéter par match) ou --finals --list pour voir les clés.",
                  file=sys.stderr)
            return 1

        # Parse des specs --match.
        specs: dict[str, tuple[str | None, tuple[int, int]]] = {}
        try:
            for raw in args.match:
                key, url, rng = parse_match_spec(raw)
                if key not in valid_keys:
                    print(f"[erreur] Clé de match inconnue : {key!r}. "
                          "Voir --finals --list.", file=sys.stderr)
                    return 1
                specs[key] = (url, rng)
        except ValueError as exc:
            print(f"[erreur] {exc}", file=sys.stderr)
            return 1

        filmed_keys = set(specs)
        out_rows = finals_dataset_rows(
            matches,
            filmed_keys,
            args.competition,
            category,
            competition_type,
        )

        output = args.output or Path("data/input") / \
            f"{args.competition}_{args.year}_{args.sex}_finals.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(OUTPUT_HEADER)
            writer.writerows(out_rows)
        print(f"[ok] {len(out_rows)} match(s) de finales filmés écrits -> {output}")
        print(f"     {' '.join(r[14] for r in out_rows)}")

        # Lignes lives : une par match ayant une URL (inline ou --url global).
        matches_info: list[tuple[str, str, tuple[int, int]]] = []
        missing_url: list[str] = []
        for m in matches:
            if m.key not in specs:
                continue
            url, rng = specs[m.key]
            eff_url = url or args.url
            if eff_url:
                matches_info.append((m.key, eff_url, rng))
            else:
                missing_url.append(m.key)

        print()
        if missing_url:
            print(f"[!] Pas d'URL pour {', '.join(missing_url)} "
                  "(ni --url global ni URL inline) : lignes lives ignorées.",
                  file=sys.stderr)
        if matches_info:
            append_finals_lives(args.lives, args.competition, args.year, args.sex,
                                matches_info, args.video_name, competition_type)
        elif not missing_url:
            print("[i] Aucun timecode exploitable pour le lives.")
        return 0
    # ------------------------------------------------------------------------

    pools = split_into_pools(rows, pool_rounds)
    if not pools:
        print(f"[erreur] Aucune pool détectée (pas de tour {pool_rounds[0]}).",
              file=sys.stderr)
        return 1

    summarize(pools, pool_rounds)

    if args.list:
        return 0

    if args.pool is None:
        print("\n[erreur] Préciser --pool <N> pour générer un fichier "
              "(voir la liste ci-dessus).", file=sys.stderr)
        return 1
    if not 1 <= args.pool <= len(pools):
        print(f"\n[erreur] Pool {args.pool} hors limites (1..{len(pools)}).",
              file=sys.stderr)
        return 1

    category = SEX_TO_CATEGORY[args.sex]
    matches = pair_rows(pools[args.pool - 1], pool_rounds)
    out_rows = build_dataset(matches, args.competition, category, args.pool,
                             args.start_athlete, args.end_athlete,
                             pool_rounds, competition_type)

    if args.start_athlete and not out_rows:
        print(f"\n[erreur] start-athlete '{args.start_athlete}' introuvable dans la pool.",
              file=sys.stderr)
        return 1

    output = args.output or Path("data/input") / \
        f"{args.competition}_{args.year}_{args.sex}_pool{args.pool}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUTPUT_HEADER)
        writer.writerows(out_rows)

    per_round: dict[str, int] = {}
    for r in out_rows:
        per_round[r[2]] = per_round.get(r[2], 0) + 1
    detail = " ".join(f"{k}:{v}" for k, v in per_round.items())
    print(f"\n[ok] {len(out_rows)} match(s) écrits -> {output}")
    print(f"     {detail}")

    # Ajout optionnel des lignes du fichier lives (si --url fourni).
    if args.url:
        try:
            ranges: dict[str, tuple[int, int]] = {}
            for rnd, raw in (("T1", args.t1), ("T2", args.t2),
                             ("T3", args.t3), ("PW1", args.pw1)):
                if raw:
                    if rnd not in pool_rounds:
                        raise ValueError(
                            f"Le tour {rnd} n'appartient pas au format {competition_type}"
                        )
                    ranges[rnd] = parse_range(raw)
            for raw in args.round_specs or []:
                rnd, parsed_range = parse_round_spec(raw)
                if rnd not in pool_rounds:
                    raise ValueError(
                        f"Tour {rnd!r} inconnu pour {competition_type}; "
                        f"attendu : {', '.join(pool_rounds)}"
                    )
                ranges[rnd] = parsed_range
        except ValueError as exc:
            print(f"\n[erreur] {exc}", file=sys.stderr)
            return 1
        if not ranges:
            print("\n[erreur] --url fourni mais aucun timecode "
                  "(--round TOUR=DEBUT-FIN, ou options SA historiques).",
                  file=sys.stderr)
            return 1
        video_name = args.video_name or \
            f"{args.competition.lower()}_{args.year}_{args.sex.lower()}_pool{args.pool}.mp4"
        print()
        append_lives(args.lives, args.competition, args.year, args.sex, args.pool,
                     args.url, video_name, ranges, list(per_round.keys()),
                     pool_rounds, competition_type)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
