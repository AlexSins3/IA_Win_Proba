"""Interface Streamlit de validation des clips kata."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Chemins par défaut
DEFAULT_CLIPS_DIR = Path("data/clips")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_BACKUP_DIR = Path("data/backups")


def resolve_clip_path(clip: pd.Series) -> Path | None:
    """Retrouve un clip après migration entre l'ancien chemin et SA/K1."""

    raw_path = clip.get("clip_path")
    if pd.notna(raw_path):
        candidate = Path(str(raw_path))
        if candidate.exists():
            return candidate

    from kata_pipeline.utils.filenames import sanitize_filename

    try:
        order = int(clip["match_order"])
        filename = "_".join(
            [
                sanitize_filename(str(clip["competition"]), max_length=20),
                sanitize_filename(str(clip["category"]), max_length=15),
                sanitize_filename(str(clip["round"]), max_length=10),
                f"m{order:02d}",
                str(clip["color"]),
                sanitize_filename(str(clip["athlete"]), max_length=25),
                sanitize_filename(str(clip["kata"]), max_length=25),
            ]
        ) + ".mp4"
    except (KeyError, TypeError, ValueError):
        return None
    matches = list(DEFAULT_CLIPS_DIR.rglob(filename))
    return matches[0] if len(matches) == 1 else None


def find_clips_dataset() -> Path | None:
    """Trouver le dataset de clips le plus récent."""
    output_dir = DEFAULT_OUTPUT_DIR
    if not output_dir.exists():
        return None
    csv_files = sorted(
        output_dir.glob("*_clips.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return csv_files[0] if csv_files else None


def backup_dataset(path: Path) -> Path:
    """Créer un backup du dataset avant modification."""
    DEFAULT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DEFAULT_BACKUP_DIR / f"{path.stem}_backup_{timestamp}.csv"
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    """Application Streamlit de validation."""
    st.set_page_config(page_title="Kata Pipeline - Validation", layout="wide")
    st.title("Validation des clips kata")

    # Sélection du dataset
    dataset_path = find_clips_dataset()

    uploaded = st.sidebar.file_uploader("Charger un dataset de clips (CSV)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        dataset_source = "uploaded"
    elif dataset_path is not None:
        df = pd.read_csv(dataset_path)
        dataset_source = str(dataset_path)
    else:
        st.warning("Aucun dataset de clips trouvé. Lancez d'abord la pipeline.")
        return

    st.sidebar.info(f"Source: {dataset_source}")
    st.sidebar.metric("Total clips", len(df))
    st.sidebar.metric("Auto-validés", len(df[df["validation_status"] == "auto_validated"]))
    st.sidebar.metric("Needs review", len(df[df["needs_review"] == True]))  # noqa: E712

    # Filtres
    st.sidebar.subheader("Filtres")
    status_filter = st.sidebar.multiselect(
        "Statut",
        options=["pending", "auto_validated", "manually_validated", "needs_correction", "rejected"],
        default=["pending", "needs_correction"],
    )
    type_filter: list[str] = []
    if "competition_type" in df.columns:
        available_types = sorted(df["competition_type"].dropna().astype(str).unique())
        type_filter = st.sidebar.multiselect(
            "Circuit", options=available_types, default=available_types
        )

    if status_filter:
        filtered_df = df[df["validation_status"].isin(status_filter)]
    else:
        filtered_df = df
    if type_filter:
        filtered_df = filtered_df[filtered_df["competition_type"].isin(type_filter)]

    st.subheader(f"Clips à valider ({len(filtered_df)})")

    if filtered_df.empty:
        st.success("Tous les clips ont été traités !")
        return

    # Navigation
    if "clip_index" not in st.session_state:
        st.session_state.clip_index = 0

    col_prev, col_idx, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("< Précédent") and st.session_state.clip_index > 0:
            st.session_state.clip_index -= 1
    with col_next:
        if st.button("Suivant >") and st.session_state.clip_index < len(filtered_df) - 1:
            st.session_state.clip_index += 1
    with col_idx:
        st.write(f"Clip {st.session_state.clip_index + 1} / {len(filtered_df)}")

    # Clip courant
    clip_idx = min(st.session_state.clip_index, len(filtered_df) - 1)
    clip = filtered_df.iloc[clip_idx]

    # Affichage métadonnées
    col_video, col_meta = st.columns([2, 1])

    with col_meta:
        st.subheader("Métadonnées")
        st.write(f"**Athlète:** {clip['athlete']}")
        st.write(f"**Couleur:** {clip['color']}")
        st.write(f"**Kata:** {clip['kata']}")
        st.write(f"**Compétition:** {clip['competition']}")
        if pd.notna(clip.get("competition_type")):
            st.write(f"**Circuit:** {clip['competition_type']}")
        st.write(f"**Catégorie:** {clip['category']}")
        st.write(f"**Tour:** {clip['round']}")
        st.write(f"**Match:** {clip['match_order']}")
        st.write(f"**Adversaire:** {clip['opponent']}")
        if pd.notna(clip.get("score")):
            st.write(f"**Score:** {clip['score']}")
        st.write(f"**Issue:** {clip['issue']}")
        st.write(f"**Confiance:** {clip['confidence_score']:.2f}")
        st.write(f"**Durée:** {clip.get('duration', 0):.1f}s")

    with col_video:
        st.subheader("Vidéo")
        clip_path = resolve_clip_path(clip)
        if clip_path is not None:
            st.video(str(clip_path))
        else:
            st.info(
                f"Clip non encore généré.\n"
                f"Temps: {clip['start_time']:.1f}s -> {clip['end_time']:.1f}s"
            )

    # Actions de validation
    st.subheader("Validation")

    col_actions = st.columns(4)

    with col_actions[0]:
        if st.button("Valider", type="primary"):
            _update_status(df, clip, "manually_validated", dataset_path)
            st.success("Clip validé !")
            st.rerun()

    with col_actions[1]:
        if st.button("Rejeter"):
            _update_status(df, clip, "rejected", dataset_path)
            st.warning("Clip rejeté.")
            st.rerun()

    with col_actions[2]:
        if st.button("À corriger"):
            _update_status(df, clip, "needs_correction", dataset_path)
            st.info("Clip marqué à corriger.")
            st.rerun()

    with col_actions[3]:
        if st.button("Auto-valider"):
            _update_status(df, clip, "auto_validated", dataset_path)
            st.success("Clip auto-validé.")
            st.rerun()

    # Correction manuelle des timestamps
    st.subheader("Correction des timestamps")
    col_start, col_end = st.columns(2)
    with col_start:
        new_start = st.number_input(
            "Start time (s)", value=float(clip["start_time"]), step=0.5, format="%.1f"
        )
    with col_end:
        new_end = st.number_input(
            "End time (s)", value=float(clip["end_time"]), step=0.5, format="%.1f"
        )

    if st.button("Sauvegarder les timestamps"):
        _update_timestamps(df, clip, new_start, new_end, dataset_path)
        st.success("Timestamps mis à jour.")
        st.rerun()


def _update_status(
    df: pd.DataFrame, clip: pd.Series, new_status: str, dataset_path: Path | None
) -> None:
    """Mettre à jour le statut de validation d'un clip."""
    if dataset_path and dataset_path.exists():
        backup_dataset(dataset_path)

    mask = df["id_clip"] == clip["id_clip"]
    df.loc[mask, "validation_status"] = new_status
    df.loc[mask, "needs_review"] = new_status in ("needs_correction", "pending")

    if dataset_path:
        df.to_csv(dataset_path, index=False, encoding="utf-8")


def _update_timestamps(
    df: pd.DataFrame, clip: pd.Series, start: float, end: float, dataset_path: Path | None
) -> None:
    """Mettre à jour les timestamps d'un clip."""
    if dataset_path and dataset_path.exists():
        backup_dataset(dataset_path)

    mask = df["id_clip"] == clip["id_clip"]
    df.loc[mask, "start_time"] = start
    df.loc[mask, "end_time"] = end

    if dataset_path:
        df.to_csv(dataset_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
