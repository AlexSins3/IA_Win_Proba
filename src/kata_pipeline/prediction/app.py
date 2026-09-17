"""Interface Streamlit : prédire le vainqueur à partir de deux vidéos."""

from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from kata_pipeline.gnn.config import load_gnn_config
from kata_pipeline.gnn.predict import predict_sequences
from kata_pipeline.gnn.skeletons import get_schema
from kata_pipeline.prediction.pipeline import (
    extract_pose_sequence,
    normalize_video,
    pose_summary,
    probable_match_score,
    render_web_motion_overlay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "outcome.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "runs" / "outcome_20260915_100930" / "config.yaml"


def _asset_path(environment_name: str, default: Path) -> Path:
    value = os.environ.get(environment_name)
    return Path(value).expanduser().resolve() if value else default


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("o", "Ko", "Mo", "Go"):
        if value < 1024 or unit == "Go":
            return f"{value:.0f} {unit}" if unit == "o" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} Go"


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if 1 < len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".video"


def _write_upload(upload: Any, path: Path) -> None:
    with path.open("wb") as destination:
        destination.write(upload.getbuffer())


def _step(container: Any, message: str, done: bool = False) -> None:
    icon = "✅" if done else "⏳"
    container.markdown(f"{icon} {message}")


def _run_prediction(upload_a: Any, upload_b: Any) -> dict[str, Any]:
    model_path = _asset_path("KATA_PREDICTION_MODEL", DEFAULT_MODEL_PATH)
    config_path = _asset_path("KATA_PREDICTION_CONFIG", DEFAULT_CONFIG_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint du modèle introuvable : {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration du modèle introuvable : {config_path}")

    config = load_gnn_config(config_path)
    schema = get_schema(config.skeleton.schema_name)

    st.markdown("### Analyse en cours")
    preview_title = st.empty()
    preview_a_col, preview_b_col = st.columns(2, gap="large")
    preview_a = preview_a_col.empty()
    preview_b = preview_b_col.empty()
    progress = st.progress(0, text="Initialisation du traitement…")

    with tempfile.TemporaryDirectory(prefix="kata_match_") as temporary_directory:
        workdir = Path(temporary_directory)
        source_a = workdir / f"source_a{_safe_suffix(upload_a.name)}"
        source_b = workdir / f"source_b{_safe_suffix(upload_b.name)}"
        normalized_a = workdir / "athlete_a.mp4"
        normalized_b = workdir / "athlete_b.mp4"
        overlay_a = workdir / "athlete_a_skeleton.mp4"
        overlay_b = workdir / "athlete_b_skeleton.mp4"

        with st.status("Prédiction en cours…", expanded=True) as status:
            save_item = st.empty()
            conversion_item = st.empty()
            pose_a_item = st.empty()
            overlay_a_item = st.empty()
            pose_b_item = st.empty()
            overlay_b_item = st.empty()
            inference_item = st.empty()
            cleanup_item = st.empty()

            try:
                _step(save_item, "Chargement sécurisé des deux fichiers")
                _write_upload(upload_a, source_a)
                _write_upload(upload_b, source_b)
                _step(save_item, "Deux fichiers reçus", done=True)
                progress.progress(8, text="Fichiers reçus")

                _step(conversion_item, "Conversion des vidéos en MP4/H.264")
                metadata_a = normalize_video(source_a, normalized_a)
                progress.progress(20, text="Vidéo 1 normalisée")
                metadata_b = normalize_video(source_b, normalized_b)
                _step(conversion_item, "Vidéos normalisées en MP4/H.264", done=True)
                progress.progress(32, text="Conversion terminée")

                _step(pose_a_item, "Extraction des poses — vidéo 1")
                sequence_a = extract_pose_sequence(normalized_a, config, schema)
                summary_a = pose_summary(sequence_a, config.preprocess.confidence_threshold)
                _step(pose_a_item, "Poses extraites — vidéo 1", done=True)
                progress.progress(45, text="Poses de la vidéo 1 extraites")

                _step(overlay_a_item, "Génération du squelette animé — vidéo 1")
                render_web_motion_overlay(
                    normalized_a,
                    overlay_a,
                    sequence_a,
                    schema,
                    config.preprocess.confidence_threshold,
                )
                overlay_a_bytes = overlay_a.read_bytes()
                preview_title.markdown("#### Squelettes disponibles pendant l'analyse")
                preview_a_col.markdown("**Vidéo 1**")
                preview_a.video(overlay_a_bytes, format="video/mp4", autoplay=True, muted=True)
                _step(overlay_a_item, "Squelette prêt — vidéo 1", done=True)
                progress.progress(57, text="Squelette de la vidéo 1 disponible")

                _step(pose_b_item, "Extraction des poses — vidéo 2")
                sequence_b = extract_pose_sequence(normalized_b, config, schema)
                summary_b = pose_summary(sequence_b, config.preprocess.confidence_threshold)
                _step(pose_b_item, "Poses extraites — vidéo 2", done=True)
                progress.progress(70, text="Poses de la vidéo 2 extraites")

                _step(overlay_b_item, "Génération du squelette animé — vidéo 2")
                render_web_motion_overlay(
                    normalized_b,
                    overlay_b,
                    sequence_b,
                    schema,
                    config.preprocess.confidence_threshold,
                )
                overlay_b_bytes = overlay_b.read_bytes()
                preview_b_col.markdown("**Vidéo 2**")
                preview_b.video(overlay_b_bytes, format="video/mp4", autoplay=True, muted=True)
                _step(overlay_b_item, "Squelette prêt — vidéo 2", done=True)
                progress.progress(82, text="Deux squelettes disponibles")

                _step(inference_item, "Prétraitement, fenêtrage et inférence du GNN")
                prediction = predict_sequences(
                    sequence_a,
                    sequence_b,
                    model_path,
                    config,
                    debug=True,
                )
                _step(inference_item, "Prédiction du modèle terminée", done=True)
                progress.progress(96, text="Calcul des résultats")

                _step(cleanup_item, "Suppression des vidéos temporaires")
                result = {
                    "prediction": prediction,
                    "videos": {
                        "a": {
                            "original_name": upload_a.name,
                            "original_size_bytes": int(upload_a.size),
                            "normalized": metadata_a.to_dict(),
                            "pose": summary_a,
                        },
                        "b": {
                            "original_name": upload_b.name,
                            "original_size_bytes": int(upload_b.size),
                            "normalized": metadata_b.to_dict(),
                            "pose": summary_b,
                        },
                    },
                    "model": {
                        "schema": schema.name,
                        "pose_fps": config.pose.target_fps,
                        "checkpoint": model_path.name,
                    },
                    "overlays": {"a": overlay_a_bytes, "b": overlay_b_bytes},
                }
                _step(cleanup_item, "Vidéos sources et intermédiaires supprimées", done=True)
                progress.progress(100, text="Prédiction terminée")
                status.update(label="Analyse terminée", state="complete", expanded=False)
                return result
            except Exception:
                status.update(label="La prédiction a échoué", state="error", expanded=True)
                raise


def _json_payload(bundle: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in bundle.items() if key != "overlays"}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _render_probability(label: str, probability: float, color: str) -> None:
    percent = probability * 100
    st.markdown(
        f"""
        <div class="probability-row">
          <span>{html.escape(label)}</span><strong style="color:{color}">{percent:.1f} %</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(probability))


def _render_video_details(label: str, data: dict[str, Any], accent: str) -> None:
    pose = data["pose"]
    normalized = data["normalized"]
    st.markdown(f"#### {label}")
    st.caption(html.escape(data["original_name"]))
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Durée analysée", f"{pose['duration_seconds']:.1f} s")
    metric_b.metric("Frames de pose", pose["pose_frames"])
    st.markdown(
        f"<span style='color:{accent};font-weight:700'>Détection du corps : "
        f"{pose['frame_detection_rate'] * 100:.1f} %</span>",
        unsafe_allow_html=True,
    )
    st.progress(float(pose["frame_detection_rate"]))
    st.caption(
        f"{normalized['width']}×{normalized['height']} · {normalized['fps']:.1f} fps · "
        f"{_format_size(data['original_size_bytes'])}"
    )


def _render_results(bundle: dict[str, Any]) -> None:
    prediction = bundle["prediction"]
    winner = prediction["predicted_winner"]
    winner_number = "1" if winner == "A" else "2"
    winner_color = "#ef4444" if winner == "A" else "#3b82f6"
    confidence = float(prediction["confidence"])
    winner_score, loser_score = probable_match_score(confidence)

    st.markdown(
        f"""
        <section class="winner-card">
          <div class="winner-kicker">PRÉDICTION FINALE</div>
          <div class="winner-name">Vidéo {winner_number} gagnante</div>
          <div class="winner-confidence" style="color:{winner_color}">
            Écart entre les prestations : {confidence * 100:.1f} %
          </div>
          <div class="match-score">
            Score le plus probable : <strong>{winner_score}–{loser_score}</strong>
            pour la vidéo {winner_number}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Cet écart correspond à la différence entre les probabilités de victoire "
        "attribuées aux deux prestations."
    )

    probability_a, probability_b = st.columns(2, gap="large")
    with probability_a:
        _render_probability("Vidéo 1", prediction["probability_a_wins"], "#ef4444")
        st.metric("Score technique latent", f"{prediction['athlete_a']['quality_score']:.4f}")
    with probability_b:
        _render_probability("Vidéo 2", prediction["probability_b_wins"], "#3b82f6")
        st.metric("Score technique latent", f"{prediction['athlete_b']['quality_score']:.4f}")
    st.caption(
        "Le score technique latent est la valeur interne utilisée par le modèle pour "
        "comparer la qualité des deux prestations : plus il est élevé, plus la prestation "
        "est favorisée. Ce n'est ni une note sur 10 ou 100, ni un score officiel des juges."
    )

    st.markdown("### Vidéos avec squelette motion")
    skeleton_a, skeleton_b = st.columns(2, gap="large")
    with skeleton_a:
        st.markdown("**Vidéo 1**")
        st.video(bundle["overlays"]["a"], format="video/mp4")
        st.download_button(
            "Télécharger le squelette 1",
            bundle["overlays"]["a"],
            file_name="video_1_squelette_motion.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    with skeleton_b:
        st.markdown("**Vidéo 2**")
        st.video(bundle["overlays"]["b"], format="video/mp4")
        st.download_button(
            "Télécharger le squelette 2",
            bundle["overlays"]["b"],
            file_name="video_2_squelette_motion.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    st.markdown("### Qualité de l'analyse")
    details_a, details_b = st.columns(2, gap="large")
    with details_a:
        _render_video_details("Vidéo 1", bundle["videos"]["a"], "#ef4444")
    with details_b:
        _render_video_details("Vidéo 2", bundle["videos"]["b"], "#3b82f6")

    debug = prediction.get("debug", {})
    validation_accuracy = debug.get("validation_accuracy")
    with st.expander("Résultats techniques et traçabilité"):
        tech_a, tech_b, tech_c = st.columns(3)
        tech_a.metric("Fenêtres vidéo 1", debug.get("num_windows_a", "—"))
        tech_b.metric("Fenêtres vidéo 2", debug.get("num_windows_b", "—"))
        tech_c.metric("Device", str(debug.get("device", "—")).upper())
        if validation_accuracy is not None:
            st.info(
                f"Accuracy de validation du checkpoint : {validation_accuracy * 100:.1f} % "
                f"(ROC AUC : {debug.get('validation_roc_auc', 0.0):.3f})."
            )
        st.json({key: value for key, value in bundle.items() if key != "overlays"})
        st.download_button(
            "Télécharger les résultats JSON",
            _json_payload(bundle),
            file_name="prediction_match.json",
            mime="application/json",
        )

    st.warning(
        "Cette estimation est une aide expérimentale issue d'un modèle statistique ; "
        "elle ne remplace pas la décision des juges."
    )
    reset_left, reset_center, reset_right = st.columns([1, 1.2, 1])
    with reset_center:
        if st.button("↻ Prédire un autre match", type="primary", use_container_width=True):
            st.session_state.pop("prediction_bundle", None)
            st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1
            st.rerun()


def _render_uploads() -> None:
    generation = st.session_state.get("upload_generation", 0)
    col_a, col_vs, col_b = st.columns([1, 0.18, 1], gap="medium", vertical_alignment="center")
    with col_a:
        st.markdown('<div class="side-label red">VIDÉO 1</div>', unsafe_allow_html=True)
        upload_a = st.file_uploader(
            "Déposer ou parcourir le fichier de la vidéo 1",
            type=None,
            accept_multiple_files=False,
            key=f"video_a_{generation}",
            help="Tout format vidéo décodable par FFmpeg est accepté.",
        )
        if upload_a is not None:
            st.success(f"✓ {upload_a.name} · {_format_size(upload_a.size)}")
    with col_vs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="side-label blue">VIDÉO 2</div>', unsafe_allow_html=True)
        upload_b = st.file_uploader(
            "Déposer ou parcourir le fichier de la vidéo 2",
            type=None,
            accept_multiple_files=False,
            key=f"video_b_{generation}",
            help="Tout format vidéo décodable par FFmpeg est accepté.",
        )
        if upload_b is not None:
            st.success(f"✓ {upload_b.name} · {_format_size(upload_b.size)}")

    ready = upload_a is not None and upload_b is not None
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    button_left, button_center, button_right = st.columns([1, 1.25, 1])
    with button_center:
        predict_clicked = st.button(
            "Prédire le gagnant",
            type="primary",
            disabled=not ready,
            use_container_width=True,
        )
    if not ready:
        st.caption("Ajoutez les deux vidéos pour activer la prédiction.")
    st.markdown(
        '<div class="privacy-note">🔒 Les fichiers sources sont traités dans un dossier '
        "temporaire et supprimés dès la fin de l'analyse.</div>",
        unsafe_allow_html=True,
    )

    if predict_clicked:
        try:
            bundle = _run_prediction(upload_a, upload_b)
        except Exception as exc:  # noqa: BLE001 - frontière UI, message utilisateur requis
            st.error(f"Impossible de terminer la prédiction : {exc}")
            return
        st.session_state["prediction_bundle"] = bundle
        # Un rerun remplace l'écran d'upload par les résultats et libère les
        # buffers des vidéos sources conservés par les widgets Streamlit.
        st.rerun()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%); }
          .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
          h1, h2, h3 { letter-spacing: -0.025em; }
          .app-kicker { color:#64748b; font-size:.75rem; letter-spacing:.18em;
                        font-weight:800; text-align:center; }
          .app-title { color:#0f172a; font-size:clamp(2rem,5vw,3.7rem); line-height:1.02;
                       font-weight:850; text-align:center; margin:.35rem 0 .65rem; }
          .app-subtitle { color:#64748b; max-width:690px; margin:0 auto 2.2rem;
                          text-align:center; font-size:1.05rem; }
          .side-label { font-size:.78rem; font-weight:900; letter-spacing:.14em;
                        margin-bottom:.4rem; }
          .side-label.red { color:#dc2626; } .side-label.blue { color:#2563eb; }
          .vs-badge { width:58px; height:58px; border-radius:50%; background:#0f172a;
                      color:white; display:flex; align-items:center; justify-content:center;
                      margin:0 auto; font-size:1rem; font-weight:900;
                      box-shadow:0 10px 30px rgba(15,23,42,.2); }
          [data-testid="stFileUploaderDropzone"] { min-height:180px; border:1.5px dashed #94a3b8;
                      border-radius:18px; background:rgba(255,255,255,.72); }
          .privacy-note { color:#64748b; font-size:.85rem; text-align:center; margin-top:1rem; }
          .winner-card { background:#0f172a; color:white; padding:2rem; border-radius:24px;
                         text-align:center; margin:.7rem 0 1rem;
                         box-shadow:0 24px 55px rgba(15,23,42,.18); }
          .winner-kicker { font-size:.7rem; letter-spacing:.2em; color:#94a3b8; font-weight:800; }
          .winner-name { font-size:clamp(1.8rem,4vw,3rem); font-weight:850; margin:.25rem 0; }
          .winner-confidence { font-size:1rem; font-weight:750; }
          .match-score { width:fit-content; margin:1rem auto 0; padding:.7rem 1.15rem;
                         border:1px solid rgba(255,255,255,.18); border-radius:999px;
                         background:rgba(255,255,255,.08); font-size:1.05rem; }
          .match-score strong { color:white; font-size:1.25rem; }
          .probability-row { display:flex; justify-content:space-between; align-items:center;
                             font-size:1.15rem; margin-top:.4rem; }
          div[data-testid="stMetric"] { background:rgba(255,255,255,.72); border:1px solid #e2e8f0;
                                        border-radius:14px; padding:1rem; }
          div.stButton > button[kind="primary"] { min-height:3rem; border-radius:999px;
                                                   font-weight:800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Kata Match Predictor",
        page_icon="🥋",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()
    st.markdown('<div class="app-kicker">KATA · MOTION INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">Qui remportera le match ?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Déposez deux prestations. Le moteur normalise les vidéos, '
        "extrait les mouvements et compare les séquences avec le modèle GNN.</div>",
        unsafe_allow_html=True,
    )

    bundle = st.session_state.get("prediction_bundle")
    if bundle is None:
        _render_uploads()
    else:
        _render_results(bundle)


if __name__ == "__main__":
    main()
