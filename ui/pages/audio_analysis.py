"""Audio analysis: Faster-Whisper transcription with original-language text and English translation."""

import streamlit as st

from core.config import WHISPER_DEFAULT_MODEL, WHISPER_DEVICE
from services.audio import whisper_transcriber
from services.media import media_processor
from storage import db_manager
from ui.components import page_header
from ui.state import clean_text, sync_session_from_database


def _transcribe(source, audio_path, label):
    model = st.session_state.get("select_whisper_size", WHISPER_DEFAULT_MODEL)
    with st.spinner(f"Transcribing {label} with Whisper '{model}' on CPU (the model downloads on first use)..."):
        try:
            rows = whisper_transcriber.transcribe_and_translate_audio(
                audio_file_path=audio_path, source_id=source["source_id"], source_name=label,
                camera_id=source["camera_id"], model_size=model, device=WHISPER_DEVICE)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return
    db_manager.save_observations_batch(rows)
    if source["modality"] == "Audio":
        db_manager.set_processing_status(source["source_id"], "Processed")
    db_manager.log_investigator_action("Ran speech transcription", "Audio", source["source_id"],
                                       f"{len(rows)} segment(s), model {model}")
    sync_session_from_database()
    if rows:
        st.success(f"{len(rows)} speech segment(s) transcribed.")
    else:
        st.info("No speech was detected.")


def render():
    page_header(
        "Audio analysis",
        "Transcribe speech in its original language with an English translation and timestamps. Audio taken from "
        "a CCTV file shares that recording's clock, so speech can be aligned with what the camera saw.",
        eyebrow="Analysis",
    )
    cctv = db_manager.get_all_sources("CCTV")
    audio = db_manager.get_all_sources("Audio")
    if not cctv and not audio:
        st.info("No CCTV or audio evidence registered yet.")
        st.page_link("ui/pages/evidence_upload.py", label="Upload evidence", icon=":material/upload:")
        return

    with st.container(border=True):
        options = {f"{s['camera_id']} · {s['source_name']} (audio track of video)": s for s in cctv}
        options.update({f"{s['camera_id']} · {s['source_name']}": s for s in audio})
        c1, c2 = st.columns([3, 1])
        choice = c1.selectbox("Recording", list(options), key="select_audio_source")
        c2.selectbox("Whisper model", ["tiny", "base", "small"], index=1, key="select_whisper_size",
                     help="Larger models are more accurate but slower. All run on CPU.")
        src = options[choice]
        if st.button("Transcribe", type="primary", key="btn_run_whisper"):
            if src["modality"] == "CCTV":
                with st.spinner("Extracting the audio track..."):
                    wav = media_processor.extract_audio_from_cctv(src["file_path"])
                if not wav:
                    st.warning(f"{src['source_name']} has no audio track.")
                else:
                    _transcribe(src, wav, f"Audio_{src['source_name']}")
            else:
                _transcribe(src, src["file_path"], src["source_name"])

    obs = db_manager.get_all_observations_df(modality="Audio")
    if obs.empty:
        st.caption("No transcripts yet.")
        return
    st.subheader("Transcripts")
    sources = sorted(obs["Camera ID"].dropna().unique())
    pick = st.multiselect("Recordings", sources, default=sources, key="audio_filter_sources")
    view = obs[obs["Camera ID"].isin(pick)].copy()
    view["Language"] = view["Detected Language"].map(clean_text).str.upper()
    st.dataframe(view[["Camera ID", "Timestamp", "Language", "Original Transcript", "English Translation",
                       "Review Status", "Observation ID"]],
                 hide_index=True, width="stretch",
                 column_config={"Original Transcript": st.column_config.TextColumn(width="large"),
                                "English Translation": st.column_config.TextColumn(width="large")})
    st.download_button("Download transcripts (CSV)", view.to_csv(index=False).encode("utf-8"), "transcripts.csv",
                       "text/csv", key="dl_transcripts")


if __name__ == "__main__":
    render()
