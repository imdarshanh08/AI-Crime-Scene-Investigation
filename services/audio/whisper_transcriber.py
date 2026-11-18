"""Audio Intelligence and Transcription Engine using Faster-Whisper.
Supports:
- Original-language speech-to-text with timestamping
- English translation
- Language detection with confidence
- Association with source CCTV recording or standalone audio files
- Optimized on CPU int8 to leave 4 GB VRAM exclusively for YOLOv8
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from faster_whisper import WhisperModel

from core.config import ROOT  # noqa: E402
OUTPUT_DIR = ROOT / "outputs"
AUDIO_EVIDENCE_DIR = OUTPUT_DIR / "audio_evidence"
AUDIO_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

_model_cache: Dict[str, WhisperModel] = {}


def load_whisper_engine(model_size: str = "base", device: str = "cpu") -> WhisperModel:
    """Load Faster-Whisper model cached in memory."""
    cache_key = f"{model_size}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_cache[cache_key] = model
    return model


def format_seconds(seconds: float) -> str:
    if seconds is None:
        return "00:00.00"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def _best_overlap(start: float, end: float, segments: List[Any]) -> str:
    """Return the text of the segment that overlaps [start, end] the most.

    Transcription and translation are two separate Whisper passes, so their segment
    boundaries differ; pairing them by list index misaligned translations.
    """
    best_text, best_overlap = "", 0.0
    for seg in segments:
        overlap = min(end, float(seg.end)) - max(start, float(seg.start))
        if overlap > best_overlap:
            best_overlap, best_text = overlap, seg.text.strip()
    if not best_text and segments:
        mid = (start + end) / 2
        nearest = min(segments, key=lambda s: abs((float(s.start) + float(s.end)) / 2 - mid))
        best_text = nearest.text.strip()
    return best_text


def transcribe_and_translate_audio(
    audio_file_path: str,
    source_id: str,
    source_name: str,
    camera_id: str = "AUDIO-01",
    model_size: str = "base",
    device: str = "cpu",
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[Dict[str, Any]]:
    """Generate timestamped original transcripts and English translations."""
    path = Path(audio_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    if progress_callback:
        progress_callback(f"Loading Whisper ({model_size}) engine on {device}...")

    whisper = load_whisper_engine(model_size, device)

    if progress_callback:
        progress_callback("Transcribing original speech segments...")

    orig_segments, info = whisper.transcribe(
        str(path),
        task="transcribe",
        beam_size=5,
        vad_filter=True
    )
    orig_segments = list(orig_segments)

    detected_lang = getattr(info, "language", None) or "unknown"
    lang_prob = getattr(info, "language_probability", 0.0)

    # English speech needs no second (slow) translation pass.
    trans_segments: List[Any] = []
    if detected_lang != "en" and orig_segments:
        if progress_callback:
            progress_callback("Generating English translations...")
        trans_segments, _ = whisper.transcribe(
            str(path),
            task="translate",
            beam_size=5,
            vad_filter=True
        )
        trans_segments = list(trans_segments)

    observations: List[Dict[str, Any]] = []

    for idx, seg in enumerate(orig_segments):
        start_sec = float(seg.start)
        end_sec = float(seg.end)
        orig_text = seg.text.strip()

        # Match translation by time overlap
        if detected_lang == "en":
            eng_text = orig_text
        else:
            eng_text = _best_overlap(start_sec, end_sec, trans_segments)

        obs_id = hashlib.sha1(
            f"{source_id}_{idx}_{start_sec:.2f}".encode("utf-8")
        ).hexdigest()[:12]

        row = {
            "Observation ID": obs_id,
            "Modality": "Audio",
            "Source": source_name,
            "Source ID": source_id,
            "Camera ID": camera_id,
            "Timestamp": f"{format_seconds(start_sec)} - {format_seconds(end_sec)}",
            "Timestamp (seconds)": round(start_sec, 2),
            "Segment End (seconds)": round(end_sec, 2),
            "Object": "Speech",
            "Raw Track ID": "Not applicable",
            "Track ID": "Not applicable",
            "Confidence": round(float(lang_prob), 3) if lang_prob else None,
            "Bounding Box": "",
            "Observation": f"Speech detected [{detected_lang.upper()}]: \"{orig_text}\"",
            "Original Transcript": orig_text,
            "English Translation": eng_text,
            "Detected Language": detected_lang,
            "Language Probability": round(float(lang_prob), 3) if lang_prob else None,
            "Evidence Path": str(path),
            "Review Status": "Pending",
            "Review Notes": "",
            "Reviewed At": ""
        }
        observations.append(row)

    if progress_callback:
        progress_callback(f"Transcribed {len(observations)} speech segments.")

    return observations

