"""Audio Intelligence and Transcription Engine using Faster-Whisper.
Supports:
- Original-language speech-to-text with timestamping
- English translation
- Language detection with confidence
- Association with source CCTV recording or standalone audio files
- Optimized on CPU int8 to leave 4 GB VRAM exclusively for YOLOv8
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
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

    if progress_callback:
        progress_callback("Generating English translations...")

    trans_segments, _ = whisper.transcribe(
        str(path),
        task="translate",
        beam_size=5,
        vad_filter=True
    )
    trans_segments = list(trans_segments)

    detected_lang = getattr(info, "language", "unknown")
    lang_prob = getattr(info, "language_probability", 0.0)

    observations: List[Dict[str, Any]] = []

    for idx, seg in enumerate(orig_segments):
        start_sec = float(seg.start)
        end_sec = float(seg.end)
        orig_text = seg.text.strip()

        # Match translation
        eng_text = ""
        if idx < len(trans_segments):
            eng_text = trans_segments[idx].text.strip()
        elif trans_segments:
            eng_text = trans_segments[-1].text.strip()

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

