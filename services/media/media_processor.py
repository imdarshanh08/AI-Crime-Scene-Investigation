"""Media processing utilities for CCTV video and audio intelligence.
Optimized for Windows 11 with 4 GB RTX 3050 VRAM and 16 GB system RAM.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.config import ROOT  # noqa: E402
OUTPUT_DIR = ROOT / "outputs"
EVIDENCE_FRAMES_DIR = OUTPUT_DIR / "evidence_frames"
ANNOTATED_VIDEOS_DIR = OUTPUT_DIR / "annotated_videos"
AUDIO_EVIDENCE_DIR = OUTPUT_DIR / "audio_evidence"

for d in [OUTPUT_DIR, EVIDENCE_FRAMES_DIR, ANNOTATED_VIDEOS_DIR, AUDIO_EVIDENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Sanitize filename to prevent Windows path issues."""
    base = Path(name).name
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return clean


def hash_file_stream(file_path: Path, chunk_size: int = 65536) -> Tuple[str, str]:
    """Compute (short_id, full_sha256) streaming in chunks without high memory consumption."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    full_hex = hasher.hexdigest()
    return full_hex[:12], full_hex


def save_uploaded_stream(uploaded_file, target_dir: Path) -> Dict[str, Any]:
    """Stream save uploaded file to target directory while calculating SHA-256."""
    target_dir.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0

    # Ensure pointer is at beginning
    uploaded_file.seek(0)
    while chunk := uploaded_file.read(65536):
        hasher.update(chunk)
        size += len(chunk)

    full_sha256 = hasher.hexdigest()
    short_id = full_sha256[:12]
    clean_name = safe_filename(uploaded_file.name)
    target_path = target_dir / f"{short_id}_{clean_name}"

    # Write file to disk if not already existing
    if not target_path.exists():
        uploaded_file.seek(0)
        with open(target_path, "wb") as f:
            while chunk := uploaded_file.read(65536):
                f.write(chunk)

    uploaded_file.seek(0)

    return {
        "id": short_id,
        "name": uploaded_file.name,
        "path": str(target_path),
        "size_bytes": size,
        "sha256": full_sha256
    }


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Inspect video file using OpenCV and extract metadata and thumbnail preview."""
    path = Path(video_path)
    meta = {
        "is_valid": False,
        "fps": 25.0,
        "total_frames": 0,
        "duration_seconds": 0.0,
        "width": 0,
        "height": 0,
        "thumbnail_path": "",
        "error": ""
    }

    if not path.is_file():
        meta["error"] = "File not found"
        return meta

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        meta["error"] = "Failed to open video container with OpenCV"
        return meta

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or np.isnan(fps) or fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

        meta["is_valid"] = True
        meta["fps"] = round(fps, 2)
        meta["total_frames"] = total_frames
        meta["duration_seconds"] = round(duration, 2)
        meta["width"] = width
        meta["height"] = height

        # First frame as thumbnail. Reuse it if already extracted: this function runs
        # on every Streamlit rerun, so re-decoding and rewriting the JPEG each time was wasteful.
        thumb_path = EVIDENCE_FRAMES_DIR / f"thumb_{path.stem}.jpg"
        if thumb_path.is_file():
            meta["thumbnail_path"] = str(thumb_path)
        else:
            success, frame = cap.read()
            if success and frame is not None:
                cv2.imwrite(str(thumb_path), frame)
                meta["thumbnail_path"] = str(thumb_path)
    except Exception as e:
        meta["error"] = str(e)
    finally:
        cap.release()

    return meta


def extract_audio_from_cctv(video_path: str, output_wav_path: Optional[str] = None) -> Optional[str]:
    """Extract audio track from video container using PyAV and convert to 16kHz mono WAV.
    Returns path to WAV file, or None if no audio stream exists.
    """
    try:
        import av
    except ImportError:
        return None

    video_p = Path(video_path)
    if not video_p.is_file():
        return None

    if output_wav_path is None:
        out_p = AUDIO_EVIDENCE_DIR / f"{video_p.stem}_extracted.wav"
    else:
        out_p = Path(output_wav_path)

    out_p.parent.mkdir(parents=True, exist_ok=True)

    container = None
    output_container = None
    try:
        container = av.open(str(video_p))
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            return None

        # Transcode audio to 16kHz mono PCM WAV for Faster-Whisper
        output_container = av.open(str(out_p), mode="w", format="wav")
        out_stream = output_container.add_stream("pcm_s16le", rate=16000, layout="mono")

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        for frame in container.decode(audio_streams[0]):
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for r_frame in resampled_frames:
                    for packet in out_stream.encode(r_frame):
                        output_container.mux(packet)

        # Flush encoder
        for packet in out_stream.encode(None):
            output_container.mux(packet)

        output_container.close()
        output_container = None

        if out_p.exists() and out_p.stat().st_size > 44:  # More than just WAV header
            return str(out_p)
        return None
    except Exception as e:
        print(f"Audio extraction notice for {video_path}: {e}")
        return None
    finally:
        for c in (output_container, container):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass


class WebVideoWriter:
    """Writes 100% web-compatible H.264 (AVC1, yuv420p) MP4 video using PyAV (libx264).
    Guarantees immediate playback in Google Chrome, Edge, Firefox, Safari, and Streamlit st.video().
    Falls back gracefully to cv2.VideoWriter if PyAV fails.
    """
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = str(output_path)
        self.fps = int(round(fps)) if fps > 0 else 25
        self.width = int(width)
        self.height = int(height)
        self.use_pyav = True
        self.container = None
        self.stream = None
        self.cv2_writer = None

        try:
            import av
            self.container = av.open(self.output_path, mode="w", format="mp4")
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.width = self.width
            self.stream.height = self.height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"preset": "ultrafast", "crf": "23"}
        except Exception as e:
            print(f"PyAV writer notice: {e}, falling back to cv2.VideoWriter")
            self.use_pyav = False
            if self.container:
                try:
                    self.container.close()
                except Exception:
                    pass
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv2_writer = cv2.VideoWriter(self.output_path, fourcc, float(self.fps), (self.width, self.height))

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.use_pyav and self.container and self.stream:
            try:
                import av
                frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                for packet in self.stream.encode(frame):
                    self.container.mux(packet)
            except Exception:
                pass
        elif self.cv2_writer:
            self.cv2_writer.write(frame_bgr)

    def release(self) -> None:
        if self.use_pyav and self.container and self.stream:
            try:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
            except Exception:
                pass
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
            self.stream = None
        elif self.cv2_writer:
            self.cv2_writer.release()
            self.cv2_writer = None


def convert_to_web_h264(input_path: str, output_path: Optional[str] = None) -> str:
    """Converts any video file (e.g. OpenCV mp4v/mpeg4) into a web-compatible H.264 (AVC1, yuv420p) MP4.
    If the video is already H.264 yuv420p, returns input_path immediately without re-encoding.
    """
    in_p = Path(input_path)
    if not in_p.is_file():
        return input_path

    # Check if already h264 yuv420p
    try:
        import av
        in_c = av.open(str(in_p))
        if in_c.streams.video:
            v_st = in_c.streams.video[0]
            if v_st.codec_context.name == "h264" and v_st.codec_context.pix_fmt == "yuv420p":
                in_c.close()
                return str(in_p)
        in_c.close()
    except Exception:
        pass

    if output_path is None:
        out_p = in_p.parent / f"{in_p.stem}_web.mp4"
    else:
        out_p = Path(output_path)

    # Reuse an earlier conversion instead of re-encoding on every page rerun.
    if out_p.is_file() and out_p.stat().st_size > 1000 and out_p.stat().st_mtime >= in_p.stat().st_mtime:
        return str(out_p)

    try:
        import av
        in_container = av.open(str(in_p))
        if not in_container.streams.video:
            in_container.close()
            return str(in_p)

        in_video = in_container.streams.video[0]
        fps = float(in_video.average_rate) if in_video.average_rate else 25.0
        fps = int(round(fps)) if fps > 0 else 25
        width = in_video.width
        height = in_video.height

        out_container = av.open(str(out_p), mode="w", format="mp4")
        out_stream = out_container.add_stream("libx264", rate=fps)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"preset": "ultrafast", "crf": "23"}

        for frame in in_container.decode(in_video):
            out_frame = frame.reformat(width=width, height=height, format="yuv420p")
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

        for packet in out_stream.encode(None):
            out_container.mux(packet)

        out_container.close()
        in_container.close()

        if out_p.exists() and out_p.stat().st_size > 1000:
            return str(out_p)
    except Exception as e:
        print(f"H.264 web conversion notice for {input_path}: {e}")

    return str(in_p)


