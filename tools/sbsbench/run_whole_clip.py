#!/usr/bin/env python3
"""
Run one continuous clip through Apollo's real Host SBS pipeline.

Unlike ``run_eval.py``, this command accepts arbitrary user media and does not compare it with
committed baselines.  It keeps one native harness process alive for the complete presentation
sequence so scene cuts, adaptive pop, zero-plane latches, EMA, and subject history cannot reset at
an artificial chunk boundary.

Video input uses a bounded, backpressured decoder rather than a whole-clip image spool. Original
presentation timestamps are retained in ``timeline.json`` and used by one persistent optional SBS
encoder, including variable-frame-rate sources. Static PQ/HLG HDR is converted through linear
scRGB PFM interchange and restored with its source color/static metadata; rotation, dynamic HDR,
and ambiguous high-bit-depth formats fail closed.

The native estimator runs exactly once. A future-corroborated streaming planner finalizes
scene-wide pop and zero-anchor values, then cache replay renders that scene without TensorRT.
Results are diagnostic policy outputs, not ground truth or an assertion of universally best
parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import threading
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any, Iterable
from urllib.parse import urlsplit

if __package__:
    from .adaptive_state_contract import TRACE_SCHEMA as ADAPTIVE_TRACE_SCHEMA
else:
    from adaptive_state_contract import TRACE_SCHEMA as ADAPTIVE_TRACE_SCHEMA


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
DEFAULT_BUILD_DIR = REPO / "cmake-build-relwithdebinfo"
DEFAULT_CONF = SCRIPT_DIR / "bench.conf"
MANIFEST_NAME = "conversion_manifest.json"
TIMELINE_NAME = "timeline.json"
TRACE_NAME = "adaptive_state.jsonl"
NATIVE_CONTRACT_NAME = "whole_clip_contract.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CODECS = ("hevc_nvenc", "av1_nvenc", "libx265")
SPOOL_SAFETY_FACTOR = Fraction(5, 4)
SPOOL_FIXED_RESERVE_BYTES = 256 * 1024 * 1024
DEFAULT_SCENE_CACHE_MAX_BYTES = 4 * 1024 * 1024 * 1024
SCENE_CACHE_CONTRACT_SCHEMA = 2
SCENE_CACHE_METADATA_MAGIC = 0x32434253
SCENE_CACHE_METADATA_SCHEMA = 1
SCENE_CACHE_METADATA_WORDS = 48
SCENE_CACHE_METADATA_BYTES = SCENE_CACHE_METADATA_WORDS * 4
SCENE_CACHE_METADATA_ROI_OFFSET = 16
SCENE_CACHE_ROI_WORDS = 32
SCENE_CACHE_STATE_SCHEMA = 2
SCENE_CACHE_SUBJECT_WORDS = 12
SCENE_CACHE_DEPTH_FRAME_STATE_WORDS = 4
SCENE_CACHE_STATE_WORDS = 16
SCENE_CACHE_MAX_DEPTH_DIMENSION = 1036
FRAME_ROI_TRANSFORM_SCHEMA = 1
FRAME_ROI_TRANSFORM_PATCH_SIZE = 14
FRAME_ROI_TRANSFORM_BANK_COUNT = 2
ADAPTIVE_INITIALIZED_WORD = 3
ADAPTIVE_ZERO_ANCHOR_VALID_WORD = 9
ADAPTIVE_CUT_FLAGS_WORD = 10
ADAPTIVE_MODEL_INPUT_HISTORY_STATE_WORD = 11
ADAPTIVE_KNOWN_CUT_FLAG_MASK = 63
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
RESERVED_NATIVE_OPTIONS = {
    "--artifacts",
    "--frames",
    "--follow",
    "--follow-count",
    "--follow-format",
    "--limit",
    "--out",
    "--output-every",
    "--render-cache",
    "--scene-cache",
    "--scene-plan",
}

sys.path.insert(0, str(SCRIPT_DIR))
from split_video import resolve_ffmpeg  # noqa: E402


class WholeClipError(RuntimeError):
    """A user-facing setup, media, or pipeline error."""


class DiskSpaceError(WholeClipError):
    def __init__(self, message: str, preflight: dict[str, Any]):
        super().__init__(message)
        self.preflight = preflight


class SbsFrameHttpBridge:
    """Serve rolling SBS artifacts to one persistent FFmpeg concat demuxer.

    FFmpeg's concat demuxer parses its complete script before opening the first child, so a
    growing script cannot provide bounded conversion. A complete script whose child URLs point
    here preserves the exact timeline while each GET blocks until native rendering publishes that
    one frame. Concat opens children strictly in script order; opening N proves it closed N-1.
    """

    def __init__(
        self,
        frame_count: int,
        extension: str,
        *,
        wait_seconds: float = 900.0,
        delete_released: bool = True,
        token: str | None = None,
    ):
        if frame_count <= 0:
            raise WholeClipError("HTTP SBS bridge requires a positive frame count")
        if extension not in ("png", "pfm"):
            raise WholeClipError("HTTP SBS bridge extension must be png or pfm")
        if not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise WholeClipError("HTTP SBS bridge wait must be positive and finite")
        self.frame_count = frame_count
        self.extension = extension
        self.wait_seconds = wait_seconds
        self.delete_released = delete_released
        self.token = token or secrets.token_urlsafe(32)
        if not re.fullmatch(r"[A-Za-z0-9_-]{24,}", self.token):
            raise WholeClipError("HTTP SBS bridge token is not sufficiently strong")

        self._condition = threading.Condition()
        self._published: dict[int, Path] = {}
        self._published_count = 0
        self._requested = 0
        self._released = 0
        self._served = 0
        self._error: str | None = None
        self._stopping = False

        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                owner._handle_get(self)

            def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self.send_error(405, "HEAD is not supported")

            def log_message(self, _format: str, *args: Any) -> None:
                del args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="sbs-frame-http-bridge",
            daemon=True,
        )
        self._server_thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        if host != "127.0.0.1":
            raise WholeClipError("HTTP SBS bridge is not bound to IPv4 loopback")
        return f"http://127.0.0.1:{port}/{self.token}"

    @property
    def requested_count(self) -> int:
        with self._condition:
            return self._requested

    @property
    def released_count(self) -> int:
        with self._condition:
            return self._released

    @property
    def served_count(self) -> int:
        with self._condition:
            return self._served

    @property
    def error(self) -> str | None:
        with self._condition:
            return self._error

    def wait_for_served(
        self,
        sequence: int,
        encoder: "LoggedSubprocess",
        *,
        timeout_seconds: float,
    ) -> None:
        if not 1 <= sequence <= self.frame_count:
            raise WholeClipError(f"served wait sequence is out of range: {sequence}")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._served < sequence and not self._error:
                return_code = encoder.poll()
                if return_code is not None:
                    encoder.close_log()
                    raise WholeClipError(
                        f"persistent SBS encoder exited {return_code} before consuming "
                        f"frame {sequence}; diagnostics: {encoder.log_path}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WholeClipError(
                        f"timed out waiting for encoder to consume SBS frame {sequence}")
                self._condition.wait(timeout=min(0.05, remaining))
            if self._error:
                raise WholeClipError(f"HTTP SBS bridge failed: {self._error}")

    def frame_url(self, sequence: int) -> str:
        if not 1 <= sequence <= self.frame_count:
            raise WholeClipError(f"HTTP SBS sequence is out of range: {sequence}")
        return (
            f"{self.base_url}/frame_{sequence:010d}.{self.extension}"
        )

    def publish(self, sequence: int, path: Path) -> None:
        path = path.resolve()
        if not 1 <= sequence <= self.frame_count:
            raise WholeClipError(f"cannot publish out-of-range SBS sequence {sequence}")
        if path.suffix.lower() != f".{self.extension}":
            raise WholeClipError(
                f"SBS bridge expected .{self.extension}, got {path.suffix}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise WholeClipError(f"cannot publish missing/empty SBS frame: {path}")
        with self._condition:
            if self._error:
                raise WholeClipError(f"HTTP SBS bridge has failed: {self._error}")
            if sequence in self._published:
                raise WholeClipError(f"SBS sequence {sequence} was already published")
            if sequence != self._published_count + 1:
                raise WholeClipError(
                    "SBS frames must be published monotonically: "
                    f"got {sequence}, expected "
                    f"{self._published_count + 1}")
            self._published[sequence] = path
            self._published_count = sequence
            self._condition.notify_all()

    def abort(self, message: str) -> None:
        with self._condition:
            if not self._error:
                self._error = message
            self._stopping = True
            self._condition.notify_all()

    def close(self, *, encoder_succeeded: bool) -> None:
        if not encoder_succeeded:
            self.abort("encoder did not complete")
        close_error: str | None = None
        with self._condition:
            if encoder_succeeded:
                # FFmpeg can return immediately after reading the final socket byte while the
                # server handler still needs one scheduling slice to publish its served ACK.
                deadline = time.monotonic() + min(self.wait_seconds, 5.0)
                while (
                    self._served < self.frame_count and
                    not self._error and
                    time.monotonic() < deadline
                ):
                    self._condition.wait(timeout=max(
                        0.0, deadline - time.monotonic()))
            if encoder_succeeded and self._error:
                close_error = self._error
            if encoder_succeeded and (
                self._requested != self.frame_count or
                self._served != self.frame_count
            ):
                close_error = (
                    "encoder completed without consuming every SBS frame: "
                    f"requested={self._requested}, served={self._served}, "
                    f"expected={self.frame_count}")
            if (
                encoder_succeeded and self.delete_released and
                self._served == self.frame_count
            ):
                last = self._published.pop(self.frame_count, None)
                if last is None or not last.is_file():
                    close_error = (
                        f"final SBS frame {self.frame_count} is unavailable for release")
                else:
                    try:
                        last.unlink()
                    except OSError as exc:
                        close_error = (
                            f"cannot release final SBS frame {self.frame_count}: {exc}")
                    else:
                        self._released = self.frame_count
            self._stopping = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._server_thread.join(timeout=5.0)
        if self._server_thread.is_alive():
            raise WholeClipError("HTTP SBS bridge server did not stop")
        if close_error:
            raise WholeClipError(close_error)

    def _fail(self, message: str) -> None:
        with self._condition:
            if not self._error:
                self._error = message
            self._condition.notify_all()

    def _send_error(
        self,
        handler: BaseHTTPRequestHandler,
        status: int,
        message: str,
    ) -> None:
        handler.send_error(status, message)
        handler.close_connection = True

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlsplit(handler.path)
        if parsed.query or parsed.fragment:
            self._send_error(handler, 404, "unknown frame")
            return
        expected_prefix = f"/{self.token}/frame_"
        expected_suffix = f".{self.extension}"
        if (
            not parsed.path.startswith(expected_prefix) or
            not parsed.path.endswith(expected_suffix)
        ):
            # Random local requests must not abort a long conversion.
            self._send_error(handler, 404, "unknown frame")
            return
        digits = parsed.path[
            len(expected_prefix):-len(expected_suffix)
        ]
        if len(digits) != 10 or not digits.isdigit():
            self._send_error(handler, 404, "unknown frame")
            return
        sequence = int(digits)
        if not 1 <= sequence <= self.frame_count:
            self._send_error(handler, 404, "unknown frame")
            return

        previous_to_delete: Path | None = None
        deadline = time.monotonic() + self.wait_seconds
        with self._condition:
            expected = self._requested + 1
            if sequence != expected:
                message = (
                    "non-monotonic FFmpeg SBS request: "
                    f"got {sequence}, expected {expected}")
                self._fail(message)
                self._send_error(handler, 409, message)
                return
            self._requested = sequence
            if sequence > 1 and self.delete_released:
                while (
                    self._served < sequence - 1 and
                    not self._error and not self._stopping
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._error = (
                            f"timed out waiting for delivery of SBS frame "
                            f"{sequence - 1}")
                        break
                    self._condition.wait(timeout=remaining)
                if self._error or self._stopping:
                    message = self._error or "SBS bridge stopped"
                    self._send_error(handler, 503, message)
                    return
                previous_to_delete = self._published.pop(sequence - 1, None)
                if previous_to_delete is None:
                    message = (
                        f"prior SBS frame {sequence - 1} was not available for release")
                    self._fail(message)
                    self._send_error(handler, 500, message)
                    return
            while sequence not in self._published and not self._error and not self._stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._error = (
                        f"timed out waiting for SBS frame {sequence}")
                    break
                self._condition.wait(timeout=remaining)
            if self._error or self._stopping:
                message = self._error or "SBS bridge stopped"
                self._send_error(handler, 503, message)
                return
            path = self._published[sequence]

        if previous_to_delete is not None:
            try:
                if not previous_to_delete.is_file():
                    raise FileNotFoundError(previous_to_delete)
                previous_to_delete.unlink()
            except OSError as exc:
                message = (
                    f"cannot release SBS frame {sequence - 1}: {exc}")
                self._fail(message)
                self._send_error(handler, 500, message)
                return
            with self._condition:
                self._released = sequence - 1
                self._condition.notify_all()

        try:
            size = path.stat().st_size
            if size <= 0:
                raise OSError("published frame is empty")
            handler.send_response(200)
            handler.send_header(
                "Content-Type",
                "image/png" if self.extension == "png" else "image/x-portable-floatmap",
            )
            handler.send_header("Content-Length", str(size))
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-Content-Type-Options", "nosniff")
            handler.send_header("Connection", "close")
            handler.end_headers()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    handler.wfile.write(chunk)
            handler.wfile.flush()
            handler.close_connection = True
        except (OSError, ConnectionError) as exc:
            self._fail(f"failed serving SBS frame {sequence}: {exc}")
            return
        with self._condition:
            self._served = sequence
            self._condition.notify_all()


class VideoFollowFrameDecoder:
    """Decode one presentation frame at a time into the native follow protocol.

    Only one frame is read from FFmpeg before control returns to the orchestrator. The stdout
    pipe therefore supplies bounded backpressure without suspending FFmpeg or spooling the clip.
    """

    def __init__(
        self,
        ffmpeg: str,
        source: Path,
        width: int,
        height: int,
        color: dict[str, Any],
        log_path: Path,
    ):
        if width <= 0 or height <= 0:
            raise WholeClipError("streaming decoder requires positive source dimensions")
        self.width = width
        self.height = height
        self.color = color
        self.extension = (
            "pfm" if color.get("mode") in ("hdr-pq", "hdr-hlg") else "png"
        )
        command = [
            ffmpeg,
            "-hide_banner", "-loglevel", "warning", "-xerror", "-nostdin",
            "-noautorotate",
            "-i", os.fspath(source),
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
            "-fps_mode", "passthrough",
        ]
        if self.extension == "pfm":
            command += [
                "-vf", hdr_decode_filter(color),
                "-c:v", "pfm",
                "-f", "image2pipe",
                "pipe:1",
            ]
        else:
            command += [
                "-pix_fmt", "bgra",
                "-f", "rawvideo",
                "pipe:1",
            ]
        self.command = command
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self._finished = False
        self._log = log_path.open(
            "w", encoding="utf-8", errors="replace", newline="\n")
        self._log.write("$ " + command_display(command) + "\n")
        self._log.flush()
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._log,
            )
        except OSError as exc:
            self._log.close()
            raise WholeClipError(f"cannot start {command[0]}: {exc}") from exc
        if self._process.stdout is None:
            self.abort()
            raise WholeClipError("FFmpeg streaming decoder did not expose stdout")
        self._next_sequence = 1

    @property
    def process(self) -> subprocess.Popen[bytes]:
        return self._process

    def _read_exact(self, size: int) -> bytes:
        result = bytearray()
        assert self._process.stdout is not None
        while len(result) < size:
            chunk = self._process.stdout.read(size - len(result))
            if not chunk:
                break
            result.extend(chunk)
        if len(result) != size:
            return_code = self._process.poll()
            raise WholeClipError(
                "FFmpeg streaming decoder ended inside a frame "
                f"({len(result)} of {size} bytes, exit={return_code}); "
                f"diagnostics: {self.log_path}")
        return bytes(result)

    def _read_pfm(self) -> bytes:
        assert self._process.stdout is not None

        def header_line(description: str) -> bytes:
            line = self._process.stdout.readline(130)
            if not line or len(line) > 129 or not line.endswith(b"\n"):
                raise WholeClipError(
                    f"FFmpeg emitted an invalid PFM {description}; "
                    f"diagnostics: {self.log_path}")
            return line

        magic = header_line("magic")
        dimensions = header_line("dimensions")
        scale = header_line("scale")
        if magic.rstrip(b"\r\n") != b"PF":
            raise WholeClipError("FFmpeg PFM stream is not RGB float32")
        try:
            width_text, height_text = dimensions.decode("ascii").split()
            width, height = int(width_text), int(height_text)
            scale_value = float(scale.decode("ascii").strip())
        except (UnicodeError, ValueError) as exc:
            raise WholeClipError("FFmpeg emitted a malformed PFM header") from exc
        if (width, height) != (self.width, self.height):
            raise WholeClipError(
                "FFmpeg PFM dimensions changed during the clip: "
                f"{width}x{height} != {self.width}x{self.height}")
        if scale_value != -1.0:
            raise WholeClipError(
                f"FFmpeg PFM must be native little-endian with scale -1, got {scale_value}")
        payload_size = width * height * 3 * 4
        return magic + dimensions + scale + self._read_exact(payload_size)

    def publish_next(self, directory: Path, sequence: int) -> Path:
        if self._finished:
            raise WholeClipError("cannot publish after the decoder was finalized")
        if sequence != self._next_sequence:
            raise WholeClipError(
                "streaming decoder publication must be monotonic: "
                f"got {sequence}, expected {self._next_sequence}")
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"frame_{sequence:010d}.{self.extension}"
        temporary = final.with_name(final.name + ".part")
        if final.exists() or temporary.exists():
            raise WholeClipError(f"refusing to replace follow frame: {final}")
        if self.extension == "pfm":
            payload = self._read_pfm()
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            payload = self._read_exact(self.width * self.height * 4)
            try:
                from PIL import Image
            except ImportError as exc:
                raise WholeClipError(
                    "SDR streaming decode requires Pillow to publish PNG frames") from exc
            image = Image.frombytes(
                "RGBA", (self.width, self.height), payload, "raw", "BGRA")
            with temporary.open("wb") as stream:
                image.save(stream, format="PNG", compress_level=1)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, final)
        if not final.is_file() or final.stat().st_size <= 0:
            raise WholeClipError(f"failed to publish follow frame atomically: {final}")
        self._next_sequence += 1
        return final

    def finish(self, expected_frame_count: int) -> None:
        if self._finished:
            return
        if self._next_sequence != expected_frame_count + 1:
            raise WholeClipError(
                "streaming decoder finalized at the wrong frame count: "
                f"{self._next_sequence - 1} != {expected_frame_count}")
        assert self._process.stdout is not None
        extra = self._process.stdout.read(1)
        if extra:
            self.abort()
            raise WholeClipError(
                "FFmpeg decoded more frames than the probed presentation timeline")
        return_code = self._process.wait()
        self._process.stdout.close()
        self._log.close()
        self._finished = True
        if return_code:
            raise WholeClipError(
                f"FFmpeg streaming decoder exited {return_code}; "
                f"diagnostics: {self.log_path}")

    def abort(self) -> None:
        if getattr(self, "_finished", True):
            return
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._log.close()
        self._finished = True


class FrameDirectoryFollowProducer:
    """Publish immutable frame-directory input through the same global PNG follow contract."""

    extension = "png"

    def __init__(self, frames: list[Path]):
        if not frames:
            raise WholeClipError("frame-directory producer requires at least one frame")
        self.frames = frames
        self._next_sequence = 1
        self._finished = False
        self.command = None

    def publish_next(self, directory: Path, sequence: int) -> Path:
        if self._finished:
            raise WholeClipError("cannot publish after frame-directory producer finalization")
        if sequence != self._next_sequence or sequence > len(self.frames):
            raise WholeClipError(
                "frame-directory publication is out of sequence: "
                f"got {sequence}, expected {self._next_sequence}")
        source = self.frames[sequence - 1]
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"frame_{sequence:010d}.png"
        temporary = final.with_name(final.name + ".part")
        if final.exists() or temporary.exists():
            raise WholeClipError(f"refusing to replace follow frame: {final}")
        if source.suffix.lower() == ".png":
            try:
                os.link(source, temporary)
            except OSError:
                with source.open("rb") as source_stream, temporary.open("wb") as output:
                    shutil.copyfileobj(source_stream, output, 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
        else:
            try:
                from PIL import Image
                with Image.open(source) as image, temporary.open("wb") as output:
                    image.convert("RGBA").save(
                        output, format="PNG", compress_level=1)
                    output.flush()
                    os.fsync(output.fileno())
            except (ImportError, OSError, ValueError) as exc:
                if temporary.exists():
                    temporary.unlink()
                raise WholeClipError(
                    f"cannot normalize frame-directory image {source}: {exc}") from exc
        os.replace(temporary, final)
        if not final.is_file() or final.stat().st_size <= 0:
            raise WholeClipError(
                f"failed to publish frame-directory follow frame: {final}")
        self._next_sequence += 1
        return final

    def finish(self, expected_frame_count: int) -> None:
        if expected_frame_count != len(self.frames):
            raise WholeClipError(
                "frame-directory expected count changed during conversion")
        if self._next_sequence != expected_frame_count + 1:
            raise WholeClipError(
                "frame-directory producer finalized before every source frame")
        self._finished = True

    def abort(self) -> None:
        self._finished = True


class LoggedSubprocess:
    """A long-running child whose complete diagnostics remain in the run directory."""

    def __init__(
        self,
        command: list[str],
        cwd: Path | None,
        log_path: Path,
    ):
        self.command = command
        self.log_path = log_path
        self._closed = False
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open(
            "w", encoding="utf-8", errors="replace", newline="\n")
        self._log.write("$ " + command_display(command) + "\n")
        self._log.flush()
        try:
            self.process = subprocess.Popen(
                command,
                cwd=os.fspath(cwd) if cwd else None,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            self._log.close()
            self._closed = True
            raise WholeClipError(f"cannot start {command[0]}: {exc}") from exc

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, *, timeout: float | None = None, description: str) -> None:
        try:
            return_code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise WholeClipError(
                f"{description} did not finish within {timeout:g}s; "
                f"diagnostics: {self.log_path}") from exc
        self.close_log()
        if return_code:
            raise WholeClipError(
                f"{description} exited {return_code}; diagnostics: {self.log_path}")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.close_log()

    def close_log(self) -> None:
        if not self._closed:
            self._log.close()
            self._closed = True


def publish_producer_terminal(
    directory: Path,
    *,
    frame_count: int | None = None,
    error: str | None = None,
) -> Path:
    """Atomically finish one native follow stream with an exact, exclusive result."""
    if (frame_count is None) == (error is None):
        raise WholeClipError(
            "producer terminal requires exactly one of frame_count or error")
    done = directory / ".producer-done.json"
    failed = directory / ".producer-failed.json"
    if done.exists() or failed.exists():
        raise WholeClipError("follow producer already published a terminal result")
    if frame_count is not None:
        if not 1 <= frame_count <= 9_999_999_999:
            raise WholeClipError("producer frame count is outside the 10-digit contract")
        path = done
        value = {
            "schema": 1,
            "status": "complete",
            "frame_count": frame_count,
        }
    else:
        assert error is not None
        normalized = error.strip()
        if not normalized:
            normalized = "wrapper producer failed"
        path = failed
        value = {
            "schema": 1,
            "status": "failed",
            "error": normalized[:1024],
        }
    write_json_atomic(path, value)
    return path


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise WholeClipError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WholeClipError(f"{description} must be a JSON object: {path}")
    return value


def wait_for_progress(
    path: Path,
    field: str,
    minimum: int,
    child: LoggedSubprocess,
    *,
    timeout_seconds: float,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wait for an atomic native progress counter while detecting early process death."""
    if minimum < 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise WholeClipError("invalid progress wait bound")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            value = read_json_object(path, "native progress")
        except FileNotFoundError:
            value = None
        if value is not None:
            if expected:
                mismatches = {
                    key: {"expected": wanted, "actual": value.get(key)}
                    for key, wanted in expected.items()
                    if value.get(key) != wanted
                }
                if mismatches:
                    raise WholeClipError(
                        "native progress identity mismatch: " +
                        json.dumps(mismatches, sort_keys=True))
            actual = value.get(field)
            if not isinstance(actual, int) or isinstance(actual, bool) or actual < 0:
                raise WholeClipError(
                    f"native progress field {field!r} is not a nonnegative integer")
            if actual >= minimum:
                return value
            if value.get("status") in ("failed", "complete"):
                raise WholeClipError(
                    f"native progress became {value.get('status')!r} before "
                    f"{field} reached {minimum}")
        return_code = child.poll()
        if return_code is not None:
            child.close_log()
            raise WholeClipError(
                f"native process exited {return_code} before {field} reached {minimum}; "
                f"diagnostics: {child.log_path}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WholeClipError(
                f"timed out waiting for native {field} >= {minimum}; "
                f"diagnostics: {child.log_path}")
        time.sleep(min(0.02, remaining))


def _float_from_word(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _join_u64_words(low: int, high: int) -> int:
    return low | (high << 32)


def _finite_roi_rect(
    words: tuple[int, ...],
    offset: int,
) -> tuple[float, float, float, float] | None:
    rect = tuple(_float_from_word(words[offset + index]) for index in range(4))
    x0, y0, x1, y1 = rect
    if (
        not all(math.isfinite(value) for value in rect) or
        x0 < 0.0 or y0 < 0.0 or
        x1 > 1.0 or y1 > 1.0 or
        x1 <= x0 or y1 <= y0
    ):
        return None
    return rect


def _accepted_roi_axis(value: float, extent: int) -> int:
    return min(max(math.ceil(value * extent - 0.5), 0), extent)


def _roi_physical_aspect_matches(
    crop: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    model_width: int,
    model_height: int,
    tolerance: float,
) -> bool:
    crop_width = (crop[2] - crop[0]) * source_width
    crop_height = (crop[3] - crop[1]) * source_height
    left = crop_width * model_height
    right = crop_height * model_width
    scale = max(abs(left), abs(right), 1.0)
    return (
        math.isfinite(left) and math.isfinite(right) and
        abs(left - right) <= scale * tolerance
    )


def _validate_roi_transform(
    words: tuple[int, ...],
    *,
    expected_source_frame_id: int,
    source_width: int,
    source_height: int,
    model_width: int,
    model_height: int,
    sequence: int,
) -> None:
    valid_flag = 1 << 0
    full_frame_flag = 1 << 1
    active_roi_flag = 1 << 2
    reset_debt_flag = 1 << 3
    known_flags = (
        valid_flag | full_frame_flag | active_roi_flag | reset_debt_flag)
    flags = words[1]
    full_frame = bool(flags & full_frame_flag)
    active_roi = bool(flags & active_roi_flag)
    transform_version = _join_u64_words(words[28], words[29])
    source_frame_id = _join_u64_words(words[2], words[3])
    focus = _finite_roi_rect(words, 12)
    crop = _finite_roi_rect(words, 16)
    if (
        words[0] != FRAME_ROI_TRANSFORM_SCHEMA or
        not flags & valid_flag or
        flags & ~known_flags or
        full_frame == active_roi or
        source_frame_id != expected_source_frame_id or
        words[6] != source_width or words[7] != source_height or
        words[8] != model_width or words[9] != model_height or
        model_width <= 0 or model_height <= 0 or
        model_width > SCENE_CACHE_MAX_DEPTH_DIMENSION or
        model_height > SCENE_CACHE_MAX_DEPTH_DIMENSION or
        model_width > source_width or model_height > source_height or
        model_width % FRAME_ROI_TRANSFORM_PATCH_SIZE != 0 or
        model_height % FRAME_ROI_TRANSFORM_PATCH_SIZE != 0 or
        transform_version == 0 or
        words[30] >= FRAME_ROI_TRANSFORM_BANK_COUNT or
        focus is None or crop is None
    ):
        raise WholeClipError(
            f"scene cache metadata {sequence} ROI transform mismatch")
    assert focus is not None and crop is not None
    if not (
        focus[0] >= crop[0] and focus[1] >= crop[1] and
        focus[2] <= crop[2] and focus[3] <= crop[3]
    ):
        raise WholeClipError(
            f"scene cache metadata {sequence} ROI transform mismatch")

    feathers = tuple(_float_from_word(words[index]) for index in range(24, 28))
    if any(not math.isfinite(value) or value < 0.0 for value in feathers):
        raise WholeClipError(
            f"scene cache metadata {sequence} ROI transform mismatch")

    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    accepted = (
        _accepted_roi_axis((focus[0] - crop[0]) / crop_width, model_width),
        _accepted_roi_axis((focus[1] - crop[1]) / crop_height, model_height),
        _accepted_roi_axis((focus[2] - crop[0]) / crop_width, model_width),
        _accepted_roi_axis((focus[3] - crop[1]) / crop_height, model_height),
    )
    x0, y0, x1, y1 = accepted
    accepted_count = (x1 - x0) * (y1 - y0)
    if (
        x1 <= x0 or y1 <= y0 or
        tuple(words[20:24]) != accepted or
        words[10] != accepted_count
    ):
        raise WholeClipError(
            f"scene cache metadata {sequence} ROI transform mismatch")

    if full_frame:
        full_rect_words = (0, 0, 0x3F800000, 0x3F800000)
        if (
            tuple(words[12:16]) != full_rect_words or
            tuple(words[16:20]) != full_rect_words or
            accepted != (0, 0, model_width, model_height) or
            words[10] != model_width * model_height or
            any(value != 0.0 for value in feathers) or
            not _roi_physical_aspect_matches(
                crop,
                source_width,
                source_height,
                model_width,
                model_height,
                0.02,
            )
        ):
            raise WholeClipError(
                f"scene cache metadata {sequence} ROI transform mismatch")
        return

    focus_width = focus[2] - focus[0]
    focus_height = focus[3] - focus[1]
    if (
        words[4] == 0 or words[5] == 0 or words[11] == 0 or
        feathers[0] > 0.25 * focus_width or
        feathers[2] > 0.25 * focus_width or
        feathers[1] > 0.25 * focus_height or
        feathers[3] > 0.25 * focus_height or
        not _roi_physical_aspect_matches(
            crop,
            source_width,
            source_height,
            model_width,
            model_height,
            0.0001,
        )
    ):
        raise WholeClipError(
            f"scene cache metadata {sequence} ROI transform mismatch")


def _validate_frame_metadata(
    words: tuple[int, ...],
    *,
    sequence: int,
    source_width: int,
    source_height: int,
    depth_reuse_interval: int,
    requires_previous: bool,
) -> int:
    expected_header = (
        SCENE_CACHE_METADATA_MAGIC,
        SCENE_CACHE_METADATA_SCHEMA,
        SCENE_CACHE_METADATA_WORDS,
        SCENE_CACHE_METADATA_ROI_OFFSET,
    )
    if tuple(words[0:4]) != expected_header:
        raise WholeClipError(
            f"scene cache metadata {sequence} header mismatch")
    depth_width, depth_height, depth_pixels, bytes_per_sample = words[4:8]
    pixels = depth_width * depth_height
    if (
        depth_width <= 0 or depth_height <= 0 or
        depth_width > SCENE_CACHE_MAX_DEPTH_DIMENSION or
        depth_height > SCENE_CACHE_MAX_DEPTH_DIMENSION or
        depth_width > source_width or depth_height > source_height or
        depth_width % FRAME_ROI_TRANSFORM_PATCH_SIZE != 0 or
        depth_height % FRAME_ROI_TRANSFORM_PATCH_SIZE != 0 or
        pixels > UINT32_MAX or depth_pixels != pixels or
        bytes_per_sample != 4 or
        _join_u64_words(words[8], words[9]) != sequence
    ):
        raise WholeClipError(
            f"scene cache metadata {sequence} depth identity mismatch")
    if tuple(words[12:16]) != (
        source_width,
        source_height,
        depth_width,
        depth_height,
    ):
        raise WholeClipError(
            f"scene cache metadata {sequence} source identity mismatch")

    if (
        not isinstance(depth_reuse_interval, int) or
        isinstance(depth_reuse_interval, bool) or
        not 1 <= depth_reuse_interval <= 8
    ):
        raise WholeClipError(
            "scene cache depth reuse interval is invalid")
    retained_source_frame_id = _join_u64_words(words[10], words[11])
    transform = tuple(words[SCENE_CACHE_METADATA_ROI_OFFSET:])
    if all(word == 0 for word in transform):
        left = source_width * depth_height
        right = source_height * depth_width
        scale = max(abs(left), abs(right), 1.0)
        if (
            retained_source_frame_id != UINT64_MAX or
            abs(left - right) > scale * 0.02
        ):
            raise WholeClipError(
                f"scene cache metadata {sequence} retained source identity mismatch")
    else:
        expected_retained_source_frame_id = (
            ((sequence - 1) // depth_reuse_interval) *
            depth_reuse_interval
        )
        cadence_aligned = (
            retained_source_frame_id <= expected_retained_source_frame_id and
            retained_source_frame_id % depth_reuse_interval == 0
        )
        state_matches_retained_identity = (
            retained_source_frame_id < expected_retained_source_frame_id
            if requires_previous else
            retained_source_frame_id == expected_retained_source_frame_id
        )
        if not cadence_aligned or not state_matches_retained_identity:
            raise WholeClipError(
                f"scene cache metadata {sequence} retained source identity mismatch")
        _validate_roi_transform(
            transform,
            expected_source_frame_id=retained_source_frame_id,
            source_width=source_width,
            source_height=source_height,
            model_width=depth_width,
            model_height=depth_height,
            sequence=sequence,
        )
    return depth_pixels * bytes_per_sample


def _validate_cached_state(payload: bytes, sequence: int) -> bool:
    words = struct.unpack(f"<{SCENE_CACHE_STATE_WORDS}I", payload)
    values = tuple(_float_from_word(word) for word in words)
    if not all(math.isfinite(value) for value in values):
        raise WholeClipError(
            f"scene cache state {sequence} contains a non-finite value")
    subject_initialized = values[ADAPTIVE_INITIALIZED_WORD]
    zero_anchor_valid = values[ADAPTIVE_ZERO_ANCHOR_VALID_WORD]
    cut_flags = values[ADAPTIVE_CUT_FLAGS_WORD]
    history_state = values[ADAPTIVE_MODEL_INPUT_HISTORY_STATE_WORD]
    initialized = values[SCENE_CACHE_SUBJECT_WORDS + 2]
    frame_state = values[SCENE_CACHE_SUBJECT_WORDS + 3]
    frame_state_valid = (
        frame_state == 0.0 or
        (
            initialized == 1.0 and
            frame_state in (1.0, 2.0)
        )
    )
    if (
        subject_initialized not in (0.0, 1.0) or
        zero_anchor_valid not in (0.0, 1.0) or
        not 0.0 <= cut_flags <= ADAPTIVE_KNOWN_CUT_FLAG_MASK or
        math.trunc(cut_flags) != cut_flags or
        not 0.0 <= history_state <= 3.0 or
        math.trunc(history_state) != history_state or
        initialized not in (0.0, 1.0) or
        not frame_state_valid
    ):
        raise WholeClipError(
            f"scene cache state {sequence} has invalid validity flags")
    return initialized == 1.0 and frame_state == 0.0


def scene_cache_max_triplet_bytes(
    source_width: int,
    source_height: int,
) -> int:
    """Bound every dynamic schema-2 depth/state/metadata publication."""
    for name, value in (
        ("width", source_width),
        ("height", source_height),
    ):
        if (
            not isinstance(value, int) or isinstance(value, bool) or
            not 1 <= value <= UINT32_MAX
        ):
            raise WholeClipError(
                f"scene-cache source {name} is invalid")
    max_width = min(
        source_width, SCENE_CACHE_MAX_DEPTH_DIMENSION)
    max_height = min(
        source_height, SCENE_CACHE_MAX_DEPTH_DIMENSION)
    max_width -= max_width % FRAME_ROI_TRANSFORM_PATCH_SIZE
    max_height -= max_height % FRAME_ROI_TRANSFORM_PATCH_SIZE
    if max_width <= 0 or max_height <= 0:
        raise WholeClipError(
            "source raster cannot hold a valid scene-cache depth shape")
    return (
        max_width * max_height * 4 +
        SCENE_CACHE_STATE_WORDS * 4 +
        SCENE_CACHE_METADATA_BYTES
    )


def preflight_scene_cache_hard_cap(
    source_width: int,
    source_height: int,
    hard_cap_bytes: int,
) -> int:
    """Reject a cap before the native producer can publish its first triplet."""
    if (
        not isinstance(hard_cap_bytes, int) or
        isinstance(hard_cap_bytes, bool) or
        hard_cap_bytes <= 0
    ):
        raise WholeClipError(
            "scene-cache max bytes must be a positive integer")
    maximum_triplet = scene_cache_max_triplet_bytes(
        source_width, source_height)
    if maximum_triplet > hard_cap_bytes // 10:
        raise WholeClipError(
            "scene-cache budget must hold at least ten maximum-sized "
            "depth/state/metadata triplets before native publication: "
            f"maximum_pair={maximum_triplet}, cap={hard_cap_bytes}")
    return maximum_triplet


class SceneCacheLedger:
    """Track only native-attested depth/state pairs and enforce a hard byte budget."""

    PRESSURE_FRACTION = Fraction(4, 5)
    FORCE_FRACTION = Fraction(9, 10)

    def __init__(self, directory: Path, max_bytes: int):
        if (
            not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or
            max_bytes <= 0
        ):
            raise WholeClipError("scene-cache max bytes must be a positive integer")
        self.directory = directory
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.high_water_bytes = 0
        self._pairs: dict[int, tuple[Path, Path, Path, int, bool]] = {}
        self.forced_segments: list[dict[str, Any]] = []

    def acknowledge_pair(
        self,
        sequence: int,
        contract: dict[str, Any],
    ) -> int:
        if sequence in self._pairs:
            raise WholeClipError(f"scene cache pair {sequence} was already acknowledged")
        if sequence <= 0:
            raise WholeClipError("scene cache sequence must be positive")
        if (
            contract.get("schema") != SCENE_CACHE_CONTRACT_SCHEMA or
            contract.get("status") != "running" or
            contract.get("first_sequence") != 1 or
            contract.get("atomic_frame_publication") is not True
        ):
            raise WholeClipError(
                "scene cache running publication contract is invalid")
        processed_count = contract.get("processed_count")
        if (
            not isinstance(processed_count, int) or
            isinstance(processed_count, bool) or
            processed_count != sequence
        ):
            raise WholeClipError(
                "scene cache contract does not exactly acknowledge the requested sequence")
        try:
            source_width = contract["source"]["width"]
            source_height = contract["source"]["height"]
            depth_reuse_interval = contract[
                "render_config"]["depth_reuse_interval"]
            state_bytes = int(contract["state"]["bytes_per_frame"])
            metadata_bytes = int(
                contract["frame_metadata"]["bytes_per_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WholeClipError(
                "scene cache contract lacks exact source/state/metadata layout") from exc
        if (
            not isinstance(source_width, int) or isinstance(source_width, bool) or
            not isinstance(source_height, int) or isinstance(source_height, bool) or
            not 1 <= source_width <= UINT32_MAX or
            not 1 <= source_height <= UINT32_MAX
        ):
            raise WholeClipError("scene cache contract source dimensions are invalid")
        state_contract = contract.get("state", {})
        metadata_contract = contract.get("frame_metadata", {})
        if (
            contract.get("depth", {}).get("dimensions") != "per-frame-metadata" or
            contract.get("depth", {}).get("dtype") != "float32-le" or
            contract.get("depth", {}).get("dxgi_format") != "R32_FLOAT" or
            contract.get("depth", {}).get("bytes_per_frame") is not None or
            contract.get("depth", {}).get("bytes_per_sample") != 4 or
            metadata_contract.get("schema") != SCENE_CACHE_METADATA_SCHEMA or
            metadata_contract.get("magic") != SCENE_CACHE_METADATA_MAGIC or
            metadata_contract.get("word_count") != SCENE_CACHE_METADATA_WORDS or
            metadata_contract.get("roi_transform_word_offset") !=
                SCENE_CACHE_METADATA_ROI_OFFSET or
            metadata_contract.get("roi_transform_word_count") !=
                SCENE_CACHE_ROI_WORDS or
            metadata_contract.get("roi_transform_contract_schema") !=
                FRAME_ROI_TRANSFORM_SCHEMA or
            state_contract.get("schema") != SCENE_CACHE_STATE_SCHEMA or
            state_contract.get("subject_word_count") !=
                SCENE_CACHE_SUBJECT_WORDS or
            state_contract.get("depth_frame_state_word_count") !=
                SCENE_CACHE_DEPTH_FRAME_STATE_WORDS or
            state_contract.get("word_count") != SCENE_CACHE_STATE_WORDS or
            state_bytes != SCENE_CACHE_STATE_WORDS * 4 or
            metadata_bytes != SCENE_CACHE_METADATA_BYTES
        ):
            raise WholeClipError("scene cache schema-2 layout is invalid")
        stem = f"frame_{sequence:010d}"
        depth = self.directory / f"{stem}.depth.r32f"
        state = self.directory / f"{stem}.state.u32"
        metadata = self.directory / f"{stem}.meta.u32"
        try:
            actual_depth = depth.stat().st_size
            actual_state = state.stat().st_size
            actual_metadata = metadata.stat().st_size
            metadata_payload = metadata.read_bytes()
            state_payload = state.read_bytes()
        except OSError as exc:
            raise WholeClipError(
                f"scene cache ACK {sequence} is missing its frame triplet") from exc
        if (
            actual_metadata != metadata_bytes or
            len(metadata_payload) != metadata_bytes
        ):
            raise WholeClipError(
                f"scene cache metadata {sequence} size mismatch")
        metadata_words = struct.unpack(
            f"<{SCENE_CACHE_METADATA_WORDS}I", metadata_payload)
        if (
            actual_state != state_bytes or
            len(state_payload) != state_bytes
        ):
            raise WholeClipError(
                f"scene cache triplet {sequence} size mismatch: "
                f"state={actual_state}/{state_bytes}")
        requires_previous = _validate_cached_state(state_payload, sequence)
        expected_depth = _validate_frame_metadata(
            metadata_words,
            sequence=sequence,
            source_width=source_width,
            source_height=source_height,
            depth_reuse_interval=depth_reuse_interval,
            requires_previous=requires_previous,
        )
        if (
            actual_depth != expected_depth or
            actual_state != state_bytes
        ):
            raise WholeClipError(
                f"scene cache triplet {sequence} size mismatch: "
                f"depth={actual_depth}/{expected_depth}, "
                f"state={actual_state}/{state_bytes}")
        pair_bytes = actual_depth + actual_state + actual_metadata
        projected = self.current_bytes + pair_bytes
        if projected > self.max_bytes:
            raise WholeClipError(
                "scene cache exceeded its hard byte budget before replay release: "
                f"{projected} > {self.max_bytes}")
        self._pairs[sequence] = (
            depth,
            state,
            metadata,
            pair_bytes,
            requires_previous,
        )
        self.current_bytes = projected
        self.high_water_bytes = max(self.high_water_bytes, projected)
        return pair_bytes

    def requires_previous_packed_frame(self, sequence: int) -> bool:
        pair = self._pairs.get(sequence)
        if pair is None:
            raise WholeClipError(
                f"scene cache pair {sequence} was not acknowledged")
        return pair[4]

    def pressure_state(self, next_pair_bytes: int = 0) -> str:
        if next_pair_bytes < 0:
            raise WholeClipError("projected scene-cache pair bytes cannot be negative")
        projected = self.current_bytes + next_pair_bytes
        if projected >= self.max_bytes:
            return "blocked"
        if projected * self.FORCE_FRACTION.denominator >= (
            self.max_bytes * self.FORCE_FRACTION.numerator
        ):
            return "force-finalize"
        if projected * self.PRESSURE_FRACTION.denominator >= (
            self.max_bytes * self.PRESSURE_FRACTION.numerator
        ):
            return "pressure"
        return "normal"

    def record_forced_segment(
        self,
        start_sequence: int,
        end_sequence_exclusive: int,
        *,
        reason: str = "scene-cache-capacity",
    ) -> None:
        if start_sequence <= 0 or end_sequence_exclusive <= start_sequence:
            raise WholeClipError("forced scene-cache segment has an invalid range")
        self.forced_segments.append({
            "start_sequence": start_sequence,
            "end_sequence_exclusive": end_sequence_exclusive,
            "reason": reason,
            "semantic_boundary": False,
            "estimator_reset": False,
            "whole_scene_lookahead": False,
        })

    def release_through(self, end_sequence_exclusive: int) -> dict[str, int]:
        if end_sequence_exclusive <= 1:
            return {"pairs": 0, "bytes": 0}
        sequences = sorted(
            sequence for sequence in self._pairs
            if sequence < end_sequence_exclusive
        )
        removed_bytes = 0
        for sequence in sequences:
            depth, state, metadata, pair_bytes, _requires_previous = (
                self._pairs[sequence])
            try:
                depth.unlink()
                state.unlink()
                metadata.unlink()
            except OSError as exc:
                raise WholeClipError(
                    f"cannot release rendered scene-cache pair {sequence}: {exc}") from exc
            removed_bytes += pair_bytes
            del self._pairs[sequence]
        self.current_bytes -= removed_bytes
        if self.current_bytes < 0:
            raise WholeClipError("scene cache ledger underflow")
        return {"pairs": len(sequences), "bytes": removed_bytes}

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_bytes": self.max_bytes,
            "pressure_at_bytes": (
                self.max_bytes * self.PRESSURE_FRACTION.numerator //
                self.PRESSURE_FRACTION.denominator
            ),
            "force_finalize_at_bytes": (
                self.max_bytes * self.FORCE_FRACTION.numerator //
                self.FORCE_FRACTION.denominator
            ),
            "hard_block_at_bytes": self.max_bytes,
            "current_bytes": self.current_bytes,
            "high_water_bytes": self.high_water_bytes,
            "open_pair_count": len(self._pairs),
            "forced_segments": list(self.forced_segments),
        }


def validate_packed_sbs_contract(
    contract: dict[str, Any],
    expected_extension: str,
) -> tuple[int, int]:
    packed = contract.get("packed_sbs")
    if not isinstance(packed, dict):
        raise WholeClipError("scene cache contract lacks packed_sbs geometry")
    try:
        eye_width = int(packed["eye_width"])
        eye_height = int(packed["eye_height"])
        width = int(packed["width"])
        height = int(packed["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WholeClipError(
            "scene cache packed_sbs dimensions are invalid") from exc
    render_config = contract.get("render_config")
    if not isinstance(render_config, dict):
        raise WholeClipError("scene cache contract lacks render_config")
    simulate_hdr = bool(render_config.get("simulate_hdr"))
    expected_format = (
        "linear-scRGB-f32-pfm"
        if expected_extension == "pfm" else
        (
            "tone-mapped-sRGB-BGRA8-PNG-preview"
            if simulate_hdr else "sRGB-BGRA8-PNG"
        )
    )
    expected_texture = (
        "R16G16B16A16_FLOAT"
        if expected_extension == "pfm" or simulate_hdr else
        "B8G8R8A8_UNORM"
    )
    expected = {
        "width": eye_width * 2,
        "height": eye_height,
        "file_extension": expected_extension,
        "file_pattern": f"sbs_%010d.{expected_extension}",
        "frame_format": expected_format,
        "texture_format": expected_texture,
        "atomic_replay_publication": True,
    }
    actual = {
        "width": width,
        "height": height,
        "file_extension": packed.get("file_extension"),
        "file_pattern": packed.get("file_pattern"),
        "frame_format": packed.get("frame_format"),
        "texture_format": packed.get("texture_format"),
        "atomic_replay_publication": packed.get(
            "atomic_replay_publication"),
    }
    mismatches = {
        key: {"expected": wanted, "actual": actual.get(key)}
        for key, wanted in expected.items()
        if actual.get(key) != wanted
    }
    if (
        eye_width <= 0 or eye_height <= 0 or
        width <= 0 or height <= 0
    ):
        mismatches["positive_dimensions"] = {
            "expected": True,
            "actual": [eye_width, eye_height, width, height],
        }
    if mismatches:
        raise WholeClipError(
            "scene cache packed SBS contract mismatch: " +
            json.dumps(mismatches, sort_keys=True))
    return width, height


def published_frame_dimensions(path: Path, extension: str) -> tuple[int, int]:
    if extension == "png":
        return frame_dimensions(path)
    if extension != "pfm":
        raise WholeClipError(f"unsupported published frame extension: {extension}")
    try:
        with path.open("rb") as stream:
            magic = stream.readline(130).rstrip(b"\r\n")
            dimensions = stream.readline(130).decode("ascii").split()
            scale = stream.readline(130).decode("ascii").strip()
        if magic != b"PF" or len(dimensions) != 2 or float(scale) != -1.0:
            raise ValueError("invalid PFM header")
        width, height = int(dimensions[0]), int(dimensions[1])
    except (OSError, UnicodeError, ValueError) as exc:
        raise WholeClipError(f"cannot inspect native PFM output {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise WholeClipError(f"native PFM output has invalid dimensions: {path}")
    return width, height


class AdaptiveTraceTail:
    """Read exactly one durable, contract-validated native trace frame per follow ACK."""

    def __init__(self, path: Path):
        try:
            import adaptive_clip_report
        except ModuleNotFoundError as exc:
            raise WholeClipError(
                "adaptive trace streaming requires adaptive_clip_report.py") from exc
        self.path = path
        self._contract_error = adaptive_clip_report.TraceContractError
        self._decoder = adaptive_clip_report.IncrementalTraceDecoder()
        self._stream = None

    @property
    def header(self) -> dict[str, Any] | None:
        return self._decoder.header

    @property
    def frame_count(self) -> int:
        return self._decoder.frame_count

    def _open_if_ready(self) -> bool:
        if self._stream is not None:
            return True
        try:
            self._stream = self.path.open(
                "r", encoding="utf-8", errors="strict", newline="")
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise WholeClipError(
                f"cannot open growing adaptive trace {self.path}: {exc}") from exc
        return True

    def read_frame(
        self,
        sequence: int,
        child: LoggedSubprocess,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if sequence != self.frame_count + 1:
            raise WholeClipError(
                "adaptive trace tail must be consumed monotonically: "
                f"got {sequence}, expected {self.frame_count + 1}")
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._open_if_ready():
                assert self._stream is not None
                offset = self._stream.tell()
                line = self._stream.readline()
                if line:
                    if not line.endswith("\n"):
                        # The writer has not yet made the record durable. Rewind so the complete
                        # line is validated exactly once after the matching native ACK.
                        self._stream.seek(offset)
                    else:
                        try:
                            frame = self._decoder.feed_line(line)
                        except self._contract_error as exc:
                            raise WholeClipError(
                                f"invalid streaming adaptive trace: {exc}") from exc
                        if frame is None:
                            continue
                        expected_id = f"{sequence:010d}"
                        if (
                            frame["frame_id"] != expected_id or
                            frame["source_index"] != sequence - 1
                        ):
                            raise WholeClipError(
                                "adaptive trace identity differs from the follow ACK: "
                                f"{frame['frame_id']}/{frame['source_index']} != "
                                f"{expected_id}/{sequence - 1}")
                        return frame
            return_code = child.poll()
            if return_code is not None:
                child.close_log()
                raise WholeClipError(
                    f"native process exited {return_code} before adaptive trace "
                    f"frame {sequence}; diagnostics: {child.log_path}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WholeClipError(
                    f"timed out waiting for adaptive trace frame {sequence}; "
                    f"diagnostics: {child.log_path}")
            time.sleep(min(0.02, remaining))

    def finish(self, expected_frame_count: int) -> dict[str, Any]:
        if self.frame_count != expected_frame_count:
            raise WholeClipError(
                "adaptive trace tail finished at the wrong frame count: "
                f"{self.frame_count} != {expected_frame_count}")
        if self._stream is None:
            raise WholeClipError("adaptive trace was never opened")
        trailing = self._stream.read()
        if trailing:
            raise WholeClipError(
                "adaptive trace contains records beyond the acknowledged source count")
        try:
            return self._decoder.finalize()
        except self._contract_error as exc:
            raise WholeClipError(f"adaptive trace did not finalize: {exc}") from exc

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


_SHOWINFO_TIME_BASE_RE = re.compile(
    r"\bconfig in time_base:\s*(?P<num>\d+)\s*/\s*(?P<den>\d+)")
_SHOWINFO_FRAME_RE = re.compile(
    r"\bn:\s*(?P<n>\d+)\s+pts:\s*(?P<pts>-?\d+)\s+"
    r"pts_time:\s*(?P<pts_time>[-+0-9.eE]+)")
_SHOWINFO_DURATION_RE = re.compile(
    r"\bduration:\s*(?P<duration>-?\d+|N/A)\s+"
    r"duration_time:\s*(?P<duration_time>[-+0-9.eE]+|N/A)")
_FRAME_ID_RE = re.compile(r"_([0-9]+)$")
_HIGH_BIT_DEPTH_RE = re.compile(
    r"(?:p|gbrp|gray|rgb|bgr)(?:9|10|12|14|16)(?:le|be)?(?:\b|[,(])",
    re.IGNORECASE,
)


def _json_ready(value: Any) -> Any:
    """Convert metadata from third-party helpers into deterministic JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_ready(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame_set(root: Path, frames: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in frames:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise WholeClipError(f"runtime asset tree is missing: {root}")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise WholeClipError(f"runtime asset tree is empty: {root}")
    digest = hashlib.sha256()
    files = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_hash))
        files.append({
            "path": relative,
            "sha256": file_hash,
            "size_bytes": size,
        })
    return {
        "path": os.fspath(root),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def runtime_provenance(
    build_dir: Path,
    native_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate and hash the exact model, engine, and shaders used by the harness."""
    model = native_contract.get("model")
    if (
        not isinstance(model, str) or not model or
        Path(model).name != model or len(Path(model).parts) != 1
    ):
        raise WholeClipError(
            f"native contract has an invalid runtime model identity: {model!r}")
    resolved = native_contract.get("resolved_runtime")
    if not isinstance(resolved, dict) or resolved.get("model") != model:
        raise WholeClipError(
            "native contract resolved-runtime model does not match its top-level model")

    assets_root = build_dir / "assets"
    manifest_path = assets_root / f"{model}.active-engine.json"
    try:
        active_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WholeClipError(
            f"missing/invalid active engine manifest {manifest_path}: {exc}") from exc
    if active_manifest.get("schema") != 1:
        raise WholeClipError(
            f"active engine manifest schema is not 1: {active_manifest.get('schema')!r}")
    if active_manifest.get("model") != model:
        raise WholeClipError(
            "active engine manifest model does not match the native runtime")
    engine_name = active_manifest.get("engine")
    if (
        not isinstance(engine_name, str) or not engine_name or
        Path(engine_name).name != engine_name or len(Path(engine_name).parts) != 1
    ):
        raise WholeClipError("active engine manifest engine must be one filename")

    onnx_path = assets_root / f"{model}.onnx"
    engine_path = assets_root / engine_name
    if not onnx_path.is_file():
        raise WholeClipError(f"runtime ONNX source is missing: {onnx_path}")
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        raise WholeClipError(f"active runtime engine is missing/empty: {engine_path}")
    onnx_hash = sha256_file(onnx_path)
    if active_manifest.get("onnx_sha256") != onnx_hash:
        raise WholeClipError(
            "active engine manifest ONNX SHA-256 does not match the runtime source")

    return {
        "assets_root": os.fspath(assets_root),
        "active_engine_manifest": {
            "path": os.fspath(manifest_path),
            "sha256": sha256_file(manifest_path),
            "value": active_manifest,
        },
        "onnx": {
            "path": os.fspath(onnx_path),
            "sha256": onnx_hash,
            "size_bytes": onnx_path.stat().st_size,
        },
        "engine": {
            "path": os.fspath(engine_path),
            "name": engine_name,
            "sha256": sha256_file(engine_path),
            "size_bytes": engine_path.stat().st_size,
        },
        "shaders": hash_tree(assets_root / "shaders"),
    }


def command_display(command: Iterable[os.PathLike[str] | str]) -> str:
    return subprocess.list2cmdline([os.fspath(item) for item in command])


def resolve_ffprobe(
    explicit: str | None = None,
    ffmpeg: str | None = None,
) -> str:
    """Resolve a real ffprobe executable; never infer HDR from FFmpeg prose."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    environment_probe = os.environ.get("FFPROBE_EXE")
    if environment_probe:
        candidates.append(environment_probe)
    discovered = shutil.which("ffprobe")
    if discovered:
        candidates.append(discovered)
    if ffmpeg:
        ffmpeg_path = Path(ffmpeg).expanduser().resolve()
        for name in ("ffprobe.exe", "ffprobe"):
            candidates.append(os.fspath(ffmpeg_path.with_name(name)))
    for candidate in candidates:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        path = Path(expanded)
        if path.is_file():
            return os.fspath(path.resolve())
        located = shutil.which(expanded)
        if located:
            return os.fspath(Path(located).resolve())
    raise WholeClipError(
        "video input requires ffprobe for exact timestamps and color metadata; "
        "install FFmpeg with ffprobe, set FFPROBE_EXE, or pass --ffprobe")


def run_probe_command(
    command: list[str],
    output_path: Path,
    log_path: Path,
) -> None:
    """Keep machine-readable probe output separate from human diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output, log_path.open(
        "w", encoding="utf-8", errors="replace", newline="\n"
    ) as log:
        log.write("$ " + command_display(command) + "\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=log,
                check=False,
            )
        except OSError as exc:
            raise WholeClipError(f"cannot start {command[0]}: {exc}") from exc
    if result.returncode:
        raise WholeClipError(
            f"probe command exited {result.returncode}; diagnostics: {log_path}")


def probe_video_summary(
    ffprobe: str,
    source: Path,
    output_path: Path,
    log_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-read_intervals", "%+#1",
        "-show_streams",
        "-show_frames",
        "-show_format",
        "-of", "json",
        os.fspath(source),
    ]
    run_probe_command(command, output_path, log_path)
    try:
        with output_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise WholeClipError(f"ffprobe summary is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise WholeClipError("ffprobe summary must be a JSON object")
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise WholeClipError("ffprobe did not resolve exactly one primary video stream")
    frames = value.get("frames")
    if not isinstance(frames, list) or not frames:
        raise WholeClipError("ffprobe did not decode the primary video stream's first frame")
    return value, command


def _probe_fraction(value: Any, description: str) -> Fraction:
    if not isinstance(value, str):
        raise WholeClipError(f"ffprobe {description} is missing")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise WholeClipError(f"ffprobe {description} is invalid: {value!r}") from exc
    if result <= 0:
        raise WholeClipError(f"ffprobe {description} must be positive")
    return result


def _canonical_side_data(
    summary: dict[str, Any],
    side_data_type: str,
) -> dict[str, Any] | None:
    def canonical_value(value: Any) -> Any:
        if isinstance(value, str) and re.fullmatch(r"-?\d+/-?\d+", value):
            try:
                fraction = Fraction(value)
            except (ValueError, ZeroDivisionError):
                return value
            return f"{fraction.numerator}/{fraction.denominator}"
        return value

    candidates = []
    sources = list(summary.get("streams") or []) + list(summary.get("frames") or [])
    for source in sources:
        if not isinstance(source, dict):
            continue
        for item in source.get("side_data_list") or []:
            if (
                isinstance(item, dict) and
                item.get("side_data_type") == side_data_type
            ):
                candidates.append({
                    key: canonical_value(item[key]) for key in sorted(item)
                    if key != "side_data_type"
                })
    if not candidates:
        return None
    first = candidates[0]
    if any(candidate != first for candidate in candidates[1:]):
        raise WholeClipError(
            f"conflicting {side_data_type} values in the probed source")
    return first


def classify_video_color(summary: dict[str, Any]) -> dict[str, Any]:
    stream = summary["streams"][0]
    frame = summary["frames"][0]
    if not isinstance(stream, dict) or not isinstance(frame, dict):
        raise WholeClipError("ffprobe video stream/frame descriptors are invalid")
    rotations = []
    tags = stream.get("tags")
    if isinstance(tags, dict) and "rotate" in tags:
        rotations.append(tags["rotate"])
    for item in stream.get("side_data_list") or []:
        if isinstance(item, dict) and "rotation" in item:
            rotations.append(item["rotation"])
    for rotation_value in rotations:
        try:
            rotation = float(rotation_value)
        except (TypeError, ValueError) as exc:
            raise WholeClipError(
                f"ffprobe reported invalid rotation {rotation_value!r}") from exc
        if not math.isfinite(rotation) or not math.isclose(
            rotation % 360.0, 0.0, abs_tol=1e-6
        ):
            raise WholeClipError(
                f"rotated video is not supported by the offline frame contract ({rotation}°)")

    properties = {}
    for key in (
        "pix_fmt", "color_range", "color_space",
        "color_transfer", "color_primaries",
    ):
        stream_value = stream.get(key)
        frame_value = frame.get(key)
        known_stream = stream_value not in (None, "", "unknown", "reserved")
        known_frame = frame_value not in (None, "", "unknown", "reserved")
        if known_stream and known_frame and stream_value != frame_value:
            raise WholeClipError(
                f"ffprobe stream/frame {key} mismatch: "
                f"{stream_value!r} != {frame_value!r}")
        properties[key] = stream_value if known_stream else frame_value

    side_data_types = []
    for source in list(summary.get("streams") or []) + list(summary.get("frames") or []):
        if not isinstance(source, dict):
            continue
        for item in source.get("side_data_list") or []:
            if isinstance(item, dict) and isinstance(item.get("side_data_type"), str):
                side_data_types.append(item["side_data_type"])
    lowered_types = " ".join(side_data_types).lower()
    if "dovi" in lowered_types or "dolby vision" in lowered_types:
        raise WholeClipError(
            "Dolby Vision input is not supported without an exact metadata round trip")
    if "hdr10+" in lowered_types or "smpte2094" in lowered_types:
        raise WholeClipError(
            "dynamic HDR10+ input is not supported without per-frame metadata validation")

    transfer = properties["color_transfer"]
    hdr_transfer = transfer in ("smpte2084", "arib-std-b67")
    pixel_format = str(properties["pix_fmt"] or "")
    high_depth = bool(re.search(r"(?:p0?10|p0?12|10le|12le|16le)", pixel_format))
    if hdr_transfer:
        for key in ("pix_fmt", "color_range", "color_space", "color_primaries"):
            if properties[key] in (None, "", "unknown", "reserved"):
                raise WholeClipError(
                    f"HDR video requires explicit ffprobe {key} metadata")
        if not high_depth:
            raise WholeClipError(
                f"HDR transfer {transfer} requires a high-bit-depth source, got {pixel_format}")
        mode = "hdr-pq" if transfer == "smpte2084" else "hdr-hlg"
    else:
        if high_depth:
            raise WholeClipError(
                "high-bit-depth non-PQ/HLG video needs an explicit SDR precision path")
        mode = "sdr"

    mastering = _canonical_side_data(summary, "Mastering display metadata")
    content_light = _canonical_side_data(summary, "Content light level metadata")
    return {
        "mode": mode,
        **properties,
        "mastering_display": mastering,
        "content_light_level": content_light,
        "side_data_types": sorted(set(side_data_types)),
    }


def probe_video_timeline(
    ffprobe: str,
    source: Path,
    summary: dict[str, Any],
    output_path: Path,
    log_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Scan presentation timestamps once without retaining decoded pixels."""
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pts,best_effort_timestamp,duration:"
        "frame_side_data=side_data_type",
        "-of", "csv=p=0",
        os.fspath(source),
    ]
    run_probe_command(command, output_path, log_path)
    stream = summary["streams"][0]
    time_base = _probe_fraction(stream.get("time_base"), "video time_base")
    rows = []
    dynamic_types: set[str] = set()
    try:
        with output_path.open("r", encoding="utf-8", errors="strict", newline="") as data:
            for index, fields in enumerate(csv.reader(data)):
                if len(fields) < 3:
                    raise WholeClipError(
                        f"ffprobe timeline row {index} has fewer than three fields")
                pts_text = fields[0]
                if pts_text == "N/A":
                    pts_text = fields[1]
                if pts_text == "N/A":
                    raise WholeClipError(
                        f"ffprobe timeline frame {index} has no presentation timestamp")
                try:
                    pts = int(pts_text)
                    duration = None if fields[2] == "N/A" else int(fields[2])
                except ValueError as exc:
                    raise WholeClipError(
                        f"ffprobe timeline frame {index} has invalid integer timing") from exc
                if index and pts <= rows[-1]["pts"]:
                    raise WholeClipError(
                        "ffprobe video PTS are not strictly increasing in presentation order")
                for side_data_type in fields[3:]:
                    lowered = side_data_type.lower()
                    if (
                        "dovi" in lowered or "dolby vision" in lowered or
                        "hdr10+" in lowered or "smpte2094" in lowered or
                        "dynamic hdr" in lowered or "hdr dynamic" in lowered
                    ):
                        dynamic_types.add(side_data_type)
                rows.append({
                    "n": index,
                    "pts": pts,
                    "pts_time_text": str(float(pts * time_base)),
                    "duration": duration,
                    "duration_time_text": (
                        None if duration is None else
                        str(float(duration * time_base))
                    ),
                })
    except OSError as exc:
        raise WholeClipError(f"cannot read ffprobe timeline: {exc}") from exc
    if not rows:
        raise WholeClipError("ffprobe timeline contains no video frames")
    if dynamic_types:
        raise WholeClipError(
            "unsupported dynamic HDR/Dolby side data appeared after the first frame: " +
            ", ".join(sorted(dynamic_types)))

    fps_fraction: Fraction | None = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            fps_fraction = _probe_fraction(
                stream.get(key), key.replace("_", " "))
        except WholeClipError:
            continue
        break
    if fps_fraction is None:
        raise WholeClipError("ffprobe did not report a usable video frame rate")
    timeline = build_video_timeline(
        time_base,
        rows,
        {"fps": float(fps_fraction)},
    )
    timeline["source"] = "ffprobe-full-presentation-scan"
    scan = {
        "frame_count": len(rows),
        "dynamic_metadata_types": [],
        "time_base": {
            "num": time_base.numerator,
            "den": time_base.denominator,
        },
    }
    return timeline, scan, command


def inspect_streaming_video(
    ffprobe: str,
    source: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Resolve exact color, dimensions, and the complete source presentation clock."""
    summary_path = work_dir / "source-summary.json"
    summary_log = work_dir / "source-probe.log"
    timeline_path = work_dir / "source-timeline.csv"
    timeline_log = work_dir / "source-timeline-probe.log"
    summary, summary_command = probe_video_summary(
        ffprobe, source, summary_path, summary_log)
    color = classify_video_color(summary)
    stream = summary["streams"][0]
    frame = summary["frames"][0]
    try:
        stream_width = int(stream["width"])
        stream_height = int(stream["height"])
        frame_width = int(frame["width"])
        frame_height = int(frame["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WholeClipError(
            "ffprobe did not expose exact stream/decoded-frame dimensions") from exc
    if (
        stream_width <= 0 or stream_height <= 0 or
        frame_width <= 0 or frame_height <= 0 or
        (stream_width, stream_height) != (frame_width, frame_height)
    ):
        raise WholeClipError(
            "ffprobe stream/decoded-frame dimensions are invalid or transformed: "
            f"{stream_width}x{stream_height} vs {frame_width}x{frame_height}")
    timeline, scan, timeline_command = probe_video_timeline(
        ffprobe, source, summary, timeline_path, timeline_log)
    return {
        "summary": summary,
        "summary_file": os.fspath(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "timeline_scan": scan,
        "timeline_file": os.fspath(timeline_path),
        "timeline_sha256": sha256_file(timeline_path),
        "timeline": timeline,
        "color": color,
        "width": stream_width,
        "height": stream_height,
        "commands": {
            "summary": summary_command,
            "timeline": timeline_command,
        },
    }


def validate_native_extra(options: list[str]) -> None:
    for option in options:
        name = option.split("=", 1)[0]
        if name in RESERVED_NATIVE_OPTIONS:
            raise WholeClipError(
                f"{name} is owned by the whole-clip wrapper and cannot be passed via --extra")


def query_native_capabilities(
    sunshine: Path,
    build_dir: Path,
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Fail before media decode unless the executable attests the replay contract."""
    command = [
        os.fspath(sunshine),
        "--sbs-bench",
        "--capabilities", os.fspath(output_path),
    ]
    run_logged_command(command, build_dir, log_path)
    value = read_json_object(output_path, "native SBS capabilities")
    expected = {
        ("schema",): 1,
        ("native_whole_clip", "follow_protocol_schema"): 1,
        ("native_whole_clip", "follow_global_first_sequence"): True,
        ("native_whole_clip", "adaptive_state_schema"): ADAPTIVE_TRACE_SCHEMA,
        ("native_whole_clip", "scene_cache_contract_schema"):
            SCENE_CACHE_CONTRACT_SCHEMA,
        ("native_whole_clip", "scene_cache_packed_sbs_contract"): True,
        ("native_whole_clip", "scene_cache_depth", "dtype"): "float32-le",
        ("native_whole_clip", "scene_cache_depth", "layout"): "row-major",
        ("native_whole_clip", "scene_cache_depth", "dxgi_format"):
            "R32_FLOAT",
        ("native_whole_clip", "scene_cache_depth", "dimensions"):
            "per-frame-metadata",
        ("native_whole_clip", "scene_cache_depth", "bytes_per_frame"): None,
        ("native_whole_clip", "scene_cache_frame_metadata", "schema"):
            SCENE_CACHE_METADATA_SCHEMA,
        ("native_whole_clip", "scene_cache_frame_metadata", "word_count"):
            SCENE_CACHE_METADATA_WORDS,
        (
            "native_whole_clip",
            "scene_cache_frame_metadata",
            "roi_transform_word_offset",
        ): SCENE_CACHE_METADATA_ROI_OFFSET,
        (
            "native_whole_clip",
            "scene_cache_frame_metadata",
            "roi_transform_word_count",
        ): SCENE_CACHE_ROI_WORDS,
        (
            "native_whole_clip",
            "scene_cache_frame_metadata",
            "roi_transform_contract_schema",
        ): FRAME_ROI_TRANSFORM_SCHEMA,
        ("native_whole_clip", "scene_cache_state", "schema"):
            SCENE_CACHE_STATE_SCHEMA,
        ("native_whole_clip", "scene_cache_state", "subject_word_count"):
            SCENE_CACHE_SUBJECT_WORDS,
        (
            "native_whole_clip",
            "scene_cache_state",
            "depth_frame_state_word_count",
        ): SCENE_CACHE_DEPTH_FRAME_STATE_WORDS,
        ("native_whole_clip", "scene_cache_state", "word_count"):
            SCENE_CACHE_STATE_WORDS,
        ("native_whole_clip", "scene_cache_state", "dtype"): "uint32-le",
        ("native_whole_clip", "scene_plan", "schema"): 1,
        ("native_whole_clip", "scene_plan", "version"): "scene-plan-v1",
        ("native_whole_clip", "scene_plan", "one_scene_per_replay"): True,
        ("native_whole_clip", "scene_plan", "absolute_pop_strength"): True,
        ("native_whole_clip", "scene_plan", "source_pixel_zero_anchor"): True,
        ("native_whole_clip", "render_cache_follow"): True,
        ("native_whole_clip", "render_skips_tensorrt"): True,
        (
            "native_whole_clip",
            "whole_clip_inference_attestation",
            "depth_inference_enabled",
        ): True,
        (
            "native_whole_clip",
            "whole_clip_inference_attestation",
            "scheduled_depth_update_count",
        ): True,
        (
            "native_whole_clip",
            "whole_clip_inference_attestation",
            "tensorrt_enqueue_count",
        ): True,
        ("native_whole_clip", "atomic_sbs_publication"): True,
    }
    mismatches = {}
    for path, wanted in expected.items():
        actual: Any = value
        try:
            for key in path:
                actual = actual[key]
        except (KeyError, TypeError):
            actual = None
        if actual != wanted:
            mismatches[".".join(path)] = {
                "expected": wanted,
                "actual": actual,
            }
    modes = (
        value.get("native_whole_clip", {}).get("artifact_modes")
        if isinstance(value.get("native_whole_clip"), dict) else None
    )
    formats = (
        value.get("native_whole_clip", {}).get("source_formats")
        if isinstance(value.get("native_whole_clip"), dict) else None
    )
    if not isinstance(modes, list) or not {"adaptive", "conversion"}.issubset(modes):
        mismatches["native_whole_clip.artifact_modes"] = {
            "expected": ["adaptive", "conversion"],
            "actual": modes,
        }
    if not isinstance(formats, list) or not {"png", "pfm"}.issubset(formats):
        mismatches["native_whole_clip.source_formats"] = {
            "expected": ["png", "pfm"],
            "actual": formats,
        }
    if mismatches:
        raise WholeClipError(
            "sunshine executable does not support the required bounded scene replay "
            "contract: " + json.dumps(mismatches, sort_keys=True))
    return {
        "value": value,
        "file": os.fspath(output_path),
        "sha256": sha256_file(output_path),
        "command": command,
    }


def require_new_or_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise WholeClipError(f"output is not a directory: {path}")
        try:
            next(path.iterdir())
        except StopIteration:
            return
        raise WholeClipError(f"output directory must be new or empty: {path}")
    path.mkdir(parents=True)


def inspect_video_metadata(path: Path) -> dict[str, Any]:
    """Read the imageio-ffmpeg header without decoding the clip twice in Python."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise WholeClipError(
            "video input requires imageio-ffmpeg "
            "(python -m pip install imageio-ffmpeg)") from exc

    reader = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24")
    try:
        metadata = next(reader)
    except Exception as exc:
        raise WholeClipError(f"cannot inspect video metadata: {exc}") from exc
    finally:
        reader.close()
    return _json_ready(metadata)


def reject_unsupported_video(metadata: dict[str, Any]) -> None:
    rotation = int(metadata.get("rotate") or 0) % 360
    source_size = metadata.get("source_size")
    decoded_size = metadata.get("size")
    implicit_transform = (
        isinstance(source_size, list) and isinstance(decoded_size, list) and
        source_size != decoded_size
    )
    if rotation or implicit_transform:
        raise WholeClipError(
            "rotated/implicitly transformed video is not supported yet "
            f"(rotation={rotation}, source={source_size}, decoded={decoded_size}); "
            "normalize rotation explicitly before conversion")

    pixel_description = str(metadata.get("pix_fmt") or "")
    lowered = pixel_description.lower()
    hdr_transfer = any(token in lowered for token in (
        "smpte2084", "smpte 2084", "arib-std-b67", "arib std b67", "hlg", "pq",
    ))
    high_bit_depth = bool(_HIGH_BIT_DEPTH_RE.search(lowered))
    if hdr_transfer or high_bit_depth:
        raise WholeClipError(
            "HDR or high-bit-depth video is not supported by the SDR PNG ingest path "
            f"(detected {pixel_description!r})")


def run_logged_command(command: list[str], cwd: Path | None, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as log:
        log.write("$ " + command_display(command) + "\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=os.fspath(cwd) if cwd else None,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            raise WholeClipError(f"cannot start {command[0]}: {exc}") from exc
    if result.returncode:
        raise WholeClipError(
            f"command exited {result.returncode}; diagnostics: {log_path}")


def parse_showinfo(
    log: str | Iterable[str],
) -> tuple[Fraction, list[dict[str, Any]]]:
    """Parse the exact integer PTS clock and presentation-ordered frame rows."""
    time_base: Fraction | None = None
    rows: list[dict[str, Any]] = []
    lines = log.splitlines() if isinstance(log, str) else log
    for line in lines:
        if "showinfo" not in line:
            continue
        if time_base is None:
            match = _SHOWINFO_TIME_BASE_RE.search(line)
            if match:
                numerator = int(match.group("num"))
                denominator = int(match.group("den"))
                if numerator <= 0 or denominator <= 0:
                    raise WholeClipError("showinfo reported an invalid time base")
                time_base = Fraction(numerator, denominator)
        frame_match = _SHOWINFO_FRAME_RE.search(line)
        if not frame_match:
            continue
        duration_match = _SHOWINFO_DURATION_RE.search(line)
        duration = None
        duration_time_text = None
        if duration_match and duration_match.group("duration") != "N/A":
            duration = int(duration_match.group("duration"))
            duration_time_text = duration_match.group("duration_time")
        rows.append({
            "n": int(frame_match.group("n")),
            "pts": int(frame_match.group("pts")),
            "pts_time_text": frame_match.group("pts_time"),
            "duration": duration,
            "duration_time_text": duration_time_text,
        })

    if time_base is None:
        raise WholeClipError("FFmpeg showinfo did not report the video time base")
    if not rows:
        raise WholeClipError("FFmpeg decoded no presentation frames")
    for index, row in enumerate(rows):
        if row["n"] != index:
            raise WholeClipError(
                f"showinfo frame sequence is incomplete at {index}: got n={row['n']}")
        if index and row["pts"] <= rows[index - 1]["pts"]:
            raise WholeClipError(
                "video PTS must be strictly increasing in presentation order "
                f"(frame {index}: {rows[index - 1]['pts']} -> {row['pts']})")
    return time_base, rows


def parse_showinfo_file(path: Path) -> tuple[Fraction, list[dict[str, Any]]]:
    """Stream a potentially multi-gigabyte FFmpeg log instead of loading it at once."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return parse_showinfo(stream)
    except OSError as exc:
        raise WholeClipError(f"cannot read FFmpeg log {path}: {exc}") from exc


def _duration_ticks(
    rows: list[dict[str, Any]], index: int, metadata_fps: float | None,
    time_base: Fraction,
) -> int:
    if index + 1 < len(rows):
        return rows[index + 1]["pts"] - rows[index]["pts"]
    reported = rows[index].get("duration")
    if isinstance(reported, int) and reported > 0:
        return reported
    deltas = [
        rows[item + 1]["pts"] - rows[item]["pts"]
        for item in range(len(rows) - 1)
    ]
    if deltas:
        return max(1, int(round(median(deltas))))
    if metadata_fps and metadata_fps > 0:
        return max(1, int(round(float(Fraction(1, 1) / time_base) / metadata_fps)))
    raise WholeClipError("cannot determine the single frame's presentation duration")


def build_video_timeline(
    time_base: Fraction,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata_fps = float(metadata.get("fps") or 0.0) or None
    frames: list[dict[str, Any]] = []
    deltas: list[int] = []
    for index, row in enumerate(rows):
        duration = _duration_ticks(rows, index, metadata_fps, time_base)
        if duration <= 0:
            raise WholeClipError(f"frame {index} has a non-positive duration")
        if index + 1 < len(rows):
            deltas.append(duration)
        pts_seconds = float(row["pts"] * time_base)
        duration_seconds = float(duration * time_base)
        frames.append({
            "index": index,
            "frame_id": f"{index + 1:010d}",
            "pts": row["pts"],
            "pts_time": pts_seconds,
            "pts_time_text": row["pts_time_text"],
            "duration": duration,
            "duration_time": duration_seconds,
        })
    return {
        "schema": 1,
        "clock": "source-video-presentation",
        "time_base": {
            "num": time_base.numerator,
            "den": time_base.denominator,
        },
        "first_pts": frames[0]["pts"],
        "first_pts_time": frames[0]["pts_time"],
        "nominal_fps": metadata_fps,
        "variable_frame_rate": len(set(deltas)) > 1,
        "frame_count": len(frames),
        "frames": frames,
    }


def _source_frame_id(path: Path) -> str | None:
    match = _FRAME_ID_RE.search(path.stem)
    return match.group(1) if match else None


def _frame_sort_key(path: Path) -> tuple[Any, ...]:
    frame_id = _source_frame_id(path)
    if frame_id is None:
        return (1, path.name)
    stripped = frame_id.lstrip("0") or "0"
    return (0, len(stripped), stripped, path.name)


def frame_directory_files(path: Path) -> list[Path]:
    frames = sorted(
        (item for item in path.iterdir()
         if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=_frame_sort_key,
    )
    if not frames:
        raise WholeClipError(f"frame directory contains no PNG/JPEG images: {path}")
    return frames


def build_frame_directory_timeline(frames: list[Path], fps: float) -> dict[str, Any]:
    if not math.isfinite(fps) or fps <= 0:
        raise WholeClipError("--fps must be a positive finite number")
    fps_fraction = Fraction(str(fps)).limit_denominator(1_001_000)
    time_base = Fraction(fps_fraction.denominator, fps_fraction.numerator)
    ids: set[str] = set()
    rows = []
    for index, path in enumerate(frames):
        frame_id = _source_frame_id(path) or f"{index:05d}"
        if frame_id in ids:
            raise WholeClipError(f"duplicate source frame identity {frame_id!r}")
        ids.add(frame_id)
        rows.append({
            "index": index,
            "frame_id": frame_id,
            "pts": index,
            "pts_time": float(index * time_base),
            "pts_time_text": str(float(index * time_base)),
            "duration": 1,
            "duration_time": float(time_base),
            "source_file": path.name,
        })
    return {
        "schema": 1,
        "clock": "synthetic-cfr",
        "time_base": {
            "num": time_base.numerator,
            "den": time_base.denominator,
        },
        "first_pts": 0,
        "first_pts_time": 0.0,
        "nominal_fps": float(fps_fraction),
        "variable_frame_rate": False,
        "frame_count": len(rows),
        "frames": rows,
    }


def frame_directory_fps(path: Path, explicit_fps: float | None) -> float:
    """Resolve a frame clock without silently inventing one."""
    if explicit_fps is not None:
        return explicit_fps
    metadata_path = path / "meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WholeClipError(
            "frame-directory input requires --fps unless meta.json supplies fps") from exc
    except (OSError, ValueError) as exc:
        raise WholeClipError(f"cannot read frame-directory meta.json: {exc}") from exc
    if not isinstance(metadata, dict):
        raise WholeClipError("frame-directory meta.json must contain an object")
    value = metadata.get("fps", metadata.get("frame_rate"))
    if isinstance(value, str) and "/" in value:
        try:
            numerator, denominator = value.split("/", 1)
            value = float(Fraction(int(numerator), int(denominator)))
        except (ValueError, ZeroDivisionError) as exc:
            raise WholeClipError(
                f"frame-directory meta.json has invalid fps {value!r}") from exc
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WholeClipError(
            "frame-directory input requires --fps or numeric fps/frame_rate in meta.json")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise WholeClipError("frame-directory FPS must be a positive finite number")
    return fps


def frame_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise WholeClipError(
            "frame input requires Pillow to inspect spool dimensions") from exc
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise WholeClipError(f"cannot inspect frame dimensions for {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise WholeClipError(f"frame has invalid dimensions: {path}")
    return int(width), int(height)


def video_dimensions(metadata: dict[str, Any]) -> tuple[int, int]:
    value = metadata.get("source_size") or metadata.get("size")
    if (
        not isinstance(value, list) or len(value) != 2 or
        not all(isinstance(item, (int, float)) for item in value)
    ):
        raise WholeClipError("video metadata does not provide usable source dimensions")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise WholeClipError("video metadata reports invalid source dimensions")
    return width, height


def estimated_video_frame_count(metadata: dict[str, Any]) -> int:
    try:
        fps = float(metadata.get("fps"))
        duration = float(metadata.get("duration"))
    except (TypeError, ValueError) as exc:
        raise WholeClipError(
            "video metadata lacks finite FPS/duration for disk preflight") from exc
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise WholeClipError(
            "video metadata lacks positive finite FPS/duration for disk preflight")
    # Two guard frames cover duration rounding and a final partial presentation interval.
    return max(1, int(math.ceil(fps * duration)) + 2)


def _native_number(
    options: list[str],
    name: str,
    default: float,
) -> float:
    value = default
    index = 0
    while index < len(options):
        option = options[index]
        if option == name:
            if index + 1 >= len(options):
                raise WholeClipError(f"{name} requires a value")
            candidate = options[index + 1]
            index += 2
        elif option.startswith(name + "="):
            candidate = option.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        try:
            value = float(candidate)
        except ValueError as exc:
            raise WholeClipError(f"{name} has invalid numeric value {candidate!r}") from exc
    return value


def estimated_sbs_dimensions(
    source_width: int,
    source_height: int,
    native_extra: list[str],
) -> tuple[int, int]:
    scale = _native_number(native_extra, "--output-scale", 1.0)
    eye_width_option = _native_number(native_extra, "--eye-w", 0.0)
    eye_height_option = _native_number(native_extra, "--eye-h", 0.0)
    if not math.isfinite(scale) or scale <= 0:
        raise WholeClipError("--output-scale must be a positive finite number")
    if (
        not math.isfinite(eye_width_option) or eye_width_option < 0 or
        not math.isfinite(eye_height_option) or eye_height_option < 0
    ):
        raise WholeClipError("--eye-w/--eye-h must be non-negative finite numbers")
    eye_width = int(eye_width_option)
    eye_height = int(eye_height_option)
    if eye_width_option != eye_width or eye_height_option != eye_height:
        raise WholeClipError("--eye-w/--eye-h must be integers")
    aspect = source_width / source_height
    if eye_width > 0 and eye_height > 0:
        pass
    elif eye_height > 0:
        eye_width = max(1, int(round(eye_height * aspect)))
    elif eye_width > 0:
        eye_height = max(1, int(round(eye_width / aspect)))
    else:
        eye_width = max(1, int(round(source_width * scale)))
        eye_height = max(1, int(round(source_height * scale)))
    return 2 * eye_width, eye_height


def disk_spool_preflight(
    output_dir: Path,
    *,
    frame_count: int,
    source_width: int,
    source_height: int,
    source_spooled: bool,
    conversion: bool,
    native_extra: list[str],
    free_bytes: int | None = None,
) -> dict[str, Any]:
    if frame_count <= 0:
        raise WholeClipError("disk preflight requires a positive estimated frame count")
    source_frame_bytes = source_width * source_height * 4
    source_spool_bytes = source_frame_bytes * frame_count if source_spooled else 0
    sbs_width = 0
    sbs_height = 0
    sbs_spool_bytes = 0
    if conversion:
        sbs_width, sbs_height = estimated_sbs_dimensions(
            source_width, source_height, native_extra)
        sbs_spool_bytes = sbs_width * sbs_height * 4 * frame_count
    estimated_spool_bytes = source_spool_bytes + sbs_spool_bytes
    guarded_spool_bytes = (
        estimated_spool_bytes * SPOOL_SAFETY_FACTOR.numerator +
        SPOOL_SAFETY_FACTOR.denominator - 1
    ) // SPOOL_SAFETY_FACTOR.denominator
    required_free_bytes = guarded_spool_bytes + SPOOL_FIXED_RESERVE_BYTES
    if free_bytes is None:
        free_bytes = shutil.disk_usage(output_dir).free
    result = {
        "passed": free_bytes >= required_free_bytes,
        "filesystem_path": os.fspath(output_dir),
        "free_bytes": free_bytes,
        "estimated_frame_count": frame_count,
        "source": {
            "width": source_width,
            "height": source_height,
            "bytes_per_pixel": 4,
            "spooled": source_spooled,
            "estimated_spool_bytes": source_spool_bytes,
        },
        "sbs": {
            "enabled": conversion,
            "width": sbs_width,
            "height": sbs_height,
            "bytes_per_pixel": 4,
            "estimated_spool_bytes": sbs_spool_bytes,
        },
        "estimated_spool_bytes": estimated_spool_bytes,
        "safety_factor": float(SPOOL_SAFETY_FACTOR),
        "fixed_reserve_bytes": SPOOL_FIXED_RESERVE_BYTES,
        "required_free_bytes": required_free_bytes,
    }
    if not result["passed"]:
        raise DiskSpaceError(
            "insufficient free space for whole-clip spool: "
            f"need {required_free_bytes} bytes, have {free_bytes} bytes",
            result,
        )
    return result


def streaming_disk_preflight(
    output_dir: Path,
    *,
    cache_max_bytes: int,
    conversion: bool,
    source_size_bytes: int = 0,
) -> dict[str, Any]:
    """Reserve the explicit cache cap plus output/log headroom; no whole-frame spool."""
    usage = shutil.disk_usage(output_dir)
    output_reserve = (
        max(1024 * 1024 * 1024, source_size_bytes * 2)
        if conversion else 0
    )
    required = (
        cache_max_bytes if conversion else 0
    ) + SPOOL_FIXED_RESERVE_BYTES + output_reserve
    result = {
        "strategy": "bounded-streaming-scene-cache",
        "passed": usage.free >= required,
        "free_bytes": usage.free,
        "cache_hard_cap_bytes": cache_max_bytes if conversion else 0,
        "fixed_log_work_reserve_bytes": SPOOL_FIXED_RESERVE_BYTES,
        "encoded_output_reserve_bytes": output_reserve,
        "required_free_bytes": required,
        "whole_source_frame_spool": False,
        "whole_sbs_frame_spool": False,
    }
    if not result["passed"]:
        raise DiskSpaceError(
            "insufficient free space for bounded whole-clip conversion: "
            f"need {required} bytes, have {usage.free} bytes",
            result,
        )
    return result


def decode_video(
    ffmpeg: str,
    source: Path,
    frames_dir: Path,
    log_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "frame_%010d.png"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "info",
        "-nostdin",
        "-copyts",
        "-i", os.fspath(source),
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", "showinfo",
        "-fps_mode", "passthrough",
        "-start_number", "1",
        "-c:v", "png",
        os.fspath(pattern),
    ]
    run_logged_command(command, None, log_path)
    time_base, rows = parse_showinfo_file(log_path)
    timeline = build_video_timeline(time_base, rows, metadata)
    files = frame_directory_files(frames_dir)
    expected_names = [
        f"frame_{index + 1:010d}.png" for index in range(len(rows))
    ]
    if [path.name for path in files] != expected_names:
        raise WholeClipError(
            "decoded PNG sequence does not exactly match showinfo presentation frames")
    return timeline, command


def validate_native_outputs(
    artifacts_dir: Path,
    timeline: dict[str, Any],
    artifact_mode: str,
) -> dict[str, Any]:
    trace_path = artifacts_dir / TRACE_NAME
    contract_path = artifacts_dir / NATIVE_CONTRACT_NAME
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise WholeClipError(f"native harness did not write {TRACE_NAME}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WholeClipError(
            f"native harness did not write a valid {NATIVE_CONTRACT_NAME}: {exc}") from exc
    if contract.get("schema") != 1:
        raise WholeClipError(
            f"unsupported whole-clip contract schema: {contract.get('schema')!r}")
    if contract.get("artifact_mode") != artifact_mode:
        raise WholeClipError(
            "whole-clip contract artifact mode mismatch: "
            f"{contract.get('artifact_mode')!r} != {artifact_mode!r}")
    expected_count = timeline["frame_count"]
    if contract.get("source_frame_count") != expected_count:
        raise WholeClipError(
            "whole-clip contract source count mismatch: "
            f"{contract.get('source_frame_count')!r} != {expected_count}")
    _validate_native_inference_attestation(
        contract,
        replay=False,
        source_frame_count=expected_count,
    )
    adaptive_contract = contract.get("adaptive_state")
    if not isinstance(adaptive_contract, dict):
        raise WholeClipError("whole-clip contract lacks the adaptive-state descriptor")
    if (
        adaptive_contract.get("file") != TRACE_NAME or
        adaptive_contract.get("schema") != ADAPTIVE_TRACE_SCHEMA or
        adaptive_contract.get("frame_count") != expected_count
    ):
        raise WholeClipError("whole-clip adaptive-state contract mismatch")
    sbs_contract = contract.get("sbs")
    if not isinstance(sbs_contract, dict):
        raise WholeClipError("whole-clip contract lacks the SBS descriptor")
    if artifact_mode == "conversion":
        expected_ids = [row["frame_id"] for row in timeline["frames"]]
        actual_ids = [
            path.stem[len("sbs_"):] for path in artifacts_dir.glob("sbs_*.png")
        ]
        if (
            len(actual_ids) != len(expected_ids) or
            len(set(actual_ids)) != len(actual_ids) or
            set(actual_ids) != set(expected_ids)
        ):
            raise WholeClipError(
                "conversion SBS frame identities do not match the source timeline")
        if not sbs_contract.get("enabled"):
            raise WholeClipError("whole-clip contract does not attest SBS conversion output")
        if sbs_contract.get("file_pattern") != "sbs_<frame-id>.png":
            raise WholeClipError("whole-clip contract SBS filename pattern mismatch")
        if sbs_contract.get("frame_count") != expected_count:
            raise WholeClipError("whole-clip contract SBS frame count mismatch")
    elif sbs_contract.get("enabled") or sbs_contract.get("frame_count") != 0:
        raise WholeClipError("adaptive-only contract unexpectedly attests SBS frames")
    return contract


def _validate_native_inference_attestation(
    contract: dict[str, Any],
    *,
    replay: bool,
    source_frame_count: int,
) -> None:
    if (
        not isinstance(source_frame_count, int) or
        isinstance(source_frame_count, bool) or
        source_frame_count <= 0
    ):
        raise WholeClipError(
            "native inference attestation requires a positive source count")
    if replay:
        expected = {
            "inference_mode": "scene-cache-replay",
            "depth_inference_enabled": False,
            "scheduled_depth_update_count": 0,
            "tensorrt_enqueue_count": 0,
        }
    else:
        depth_reuse_interval = contract.get("depth_reuse_interval")
        if (
            not isinstance(depth_reuse_interval, int) or
            isinstance(depth_reuse_interval, bool) or
            not 1 <= depth_reuse_interval <= 8
        ):
            raise WholeClipError(
                "native analysis inference attestation has an invalid depth reuse interval")
        scheduled_count = (
            source_frame_count + depth_reuse_interval - 1
        ) // depth_reuse_interval
        expected = {
            "inference_mode": "single-pass-tensorrt",
            "depth_inference_enabled": True,
            "scheduled_depth_update_count": scheduled_count,
            "tensorrt_enqueue_count": scheduled_count,
        }
    mismatches = {
        key: {"expected": wanted, "actual": contract.get(key)}
        for key, wanted in expected.items()
        if contract.get(key) != wanted
    }
    if mismatches:
        mode = "scene replay" if replay else "analysis"
        raise WholeClipError(
            f"native {mode} inference attestation mismatch: " +
            json.dumps(mismatches, sort_keys=True))


def _concat_quote(path: Path) -> str:
    # FFmpeg's concat demuxer uses shell-like single-quote escaping on every platform.
    return path.resolve().as_posix().replace("'", "'\\''")


def write_concat_file(path: Path, frames: list[Path], timeline: dict[str, Any]) -> None:
    if len(frames) != timeline["frame_count"]:
        raise WholeClipError("cannot mux: SBS frame count differs from timeline")
    time_base = timeline_time_base(timeline)
    frame_rate = Fraction(time_base.denominator, time_base.numerator)
    lines = ["ffconcat version 1.0"]
    for frame, timing in zip(frames, timeline["frames"]):
        duration = float(timing["duration_time"])
        if not math.isfinite(duration) or duration <= 0:
            raise WholeClipError("cannot mux a non-positive frame duration")
        lines.append(f"file '{_concat_quote(frame)}'")
        # image2 defaults to 25 FPS for every subfile.  A per-file option is required:
        # a global `-r` would synthesize/drop presentation frames and destroy VFR timing.
        lines.append(
            f"option framerate {frame_rate.numerator}/{frame_rate.denominator}")
        lines.append(f"duration {duration:.12f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_http_concat_file(
    path: Path,
    bridge: SbsFrameHttpBridge,
    timeline: dict[str, Any],
) -> None:
    """Write the complete exact timeline while leaving pixels demand-driven."""
    if timeline.get("frame_count") != bridge.frame_count:
        raise WholeClipError("HTTP SBS bridge frame count differs from timeline")
    time_base = timeline_time_base(timeline)
    frame_rate = Fraction(time_base.denominator, time_base.numerator)
    lines = ["ffconcat version 1.0"]
    for sequence, timing in enumerate(timeline["frames"], 1):
        duration = float(timing["duration_time"])
        if not math.isfinite(duration) or duration <= 0:
            raise WholeClipError("cannot stream a non-positive frame duration")
        url = bridge.frame_url(sequence)
        if "'" in url or "\n" in url or "\r" in url:
            raise WholeClipError("HTTP SBS bridge produced an unsafe URL")
        lines.append(f"file '{url}'")
        lines.append(
            f"option framerate {frame_rate.numerator}/{frame_rate.denominator}")
        lines.append(f"duration {duration:.12f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def codec_arguments(codec: str, *, hdr: bool = False) -> list[str]:
    pixel_format = "p010le" if hdr else "yuv420p"
    if codec == "hevc_nvenc":
        result = [
            "-c:v", codec, "-preset", "p5", "-tune", "hq",
            "-rc", "vbr", "-cq", "18", "-b:v", "0", "-bf", "0",
            "-pix_fmt", pixel_format,
        ]
        if hdr:
            result += ["-profile:v", "main10", "-extra_sei", "1"]
        return result
    if codec == "av1_nvenc":
        result = [
            "-c:v", codec, "-preset", "p5", "-tune", "hq",
            "-rc", "vbr", "-cq", "20", "-b:v", "0", "-bf", "0",
            "-pix_fmt", pixel_format,
        ]
        if hdr:
            result += ["-extra_sei", "1"]
        return result
    if codec == "libx265":
        # Disabling B-frames is deliberate. x265 and NVENC otherwise reorder the short,
        # potentially VFR image sequence in a way that can change PTS and force a muxer origin
        # shift. Offline conversion values timing fidelity over the modest compression gain.
        return [
            "-c:v", codec, "-preset", "medium", "-crf", "18", "-bf", "0",
            "-pix_fmt", "yuv420p10le" if hdr else "yuv420p",
        ]
    raise WholeClipError(f"unsupported codec: {codec}")


def _container_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in ("", ".mkv"):
        return "matroska"
    if suffix in (".mp4", ".m4v"):
        return "mp4"
    raise WholeClipError(
        f"unsupported SBS video container {suffix!r}; use .mkv or .mp4")


def timeline_time_base(timeline: dict[str, Any]) -> Fraction:
    value = timeline.get("time_base")
    if not isinstance(value, dict):
        raise WholeClipError("timeline lacks an exact time base")
    try:
        time_base = Fraction(int(value["num"]), int(value["den"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise WholeClipError("timeline has an invalid exact time base") from exc
    if time_base <= 0:
        raise WholeClipError("timeline time base must be positive")
    return time_base


def validate_output_target(
    output_video: Path,
    codec: str,
    work_dir: Path,
) -> str:
    """Fail before native conversion when the requested deliverable is impossible."""
    if codec not in CODECS:
        raise WholeClipError(f"unsupported codec: {codec}")
    container = _container_for(output_video)
    if output_video.exists():
        raise WholeClipError(f"refusing to overwrite SBS video: {output_video}")
    try:
        output_video.relative_to(work_dir)
    except ValueError:
        pass
    else:
        raise WholeClipError(
            "SBS video cannot be placed inside <out>/work because normal cleanup "
            "would delete the deliverable")
    return container


def _last_duration_bsf(timeline: dict[str, Any]) -> str:
    last = timeline["frames"][-1]
    # `-enc_time_base demux` keeps these integer values in the timeline clock.  The concat
    # demuxer cannot express the final packet duration without repeating a frame, so set only
    # that packet's duration after encoding.  PTS matching remains correct with B-frames.
    return (
        "setts=duration="
        f"if(eq(PTS\\,{int(last['pts'])})\\,{int(last['duration'])}\\,DURATION)"
    )


def _zscale_range(value: str) -> str:
    mapping = {
        "tv": "limited",
        "mpeg": "limited",
        "limited": "limited",
        "pc": "full",
        "jpeg": "full",
        "full": "full",
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise WholeClipError(f"unsupported ffprobe color range for zscale: {value!r}") from exc


def hdr_decode_filter(color: dict[str, Any]) -> str:
    if color.get("mode") not in ("hdr-pq", "hdr-hlg"):
        raise WholeClipError("HDR decode filter requires a probed PQ or HLG source")
    return (
        "zscale="
        f"rangein={_zscale_range(str(color['color_range']))}:"
        f"primariesin={color['color_primaries']}:"
        f"transferin={color['color_transfer']}:"
        f"matrixin={color['color_space']}:"
        "range=full:primaries=bt709:transfer=linear:matrix=gbr:npl=80,"
        "format=gbrpf32le"
    )


def hdr_encode_filter(
    color: dict[str, Any],
    first_pts_time: float,
    codec: str,
    sbs_width: int,
    sbs_height: int,
) -> str:
    if sbs_width <= 0 or sbs_height <= 0:
        raise WholeClipError("HDR encoder requires positive native SBS dimensions")
    output_pixel_format = "p010le" if codec in ("hevc_nvenc", "av1_nvenc") else "yuv420p10le"
    return (
        f"[0:v:0]setpts=PTS-STARTPTS,{hdr_decode_filter(color)},"
        f"scale={sbs_width}:{sbs_height}:flags=bilinear[donor];"
        "[1:v:0]setpts=PTS-STARTPTS,format=gbrpf32le[sbs];"
        "[donor][sbs]blend=all_expr=B:shortest=1,"
        "zscale="
        "rangein=full:primariesin=bt709:transferin=linear:matrixin=gbr:"
        f"range={_zscale_range(str(color['color_range']))}:"
        f"primaries={color['color_primaries']}:"
        f"transfer={color['color_transfer']}:"
        f"matrix={color['color_space']}:npl=80,"
        f"format={output_pixel_format},"
        f"setpts=PTS+{first_pts_time:.12f}/TB[vout]"
    )


def build_http_encoder_command(
    ffmpeg: str,
    concat_path: Path,
    timeline: dict[str, Any],
    source_video: Path | None,
    output_video: Path,
    codec: str,
    color: dict[str, Any],
    sbs_dimensions: tuple[int, int] | None = None,
) -> tuple[list[str], str | None]:
    """Build one persistent exact-timeline encoder fed by the loopback SBS bridge."""
    container = _container_for(output_video)
    hdr = color.get("mode") in ("hdr-pq", "hdr-hlg")
    if hdr and source_video is None:
        raise WholeClipError("HDR conversion requires the original video as metadata donor")
    first_pts_time = float(
        int(timeline["first_pts"]) * timeline_time_base(timeline))
    command = [
        ffmpeg,
        "-hide_banner", "-loglevel", "warning", "-xerror", "-nostdin", "-n",
        "-copyts",
    ]
    concat_input_index = 0
    if source_video is not None:
        command += ["-i", os.fspath(source_video)]
        concat_input_index = 1
    if not hdr:
        command += ["-itsoffset", f"{first_pts_time:.12f}"]
    command += [
        "-protocol_whitelist", "file,http,tcp",
        "-f", "concat", "-safe", "0", "-i", os.fspath(concat_path),
    ]
    filter_graph = None
    if hdr:
        if sbs_dimensions is None:
            raise WholeClipError(
                "HDR persistent encoder requires native SBS dimensions")
        filter_graph = hdr_encode_filter(
            color,
            first_pts_time,
            codec,
            int(sbs_dimensions[0]),
            int(sbs_dimensions[1]),
        )
        command += ["-filter_complex", filter_graph, "-map", "[vout]"]
    else:
        command += ["-map", f"{concat_input_index}:v:0"]
    if source_video is None:
        command += ["-an"]
    else:
        command += [
            "-map", "0:a?",
            "-map_metadata", "0",
            "-map_chapters", "0",
            "-c:a", "copy",
        ]
    command += [
        "-fps_mode", "passthrough",
        "-enc_time_base", "demux",
        *codec_arguments(codec, hdr=hdr),
        "-bsf:v", _last_duration_bsf(timeline),
        "-avoid_negative_ts", "disabled",
    ]
    if hdr:
        command += [
            "-color_range", str(color["color_range"]),
            "-colorspace", str(color["color_space"]),
            "-color_trc", str(color["color_transfer"]),
            "-color_primaries", str(color["color_primaries"]),
        ]
    if container == "mp4":
        if codec in ("hevc_nvenc", "libx265"):
            command += ["-tag:v", "hvc1"]
        command += ["-movflags", "+faststart", "-f", "mp4"]
    else:
        command += ["-f", "matroska"]
    command.append(os.fspath(output_video))
    return command, filter_graph


def render_cached_scene(
    *,
    sunshine: Path,
    conf: Path,
    build_dir: Path,
    cache_dir: Path,
    scene: dict[str, Any],
    render_producer: Any,
    bridge: SbsFrameHttpBridge,
    encoder: LoggedSubprocess,
    work_dir: Path,
    plans_dir: Path,
    timeout_seconds: float,
    expected_sbs_dimensions: tuple[int, int],
) -> dict[str, Any]:
    """Replay one immutable finalized scene without TensorRT and stream it to FFmpeg."""
    try:
        import scene_plan as scene_policy
    except ModuleNotFoundError as exc:
        raise WholeClipError("scene replay requires scene_plan.py") from exc
    scene_id = scene.get("scene_id")
    if not isinstance(scene_id, int) or isinstance(scene_id, bool) or scene_id <= 0:
        raise WholeClipError("finalized scene has no positive scene_id")
    start = scene.get("start_sequence")
    end = scene.get("end_sequence_exclusive")
    if (
        not isinstance(start, int) or isinstance(start, bool) or
        not isinstance(end, int) or isinstance(end, bool) or
        start <= 0 or end <= start
    ):
        raise WholeClipError("finalized scene has an invalid global sequence range")
    count = end - start
    extension = render_producer.extension
    if extension not in ("png", "pfm"):
        raise WholeClipError("render producer has an unsupported native follow format")

    input_dir = work_dir / "render-input" / f"scene_{scene_id:08d}"
    native_out = work_dir / "render-output" / f"scene_{scene_id:08d}"
    input_dir.mkdir(parents=True)
    native_out.mkdir(parents=True)
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"scene_{scene_id:08d}.json"
    try:
        plan_document = scene_policy.native_scene_plan_document(scene)
    except scene_policy.ScenePlanError as exc:
        raise WholeClipError(f"cannot serialize finalized scene {scene_id}: {exc}") from exc
    write_json_atomic(plan_path, plan_document)

    command = [
        os.fspath(sunshine),
        os.fspath(conf),
        "--sbs-bench",
        "--frames", os.fspath(input_dir),
        "--follow",
        "--follow-format", extension,
        "--follow-count", str(count),
        "--out", os.fspath(native_out),
        "--artifacts", "conversion",
        "--render-cache", os.fspath(cache_dir),
        "--scene-plan", os.fspath(plan_path),
    ]
    child = LoggedSubprocess(
        command, build_dir, work_dir / "logs" / f"render-{scene_id:08d}.log")
    terminal_published = False
    published_paths: list[Path] = []
    try:
        for sequence in range(start, end):
            input_path = render_producer.publish_next(input_dir, sequence)
            progress = wait_for_progress(
                native_out / "follow_progress.json",
                "processed_count",
                sequence - start + 1,
                child,
                timeout_seconds=timeout_seconds,
                expected={
                    "schema": 1,
                    "input_format": extension,
                    "artifact_mode": "conversion",
                    "first_sequence": start,
                },
            )
            if progress.get("last_completed_sequence") != sequence:
                raise WholeClipError(
                    "native replay progress global sequence mismatch: "
                    f"{progress.get('last_completed_sequence')!r} != {sequence}")
            sbs_path = native_out / f"sbs_{sequence:010d}.{extension}"
            if not sbs_path.is_file() or sbs_path.stat().st_size <= 0:
                raise WholeClipError(
                    f"native replay ACK lacks atomic SBS frame {sbs_path}")
            actual_dimensions = published_frame_dimensions(
                sbs_path, extension)
            if actual_dimensions != expected_sbs_dimensions:
                raise WholeClipError(
                    "native replay SBS dimensions differ from the running cache "
                    f"contract at frame {sequence}: "
                    f"{actual_dimensions} != {expected_sbs_dimensions}")
            bridge.publish(sequence, sbs_path)
            published_paths.append(sbs_path)
            bridge.wait_for_served(
                sequence, encoder, timeout_seconds=timeout_seconds)
            input_path.unlink()

        publish_producer_terminal(input_dir, frame_count=count)
        terminal_published = True
        child.wait(timeout=timeout_seconds, description=f"scene {scene_id} replay")
        terminal = read_json_object(
            native_out / "follow_progress.json", "terminal replay progress")
        expected_terminal = {
            "schema": 1,
            "status": "complete",
            "input_format": extension,
            "artifact_mode": "conversion",
            "processed_count": count,
            "first_sequence": start,
            "last_completed_sequence": end - 1,
            "source_frame_count": count,
            "sbs_frame_count": count,
            "producer_frame_count": count,
        }
        mismatches = {
            key: {"expected": wanted, "actual": terminal.get(key)}
            for key, wanted in expected_terminal.items()
            if terminal.get(key) != wanted
        }
        if mismatches:
            raise WholeClipError(
                "terminal native scene replay contract mismatch: " +
                json.dumps(mismatches, sort_keys=True))
        native_contract_path = native_out / NATIVE_CONTRACT_NAME
        native_contract = read_json_object(
            native_contract_path, "native scene replay contract")
        contract_expected = {
            "schema": 1,
            "artifact_mode": "conversion",
            "source_frame_count": count,
            "source_first_sequence": start,
        }
        contract_mismatches = {
            key: {"expected": wanted, "actual": native_contract.get(key)}
            for key, wanted in contract_expected.items()
            if native_contract.get(key) != wanted
        }
        try:
            _validate_native_inference_attestation(
                native_contract,
                replay=True,
                source_frame_count=count,
            )
        except WholeClipError as exc:
            contract_mismatches["inference_attestation"] = {
                "expected": "zero-inference exact scene-cache replay",
                "actual": str(exc),
            }
        resolved_runtime = native_contract.get("resolved_runtime")
        expected_runtime = {
            "scene_cache_replay": True,
            "scene_plan_schema": 1,
            "scene_plan_version": "scene-plan-v1",
            "scene_start_sequence": start,
            "scene_end_sequence_exclusive": end,
        }
        if not isinstance(resolved_runtime, dict):
            contract_mismatches["resolved_runtime"] = {
                "expected": expected_runtime,
                "actual": resolved_runtime,
            }
        else:
            for key, wanted in expected_runtime.items():
                if resolved_runtime.get(key) != wanted:
                    contract_mismatches[f"resolved_runtime.{key}"] = {
                        "expected": wanted,
                        "actual": resolved_runtime.get(key),
                    }
        sbs = native_contract.get("sbs")
        if (
            not isinstance(sbs, dict) or
            sbs.get("enabled") is not True or
            sbs.get("frame_count") != count or
            sbs.get("file_pattern") !=
                f"sbs_<frame-id>.{extension}" or
            (sbs.get("width"), sbs.get("height")) !=
                expected_sbs_dimensions
        ):
            contract_mismatches["sbs"] = {
                "expected": {
                    "enabled": True,
                    "frame_count": count,
                    "file_pattern": f"sbs_<frame-id>.{extension}",
                    "width": expected_sbs_dimensions[0],
                    "height": expected_sbs_dimensions[1],
                },
                "actual": sbs,
            }
        if contract_mismatches:
            raise WholeClipError(
                "native scene replay output contract mismatch: " +
                json.dumps(contract_mismatches, sort_keys=True))
        return {
            "scene_id": scene_id,
            "start_sequence": start,
            "end_sequence_exclusive": end,
            "frame_count": count,
            "plan": {
                "file": os.fspath(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "command": command,
            "native_contract": {
                "file": os.fspath(native_contract_path),
                "sha256": sha256_file(native_contract_path),
                "value": native_contract,
            },
            "follow_progress": terminal,
            "sbs_paths": published_paths,
        }
    except Exception as exc:
        if not terminal_published:
            try:
                publish_producer_terminal(input_dir, error=str(exc))
            except WholeClipError:
                pass
        child.abort()
        raise


def write_scene_audit(
    path: Path,
    *,
    status: str,
    planner_config: Any,
    scenes: list[dict[str, Any]],
    boundary_revisions: Iterable[dict[str, Any]],
    cache: SceneCacheLedger | None,
) -> None:
    value = {
        "schema": 1,
        "version": "whole-clip-scene-audit-v1",
        "status": status,
        "claims": {
            "ground_truth": False,
            "comfort_optimal": False,
            "best_parameters": False,
        },
        "policy": {
            "objective": (
                "stable scene-wide pop and zero-anchor values from finalized "
                "future-corroborated scene evidence"
            ),
            "rule": "StreamingScenePlanner scene-plan-v1",
            "fallback": (
                "configured pop floor and production-latched/neutral anchor when "
                "settled evidence is unavailable"
            ),
            "configuration": _json_ready(vars(planner_config)),
        },
        "scenes": scenes,
        "boundary_revisions": list(boundary_revisions),
        "cache": cache.snapshot() if cache else {
            "enabled": False,
        },
    }
    write_json_atomic(path, value)


def run_streaming_scene_pipeline(
    *,
    sunshine: Path,
    conf: Path,
    build_dir: Path,
    timeline: dict[str, Any],
    source_width: int,
    source_height: int,
    analysis_producer: Any,
    render_producer: Any | None,
    output_dir: Path,
    work_dir: Path,
    artifacts_dir: Path,
    native_extra: list[str],
    cache_max_bytes: int,
    cache_budget_policy: str,
    ffmpeg: str | None,
    ffprobe: str | None,
    source_video: Path | None,
    output_video: Path | None,
    codec: str,
    color: dict[str, Any],
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Run one inference pass and optionally replay finalized scenes into one encoder."""
    try:
        import scene_plan as scene_policy
    except ModuleNotFoundError as exc:
        raise WholeClipError("whole-scene conversion requires scene_plan.py") from exc
    frame_count = int(timeline["frame_count"])
    if frame_count <= 0:
        raise WholeClipError("streaming scene pipeline requires a non-empty timeline")
    conversion = output_video is not None
    if conversion and (
        render_producer is None or ffmpeg is None
    ):
        raise WholeClipError(
            "offline conversion requires a render producer and FFmpeg")
    if not conversion and render_producer is not None:
        raise WholeClipError(
            "adaptive-only scene evaluation must not start a render producer")
    if analysis_producer.extension not in ("png", "pfm"):
        raise WholeClipError("analysis producer has an unsupported follow format")
    if render_producer is not None and (
        render_producer.extension != analysis_producer.extension
    ):
        raise WholeClipError("analysis/render producer color formats differ")
    if conversion:
        preflight_scene_cache_hard_cap(
            source_width,
            source_height,
            cache_max_bytes,
        )

    analysis_input = work_dir / "analysis-input"
    analysis_input.mkdir()
    cache_dir = work_dir / "scene-cache"
    if conversion:
        cache_dir.mkdir()
    analysis_command = [
        os.fspath(sunshine),
        os.fspath(conf),
        "--sbs-bench",
        "--frames", os.fspath(analysis_input),
        "--follow",
        "--follow-format", analysis_producer.extension,
        "--follow-count", str(frame_count),
        "--out", os.fspath(artifacts_dir),
        "--artifacts", "adaptive",
    ]
    if conversion:
        analysis_command += ["--scene-cache", os.fspath(cache_dir)]
    analysis_command += native_extra

    analysis_child = LoggedSubprocess(
        analysis_command, build_dir, output_dir / "harness.log")
    trace_tail = AdaptiveTraceTail(artifacts_dir / TRACE_NAME)
    ledger = SceneCacheLedger(cache_dir, cache_max_bytes) if conversion else None
    planner = None
    scenes: list[dict[str, Any]] = []
    render_results: list[dict[str, Any]] = []
    audit_path = output_dir / "scene_audit.json"
    plans_dir = output_dir / "scene_plans"
    bridge = None
    encoder = None
    encoder_command = None
    encoder_filter = None
    concat_path = work_dir / "sbs-http.ffconcat"
    analysis_terminal = False
    rendered_until = 1
    sbs_dimensions: tuple[int, int] | None = None

    def ensure_encoder() -> tuple[SbsFrameHttpBridge, LoggedSubprocess]:
        nonlocal bridge, encoder, encoder_command, encoder_filter
        if bridge is not None and encoder is not None:
            return bridge, encoder
        assert conversion and output_video is not None and ffmpeg is not None
        output_video.parent.mkdir(parents=True, exist_ok=True)
        bridge = SbsFrameHttpBridge(
            frame_count,
            analysis_producer.extension,
            wait_seconds=timeout_seconds,
        )
        write_http_concat_file(concat_path, bridge, timeline)
        encoder_command, encoder_filter = build_http_encoder_command(
            ffmpeg,
            concat_path,
            timeline,
            source_video,
            output_video,
            codec,
            color,
            sbs_dimensions=sbs_dimensions,
        )
        encoder = LoggedSubprocess(
            encoder_command, None, work_dir / "encode.log")
        return bridge, encoder

    def render_scenes(finalized: list[dict[str, Any]]) -> None:
        nonlocal rendered_until
        for scene in finalized:
            if scene["start_sequence"] != rendered_until:
                raise WholeClipError(
                    "scene planner produced a gap/overlap: "
                    f"{scene['start_sequence']} != {rendered_until}")
            scenes.append(scene)
            if scene["boundary"].get("budget_forced"):
                if ledger is None:
                    raise WholeClipError(
                        "planner reported a budget split without a scene cache")
                ledger.record_forced_segment(
                    scene["start_sequence"],
                    scene["end_sequence_exclusive"],
                )
            write_scene_audit(
                audit_path,
                status="running",
                planner_config=planner.config,
                scenes=scenes,
                boundary_revisions=planner.boundary_revisions,
                cache=ledger,
            )
            if conversion:
                assert ledger is not None and render_producer is not None
                if sbs_dimensions is None:
                    raise WholeClipError(
                        "scene replay started before native SBS geometry was attested")
                current_bridge, current_encoder = ensure_encoder()
                result = render_cached_scene(
                    sunshine=sunshine,
                    conf=conf,
                    build_dir=build_dir,
                    cache_dir=cache_dir,
                    scene=scene,
                    render_producer=render_producer,
                    bridge=current_bridge,
                    encoder=current_encoder,
                    work_dir=work_dir,
                    plans_dir=plans_dir,
                    timeout_seconds=timeout_seconds,
                    expected_sbs_dimensions=sbs_dimensions,
                )
                render_results.append({
                    key: value for key, value in result.items()
                    if key != "sbs_paths"
                })
                released = ledger.release_through(
                    scene["end_sequence_exclusive"])
                if released["pairs"] != scene["frame_count"]:
                    raise WholeClipError(
                        "scene cache release count differs from rendered scene: "
                        f"{released['pairs']} != {scene['frame_count']}")
            rendered_until = scene["end_sequence_exclusive"]

    try:
        for sequence in range(1, frame_count + 1):
            analysis_path = analysis_producer.publish_next(
                analysis_input, sequence)
            progress = wait_for_progress(
                artifacts_dir / "follow_progress.json",
                "processed_count",
                sequence,
                analysis_child,
                timeout_seconds=timeout_seconds,
                expected={
                    "schema": 1,
                    "input_format": analysis_producer.extension,
                    "artifact_mode": "adaptive",
                    "first_sequence": 1,
                },
            )
            if progress.get("last_completed_sequence") != sequence:
                raise WholeClipError(
                    "analysis progress global sequence mismatch: "
                    f"{progress.get('last_completed_sequence')!r} != {sequence}")
            trace_frame = trace_tail.read_frame(
                sequence, analysis_child, timeout_seconds=timeout_seconds)
            timing = timeline["frames"][sequence - 1]
            trace_frame["source_pts_seconds"] = float(timing["pts_time"])
            trace_frame["duration_seconds"] = float(timing["duration_time"])

            pair_bytes = 0
            if conversion:
                assert ledger is not None
                cache_contract = read_json_object(
                    cache_dir / "scene_cache_contract.json",
                    "running scene cache contract",
                )
                pair_bytes = ledger.acknowledge_pair(
                    sequence, cache_contract)
                trace_frame["requires_previous_packed_frame"] = (
                    ledger.requires_previous_packed_frame(sequence)
                )
                current_sbs_dimensions = validate_packed_sbs_contract(
                    cache_contract, analysis_producer.extension)
                if sbs_dimensions is None:
                    sbs_dimensions = current_sbs_dimensions
                elif current_sbs_dimensions != sbs_dimensions:
                    raise WholeClipError(
                        "scene-cache SBS geometry changed during the clip")
                if pair_bytes * 10 > cache_max_bytes:
                    raise WholeClipError(
                        "scene-cache budget must hold at least ten maximum-sized "
                        "depth/state/metadata triplets so the 90% semantic limit "
                        "cannot cross the 100% hard cap: "
                        f"pair={pair_bytes}, cap={cache_max_bytes}")
            analysis_path.unlink()

            if planner is None:
                header = trace_tail.header
                if not isinstance(header, dict):
                    raise WholeClipError("adaptive trace header was not decoded")
                config = header["config"]
                semantic_limit = (
                    cache_max_bytes * 9 // 10 if conversion else 0
                )
                planner = scene_policy.StreamingScenePlanner(
                    scene_policy.ScenePlannerConfig(
                        pop_strength=float(config["pop_strength"]),
                        adaptive_pop=bool(config["adaptive_pop"]),
                        adaptive_pop_max=float(config["adaptive_pop_max"]),
                        zero_plane=str(config["zero_plane"]),
                        max_open_cache_bytes=semantic_limit,
                        budget_policy=cache_budget_policy,
                    )
                )
            finalized = planner.feed(
                trace_frame, frame_cache_bytes=pair_bytes)
            render_scenes(finalized)

        analysis_producer.finish(frame_count)
        publish_producer_terminal(analysis_input, frame_count=frame_count)
        analysis_terminal = True
        analysis_child.wait(
            timeout=timeout_seconds, description="whole-clip analysis")
        terminal_progress = read_json_object(
            artifacts_dir / "follow_progress.json",
            "terminal analysis progress",
        )
        expected_analysis_terminal = {
            "schema": 1,
            "status": "complete",
            "input_format": analysis_producer.extension,
            "artifact_mode": "adaptive",
            "processed_count": frame_count,
            "first_sequence": 1,
            "last_completed_sequence": frame_count,
            "source_frame_count": frame_count,
            "producer_frame_count": frame_count,
        }
        analysis_mismatches = {
            key: {"expected": wanted, "actual": terminal_progress.get(key)}
            for key, wanted in expected_analysis_terminal.items()
            if terminal_progress.get(key) != wanted
        }
        if analysis_mismatches:
            raise WholeClipError(
                "terminal native analysis contract mismatch: " +
                json.dumps(analysis_mismatches, sort_keys=True))
        trace_header = trace_tail.finish(frame_count)
        assert planner is not None
        render_scenes(planner.finish())
        if rendered_until != frame_count + 1:
            raise WholeClipError(
                "final scene plan does not cover the complete source sequence")
        write_scene_audit(
            audit_path,
            status="complete",
            planner_config=planner.config,
            scenes=scenes,
            boundary_revisions=planner.boundary_revisions,
            cache=ledger,
        )

        native_contract = validate_native_outputs(
            artifacts_dir, timeline, "adaptive")
        if conversion:
            assert (
                render_producer is not None and bridge is not None and
                encoder is not None and output_video is not None and ffmpeg is not None
            )
            render_producer.finish(frame_count)
            encoder.wait(
                timeout=timeout_seconds, description="persistent SBS encoder")
            bridge.close(encoder_succeeded=True)
            if ledger is None or ledger.current_bytes != 0:
                raise WholeClipError(
                    "scene cache was not empty after complete replay")
            video_validation = validate_encoded_timeline(
                ffmpeg,
                output_video,
                timeline,
                work_dir / "video-validate.log",
                codec=codec,
                ffprobe=ffprobe,
            )
            hdr_validation = None
            if color.get("mode") in ("hdr-pq", "hdr-hlg"):
                if ffprobe is None:
                    ffprobe = resolve_ffprobe(None, ffmpeg)
                hdr_validation = validate_hdr_metadata(
                    ffprobe,
                    output_video,
                    color,
                    work_dir / "output-hdr-summary.json",
                    work_dir / "output-hdr-probe.log",
                )
        else:
            video_validation = None
            hdr_validation = None

        return {
            "analysis": {
                "command": analysis_command,
                "terminal_progress": terminal_progress,
                "native_contract": native_contract,
                "trace_header": trace_header,
            },
            "scene_audit": {
                "file": os.fspath(audit_path),
                "sha256": sha256_file(audit_path),
                "scene_count": len(scenes),
                "boundary_revision_count": len(planner.boundary_revisions),
            },
            "cache": ledger.snapshot() if ledger else {"enabled": False},
            "render_scenes": render_results,
            "encoder": (
                {
                    "command": encoder_command,
                    "filter_graph": encoder_filter,
                    "concat_file": os.fspath(concat_path),
                    "timeline_validation": video_validation,
                    "hdr_validation": hdr_validation,
                }
                if conversion else None
            ),
        }
    except Exception as exc:
        if not analysis_terminal:
            try:
                publish_producer_terminal(analysis_input, error=str(exc))
            except WholeClipError:
                pass
        analysis_child.abort()
        analysis_producer.abort()
        if render_producer is not None:
            render_producer.abort()
        if bridge is not None:
            bridge.abort(str(exc))
        if encoder is not None:
            encoder.abort()
        if bridge is not None and bridge._server_thread.is_alive():
            try:
                bridge.close(encoder_succeeded=False)
            except WholeClipError:
                pass
        if planner is not None:
            try:
                write_scene_audit(
                    audit_path,
                    status="failed",
                    planner_config=planner.config,
                    scenes=scenes,
                    boundary_revisions=planner.boundary_revisions,
                    cache=ledger,
                )
            except Exception:
                pass
        raise
    finally:
        trace_tail.close()


def encode_sbs_video(
    ffmpeg: str,
    artifacts_dir: Path,
    timeline: dict[str, Any],
    source_video: Path | None,
    output_video: Path,
    codec: str,
    work_dir: Path,
) -> list[list[str]]:
    container = validate_output_target(output_video, codec, work_dir)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    expected_ids = [row["frame_id"] for row in timeline["frames"]]
    frames = [artifacts_dir / f"sbs_{frame_id}.png" for frame_id in expected_ids]
    if not all(path.is_file() for path in frames):
        raise WholeClipError("cannot encode: one or more SBS PNG frames are missing")

    concat_path = work_dir / "sbs.ffconcat"
    write_concat_file(concat_path, frames, timeline)
    first_pts_time = float(
        int(timeline["first_pts"]) * timeline_time_base(timeline))
    encode_command = [
        ffmpeg,
        "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
        "-copyts",
        "-itsoffset", f"{first_pts_time:.12f}",
        "-f", "concat", "-safe", "0", "-i", os.fspath(concat_path),
    ]
    if source_video is not None:
        encode_command += ["-i", os.fspath(source_video)]
    encode_command += [
        "-map", "0:v:0",
    ]
    if source_video is None:
        encode_command += ["-an"]
    else:
        encode_command += [
            "-map", "1:a?",
            "-map_metadata", "1",
            "-map_chapters", "1",
            "-c:a", "copy",
        ]
    encode_command += [
        "-fps_mode", "passthrough",
        "-enc_time_base", "demux",
        *codec_arguments(codec),
        "-bsf:v", _last_duration_bsf(timeline),
        "-avoid_negative_ts", "disabled",
    ]
    if container == "mp4":
        if codec in ("hevc_nvenc", "libx265"):
            encode_command += ["-tag:v", "hvc1"]
        encode_command += ["-movflags", "+faststart", "-f", "mp4"]
    else:
        encode_command += ["-f", "matroska"]
    encode_command.append(os.fspath(output_video))
    run_logged_command(encode_command, None, work_dir / "encode.log")
    if not output_video.is_file() or output_video.stat().st_size == 0:
        raise WholeClipError("FFmpeg did not create the requested SBS video")
    return [encode_command]


def validate_encoded_timeline(
    ffmpeg: str,
    output_video: Path,
    timeline: dict[str, Any],
    log_path: Path,
    codec: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    """Decode the deliverable and prove its presentation clock survived the round trip."""
    command = [
        ffmpeg,
        "-hide_banner", "-loglevel", "info", "-nostdin",
        "-copyts",
    ]
    if codec == "av1_nvenc":
        # The bundled FFmpeg's default libaom decoder lacks high-bit-depth support. NVIDIA's
        # decoder validates the actual Main-10 AV1 output on the supported host hardware.
        command += ["-c:v", "av1_cuvid"]
    command += [
        "-i", os.fspath(output_video),
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", "showinfo",
        "-fps_mode", "passthrough",
        "-f", "null", "-",
    ]
    run_logged_command(command, None, log_path)
    output_time_base, rows = parse_showinfo_file(log_path)
    expected = timeline["frames"]
    if len(rows) != len(expected):
        raise WholeClipError(
            "encoded SBS frame count mismatch: "
            f"decoded {len(rows)}, expected {len(expected)}")

    source_time_base = timeline_time_base(timeline)
    tolerance = output_time_base
    max_pts_error = Fraction(0)
    for index, (actual, wanted) in enumerate(zip(rows, expected)):
        actual_time = actual["pts"] * output_time_base
        expected_time = int(wanted["pts"]) * source_time_base
        error = abs(actual_time - expected_time)
        max_pts_error = max(max_pts_error, error)
        if error > tolerance:
            raise WholeClipError(
                "encoded SBS PTS mismatch at frame "
                f"{index}: actual={float(actual_time):.9f}s "
                f"expected={float(expected_time):.9f}s "
                f"(tolerance={float(tolerance):.9f}s)")

    expected_start = int(expected[0]["pts"]) * source_time_base
    expected_end = (
        int(expected[-1]["pts"]) + int(expected[-1]["duration"])
    ) * source_time_base
    actual_start = rows[0]["pts"] * output_time_base
    expected_duration = expected_end - expected_start
    actual_last_duration = rows[-1].get("duration")
    duration_source = "decoded-final-frame"
    duration_probe_command: list[str] | None = None
    if isinstance(actual_last_duration, int) and actual_last_duration > 0:
        actual_end = (
            rows[-1]["pts"] + actual_last_duration
        ) * output_time_base
        actual_duration = actual_end - actual_start
    else:
        # HEVC/AV1 packet duration is commonly absent after hardware encoding even though the
        # muxer writes an exact presentation duration. Do not invent the final interval from a
        # nominal FPS: independently verify the container duration instead.
        resolved_probe = resolve_ffprobe(ffprobe, ffmpeg)
        duration_json = log_path.with_name(log_path.stem + "-duration.json")
        duration_log = log_path.with_name(log_path.stem + "-duration.log")
        duration_probe_command = [
            resolved_probe,
            "-v", "error",
            "-show_entries", "format=start_time,duration",
            "-of", "json",
            os.fspath(output_video),
        ]
        run_probe_command(duration_probe_command, duration_json, duration_log)
        try:
            duration_value = json.loads(
                duration_json.read_text(encoding="utf-8"))
            format_value = duration_value["format"]
            encoded_duration = Fraction(str(format_value["duration"]))
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            raise WholeClipError(
                "encoded SBS decoder omitted the final-frame duration and ffprobe "
                "did not expose an exact positive container duration") from exc
        if encoded_duration <= 0:
            raise WholeClipError(
                "encoded SBS container duration must be positive")
        actual_duration = encoded_duration
        duration_source = "ffprobe-container"
    duration_error = abs(actual_duration - expected_duration)
    if duration_error > tolerance:
        raise WholeClipError(
            "encoded SBS duration mismatch: "
            f"actual={float(actual_duration):.9f}s "
            f"expected={float(expected_duration):.9f}s "
            f"(tolerance={float(tolerance):.9f}s)")
    return {
        "validated": True,
        "frame_count": len(rows),
        "output_time_base": {
            "num": output_time_base.numerator,
            "den": output_time_base.denominator,
        },
        "tolerance_seconds": float(tolerance),
        "max_pts_error_seconds": float(max_pts_error),
        "expected_duration_seconds": float(expected_duration),
        "actual_duration_seconds": float(actual_duration),
        "duration_error_seconds": float(duration_error),
        "duration_source": duration_source,
        "duration_probe_command": duration_probe_command,
        "command": command,
        "log": os.fspath(log_path),
    }


def validate_hdr_metadata(
    ffprobe: str,
    output_video: Path,
    source_color: dict[str, Any],
    summary_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    summary, command = probe_video_summary(
        ffprobe, output_video, summary_path, log_path)
    output_color = classify_video_color(summary)
    keys = (
        "mode",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "mastering_display",
        "content_light_level",
    )
    mismatches = {
        key: {
            "source": source_color.get(key),
            "output": output_color.get(key),
        }
        for key in keys
        if source_color.get(key) != output_color.get(key)
    }
    if mismatches:
        raise WholeClipError(
            "HDR output color/static metadata does not match the source: "
            + json.dumps(mismatches, sort_keys=True))
    stream = summary["streams"][0]
    profile = str(stream.get("profile") or "")
    if "10" not in profile and "10" not in str(output_color.get("pix_fmt") or ""):
        raise WholeClipError("HDR output is not reported as a 10-bit codec profile")
    return {
        "validated": True,
        "command": command,
        "source": {
            key: source_color.get(key) for key in keys
        },
        "output": {
            **{key: output_color.get(key) for key in keys},
            "profile": profile,
            "pix_fmt": output_color.get("pix_fmt"),
        },
        "summary_file": os.fspath(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }


def remove_expected_sbs_frames(
    artifacts_dir: Path,
    timeline: dict[str, Any],
) -> int:
    """Remove only contract-bound conversion frames after a successful encode."""
    removed = 0
    for timing in timeline["frames"]:
        path = artifacts_dir / f"sbs_{timing['frame_id']}.png"
        if not path.is_file():
            raise WholeClipError(
                f"refusing partial SBS cleanup because an expected frame is missing: {path}")
    for timing in timeline["frames"]:
        (artifacts_dir / f"sbs_{timing['frame_id']}.png").unlink()
        removed += 1
    return removed


def remove_available_expected_sbs_frames(
    artifacts_dir: Path,
    timeline: dict[str, Any] | None,
) -> int:
    """Best-effort failure cleanup constrained to identities owned by this run."""
    if not timeline:
        return 0
    removed = 0
    for timing in timeline.get("frames", []):
        frame_id = timing.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.isdigit():
            continue
        path = artifacts_dir / f"sbs_{frame_id}.png"
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def generate_report_outputs(
    trace_path: Path,
    output_dir: Path,
    timeline: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    try:
        import adaptive_clip_report
    except ModuleNotFoundError as exc:
        if exc.name == "adaptive_clip_report":
            return {"available": False}
        raise
    generator = getattr(adaptive_clip_report, "generate_outputs", None)
    if generator is None:
        return {"available": False}
    result = generator(
        os.fspath(trace_path),
        os.fspath(output_dir),
        timeline=timeline,
        source_name=source_name,
    )
    return {"available": True, "outputs": _json_ready(result)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Accept both `--extra --zero-plane subject` and the visually clearer
    # `--extra -- --zero-plane subject`. argparse treats the latter separator as global unless
    # it is removed before parsing the REMAINDER option.
    try:
        extra_index = arguments.index("--extra")
    except ValueError:
        extra_index = -1
    if extra_index >= 0 and arguments[extra_index + 1:extra_index + 2] == ["--"]:
        del arguments[extra_index + 1]

    parser = argparse.ArgumentParser(
        description="Evaluate a complete clip through one continuous Host SBS pipeline.",
        epilog=(
            "Video PTS are passed through without an FPS filter, so CFR stays CFR and VFR "
            "durations remain variable. Frame directories require --fps or meta.json fps. "
            "Static PQ/HLG HDR is preserved; rotation/dynamic HDR fail closed. Put native "
            "harness overrides last after --extra."
        ),
    )
    parser.add_argument("input", help="source video or a directory of PNG/JPEG frames")
    parser.add_argument("--out", required=True, help="new or empty output directory")
    parser.add_argument(
        "--build-dir", default=os.fspath(DEFAULT_BUILD_DIR),
        help="Apollo build directory (default: cmake-build-relwithdebinfo)")
    parser.add_argument("--sunshine", help="explicit sunshine.exe path")
    parser.add_argument(
        "--ffprobe",
        help="explicit ffprobe executable (otherwise FFPROBE_EXE/PATH/FFmpeg sibling)")
    parser.add_argument("--conf", default=os.fspath(DEFAULT_CONF))
    parser.add_argument(
        "--fps", type=float,
        help="timeline FPS for frame-directory input (otherwise read from meta.json)")
    parser.add_argument(
        "--sbs-video",
        help="also encode a playable SBS .mkv/.mp4 (enables conversion artifacts)")
    parser.add_argument("--codec", choices=CODECS, default="hevc_nvenc")
    parser.add_argument(
        "--scene-cache-max-bytes",
        type=int,
        default=DEFAULT_SCENE_CACHE_MAX_BYTES,
        help=(
            "hard cap for unrendered depth/state cache bytes "
            f"(default: {DEFAULT_SCENE_CACHE_MAX_BYTES})"
        ),
    )
    parser.add_argument(
        "--scene-cache-budget-policy",
        choices=("fail", "split"),
        default="fail",
        help=(
            "fail preserves complete semantic scenes; split is an explicit bounded-storage "
            "fallback and records non-semantic budget-forced segments"
        ),
    )
    parser.add_argument(
        "--keep-work", action="store_true",
        help="retain decoded source PNGs and FFmpeg intermediates after success")
    parser.add_argument(
        "--keep-sbs-frames", action="store_true",
        help="retain per-frame SBS PNGs after a successful video encode")
    parser.add_argument(
        "--extra", nargs=argparse.REMAINDER, default=[],
        help="remaining native harness options (must be the final wrapper option)")
    args = parser.parse_args(arguments)
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    if args.scene_cache_max_bytes <= 0:
        parser.error("--scene-cache-max-bytes must be a positive integer")
    return args


def _run_spooled_legacy(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    if not source.exists():
        raise WholeClipError(f"input does not exist: {source}")
    if not (source.is_file() or source.is_dir()):
        raise WholeClipError(f"input must be a video file or frame directory: {source}")
    require_new_or_empty_directory(output_dir)

    build_dir = Path(args.build_dir).expanduser().resolve()
    sunshine = (
        Path(args.sunshine).expanduser().resolve()
        if args.sunshine else build_dir / "sunshine.exe"
    )
    conf = Path(args.conf).expanduser().resolve()
    output_video = (
        Path(args.sbs_video).expanduser().resolve() if args.sbs_video else None
    )
    if not sunshine.is_file():
        raise WholeClipError(f"sunshine executable does not exist: {sunshine}")
    if not conf.is_file():
        raise WholeClipError(f"configuration file does not exist: {conf}")
    validate_native_extra(args.extra)

    artifact_mode = "conversion" if output_video else "adaptive"
    work_dir = output_dir / "work"
    artifacts_dir = output_dir / "artifacts"
    if output_video:
        validate_output_target(output_video, args.codec, work_dir)
    work_dir.mkdir()
    artifacts_dir.mkdir()
    manifest_path = output_dir / MANIFEST_NAME
    manifest: dict[str, Any] = {
        "schema": 1,
        "status": "initializing",
        "source": {
            "path": os.fspath(source),
            "kind": "frames" if source.is_dir() else "video",
        },
        "sunshine": {
            "path": os.fspath(sunshine),
            "sha256": sha256_file(sunshine),
        },
        "configuration": {
            "path": os.fspath(conf),
            "sha256": sha256_file(conf),
        },
        "artifact_mode": artifact_mode,
        "native_extra": list(args.extra),
        "commands": {},
    }
    write_json_atomic(manifest_path, manifest)

    succeeded = False
    timeline: dict[str, Any] | None = None
    generated_frames_dir: Path | None = None
    video_encode_attempted = False
    try:
        ffmpeg = None
        source_video: Path | None = None
        if source.is_file():
            source_video = source
            try:
                ffmpeg = resolve_ffmpeg()
            except RuntimeError as exc:
                raise WholeClipError(str(exc)) from exc
            metadata = inspect_video_metadata(source)
            reject_unsupported_video(metadata)
            frames_dir = work_dir / "frames"
            generated_frames_dir = frames_dir
            source_width, source_height = video_dimensions(metadata)
            try:
                manifest["disk_preflight"] = disk_spool_preflight(
                    output_dir,
                    frame_count=estimated_video_frame_count(metadata),
                    source_width=source_width,
                    source_height=source_height,
                    source_spooled=True,
                    conversion=output_video is not None,
                    native_extra=args.extra,
                )
            except DiskSpaceError as exc:
                manifest["disk_preflight"] = exc.preflight
                raise
            write_json_atomic(manifest_path, manifest)
            timeline, decode_command = decode_video(
                ffmpeg, source, frames_dir, work_dir / "decode.log", metadata)
            frames = frame_directory_files(frames_dir)
            manifest["ffmpeg"] = {
                "path": ffmpeg,
                "metadata": metadata,
            }
            manifest["commands"]["decode"] = decode_command
            manifest["source"]["sha256"] = sha256_file(source)
            manifest["source"]["size_bytes"] = source.stat().st_size
        else:
            frames_dir = source
            frames = frame_directory_files(frames_dir)
            source_width, source_height = frame_dimensions(frames[0])
            try:
                manifest["disk_preflight"] = disk_spool_preflight(
                    output_dir,
                    frame_count=len(frames),
                    source_width=source_width,
                    source_height=source_height,
                    source_spooled=False,
                    conversion=output_video is not None,
                    native_extra=args.extra,
                )
            except DiskSpaceError as exc:
                manifest["disk_preflight"] = exc.preflight
                raise
            timeline = build_frame_directory_timeline(
                frames, frame_directory_fps(source, args.fps))
            manifest["source"]["sha256"] = sha256_frame_set(source, frames)
            manifest["source"]["frame_count"] = len(frames)
            if output_video:
                try:
                    ffmpeg = resolve_ffmpeg()
                except RuntimeError as exc:
                    raise WholeClipError(str(exc)) from exc
                manifest["ffmpeg"] = {"path": ffmpeg}

        timeline_path = output_dir / TIMELINE_NAME
        write_json_atomic(timeline_path, timeline)
        manifest["timeline"] = {
            "file": TIMELINE_NAME,
            "sha256": sha256_file(timeline_path),
            "frame_count": timeline["frame_count"],
            "variable_frame_rate": timeline["variable_frame_rate"],
            "first_pts_time": timeline["first_pts_time"],
        }
        manifest["status"] = "decoded"
        write_json_atomic(manifest_path, manifest)

        harness_command = [
            os.fspath(sunshine),
            os.fspath(conf),
            "--sbs-bench",
            "--frames", os.fspath(frames_dir),
            "--out", os.fspath(artifacts_dir),
            "--artifacts", artifact_mode,
            *args.extra,
        ]
        manifest["commands"]["harness"] = harness_command
        write_json_atomic(manifest_path, manifest)
        run_logged_command(harness_command, build_dir, output_dir / "harness.log")
        native_contract = validate_native_outputs(
            artifacts_dir, timeline, artifact_mode)
        contract_path = artifacts_dir / NATIVE_CONTRACT_NAME
        trace_path = artifacts_dir / TRACE_NAME
        manifest["native_contract"] = {
            "file": f"artifacts/{NATIVE_CONTRACT_NAME}",
            "sha256": sha256_file(contract_path),
            "value": native_contract,
        }
        manifest["runtime_provenance"] = runtime_provenance(
            build_dir, native_contract)
        manifest["adaptive_trace"] = {
            "file": f"artifacts/{TRACE_NAME}",
            "sha256": sha256_file(trace_path),
        }
        manifest["status"] = "harness_complete"
        write_json_atomic(manifest_path, manifest)

        manifest["report"] = generate_report_outputs(
            trace_path, output_dir, timeline, source.name)
        if output_video:
            assert ffmpeg is not None
            video_encode_attempted = True
            video_commands = encode_sbs_video(
                ffmpeg,
                artifacts_dir,
                timeline,
                source_video,
                output_video,
                args.codec,
                work_dir,
            )
            manifest["commands"]["video"] = video_commands
            video_validation = validate_encoded_timeline(
                ffmpeg,
                output_video,
                timeline,
                work_dir / "video-validate.log",
            )
            manifest["commands"]["video_validation"] = video_validation["command"]
            manifest["sbs_video"] = {
                "path": os.fspath(output_video),
                "container": _container_for(output_video),
                "codec": args.codec,
                "sha256": sha256_file(output_video),
                "size_bytes": output_video.stat().st_size,
                "audio": "source-stream-copy" if source_video else "none",
                "timeline_validation": {
                    key: value for key, value in video_validation.items()
                    if key not in ("command", "log")
                },
            }
            if args.keep_sbs_frames:
                manifest["cleanup"] = {
                    "sbs_frames": {
                        "kept": True,
                        "removed": 0,
                    },
                }
            else:
                removed = remove_expected_sbs_frames(artifacts_dir, timeline)
                manifest["cleanup"] = {
                    "sbs_frames": {
                        "kept": False,
                        "removed": removed,
                    },
                }

        manifest["status"] = "complete"
        succeeded = True
        write_json_atomic(manifest_path, manifest)
        return manifest
    except Exception as exc:
        if not args.keep_work:
            failure_cleanup: dict[str, Any] = {}
            if generated_frames_dir and generated_frames_dir.is_dir():
                # This exact directory was created below a previously empty output root.
                shutil.rmtree(generated_frames_dir)
                failure_cleanup["source_frame_spool_removed"] = True
            removed_sbs = remove_available_expected_sbs_frames(
                artifacts_dir, timeline)
            failure_cleanup["sbs_frames_removed"] = removed_sbs
            if video_encode_attempted and output_video and output_video.is_file():
                output_video.unlink()
                failure_cleanup["partial_sbs_video_removed"] = True
            manifest["failure_cleanup"] = failure_cleanup
        else:
            manifest["failure_cleanup"] = {
                "kept": True,
                "reason": "--keep-work",
            }
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json_atomic(manifest_path, manifest)
        if isinstance(exc, WholeClipError):
            raise
        raise WholeClipError(str(exc)) from exc
    finally:
        if succeeded and not args.keep_work:
            shutil.rmtree(work_dir)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the bounded whole-clip evaluation/conversion pipeline."""
    source = Path(args.input).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    if not source.exists():
        raise WholeClipError(f"input does not exist: {source}")
    if not (source.is_file() or source.is_dir()):
        raise WholeClipError(f"input must be a video file or frame directory: {source}")
    require_new_or_empty_directory(output_dir)

    build_dir = Path(args.build_dir).expanduser().resolve()
    sunshine = (
        Path(args.sunshine).expanduser().resolve()
        if args.sunshine else build_dir / "sunshine.exe"
    )
    conf = Path(args.conf).expanduser().resolve()
    output_video = (
        Path(args.sbs_video).expanduser().resolve() if args.sbs_video else None
    )
    cache_max_bytes = int(getattr(
        args, "scene_cache_max_bytes", DEFAULT_SCENE_CACHE_MAX_BYTES))
    cache_budget_policy = str(getattr(
        args, "scene_cache_budget_policy", "fail"))
    explicit_ffprobe = getattr(args, "ffprobe", None)
    if cache_max_bytes <= 0:
        raise WholeClipError("scene-cache max bytes must be positive")
    if cache_budget_policy not in ("fail", "split"):
        raise WholeClipError("scene-cache budget policy must be fail or split")
    if getattr(args, "keep_sbs_frames", False):
        raise WholeClipError(
            "--keep-sbs-frames is incompatible with bounded streaming conversion; "
            "the final encoded video is the durable conversion artifact")
    if not sunshine.is_file():
        raise WholeClipError(f"sunshine executable does not exist: {sunshine}")
    if not conf.is_file():
        raise WholeClipError(f"configuration file does not exist: {conf}")
    validate_native_extra(args.extra)

    work_dir = output_dir / "work"
    artifacts_dir = output_dir / "artifacts"
    evidence_dir = output_dir / "evidence"
    work_dir.mkdir()
    artifacts_dir.mkdir()
    evidence_dir.mkdir()
    if output_video:
        validate_output_target(output_video, args.codec, work_dir)
    manifest_path = output_dir / MANIFEST_NAME
    manifest: dict[str, Any] = {
        "schema": 2,
        "status": "initializing",
        "pipeline": "bounded-whole-scene-v1",
        "source": {
            "path": os.fspath(source),
            "kind": "frames" if source.is_dir() else "video",
        },
        "sunshine": {
            "path": os.fspath(sunshine),
            "sha256": sha256_file(sunshine),
        },
        "configuration": {
            "path": os.fspath(conf),
            "sha256": sha256_file(conf),
        },
        "artifact_mode": (
            "offline-conversion" if output_video else "adaptive-evaluation"
        ),
        "native_extra": list(args.extra),
        "scene_cache": {
            "hard_cap_bytes": cache_max_bytes if output_video else 0,
            "semantic_limit_bytes": (
                cache_max_bytes * 9 // 10 if output_video else 0
            ),
            "budget_policy": cache_budget_policy,
            "default_preserves_complete_semantic_scene":
                cache_budget_policy == "fail",
        },
        "commands": {},
    }
    write_json_atomic(manifest_path, manifest)

    analysis_producer = None
    render_producer = None
    succeeded = False
    try:
        capabilities = query_native_capabilities(
            sunshine,
            build_dir,
            evidence_dir / "native-capabilities.json",
            evidence_dir / "native-capabilities.log",
        )
        manifest["native_capabilities"] = {
            key: value for key, value in capabilities.items()
            if key != "command"
        }
        manifest["commands"]["capabilities"] = capabilities["command"]
        manifest["status"] = "capabilities-validated"
        write_json_atomic(manifest_path, manifest)

        ffmpeg: str | None = None
        ffprobe: str | None = None
        source_video: Path | None = None
        frames: list[Path] | None = None
        color: dict[str, Any]
        if source.is_file():
            source_video = source
            try:
                ffmpeg = resolve_ffmpeg()
            except RuntimeError as exc:
                raise WholeClipError(str(exc)) from exc
            ffprobe = resolve_ffprobe(explicit_ffprobe, ffmpeg)
            inspection = inspect_streaming_video(
                ffprobe, source, evidence_dir)
            timeline = inspection["timeline"]
            color = inspection["color"]
            width = inspection["width"]
            height = inspection["height"]
            manifest["source"].update({
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
                "width": width,
                "height": height,
                "color": color,
            })
            manifest["ffmpeg"] = {
                "path": ffmpeg,
                "ffprobe": ffprobe,
                "summary_file": inspection["summary_file"],
                "summary_sha256": inspection["summary_sha256"],
                "timeline_scan_file": inspection["timeline_file"],
                "timeline_scan_sha256": inspection["timeline_sha256"],
                "timeline_scan": inspection["timeline_scan"],
            }
            manifest["commands"]["probe"] = inspection["commands"]
            source_size = source.stat().st_size
        else:
            frames = frame_directory_files(source)
            dimensions = [frame_dimensions(path) for path in frames]
            width, height = dimensions[0]
            if any(value != (width, height) for value in dimensions[1:]):
                raise WholeClipError(
                    "frame directory contains mixed source dimensions")
            timeline = build_frame_directory_timeline(
                frames, frame_directory_fps(source, args.fps))
            color = {
                "mode": "sdr",
                "pix_fmt": "rgba8-via-wic",
                "color_range": "full",
                "color_space": "bt709",
                "color_transfer": "iec61966-2-1",
                "color_primaries": "bt709",
                "mastering_display": None,
                "content_light_level": None,
            }
            if output_video:
                try:
                    ffmpeg = resolve_ffmpeg()
                except RuntimeError as exc:
                    raise WholeClipError(str(exc)) from exc
                ffprobe = resolve_ffprobe(explicit_ffprobe, ffmpeg)
            manifest["source"].update({
                "sha256": sha256_frame_set(source, frames),
                "frame_count": len(frames),
                "width": width,
                "height": height,
                "color": color,
            })
            if ffmpeg:
                manifest["ffmpeg"] = {
                    "path": ffmpeg,
                    "ffprobe": ffprobe,
                }
            source_size = 0

        manifest["disk_preflight"] = streaming_disk_preflight(
            output_dir,
            cache_max_bytes=cache_max_bytes,
            conversion=output_video is not None,
            source_size_bytes=source_size,
        )
        if output_video:
            manifest["scene_cache"]["maximum_triplet_bytes"] = (
                preflight_scene_cache_hard_cap(
                    width,
                    height,
                    cache_max_bytes,
                )
            )
        if source_video is not None:
            assert ffmpeg is not None
            analysis_producer = VideoFollowFrameDecoder(
                ffmpeg,
                source_video,
                width,
                height,
                color,
                work_dir / "decode-analysis.log",
            )
            if output_video:
                render_producer = VideoFollowFrameDecoder(
                    ffmpeg,
                    source_video,
                    width,
                    height,
                    color,
                    work_dir / "decode-render.log",
                )
        else:
            assert frames is not None
            analysis_producer = FrameDirectoryFollowProducer(frames)
            if output_video:
                render_producer = FrameDirectoryFollowProducer(frames)
        timeline_path = output_dir / TIMELINE_NAME
        write_json_atomic(timeline_path, timeline)
        manifest["timeline"] = {
            "file": TIMELINE_NAME,
            "sha256": sha256_file(timeline_path),
            "frame_count": timeline["frame_count"],
            "variable_frame_rate": timeline["variable_frame_rate"],
            "first_pts_time": timeline["first_pts_time"],
        }
        manifest["status"] = "source-validated"
        write_json_atomic(manifest_path, manifest)

        assert analysis_producer is not None
        pipeline = run_streaming_scene_pipeline(
            sunshine=sunshine,
            conf=conf,
            build_dir=build_dir,
            timeline=timeline,
            source_width=width,
            source_height=height,
            analysis_producer=analysis_producer,
            render_producer=render_producer,
            output_dir=output_dir,
            work_dir=work_dir,
            artifacts_dir=artifacts_dir,
            native_extra=list(args.extra),
            cache_max_bytes=cache_max_bytes,
            cache_budget_policy=cache_budget_policy,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            source_video=source_video,
            output_video=output_video,
            codec=args.codec,
            color=color,
        )
        manifest["commands"]["analysis"] = pipeline["analysis"]["command"]
        if pipeline["encoder"]:
            manifest["commands"]["encoder"] = pipeline["encoder"]["command"]
        manifest["native_contract"] = {
            "file": f"artifacts/{NATIVE_CONTRACT_NAME}",
            "sha256": sha256_file(artifacts_dir / NATIVE_CONTRACT_NAME),
            "value": pipeline["analysis"]["native_contract"],
        }
        manifest["runtime_provenance"] = runtime_provenance(
            build_dir, pipeline["analysis"]["native_contract"])
        trace_path = artifacts_dir / TRACE_NAME
        manifest["adaptive_trace"] = {
            "file": f"artifacts/{TRACE_NAME}",
            "sha256": sha256_file(trace_path),
            "schema": ADAPTIVE_TRACE_SCHEMA,
        }
        manifest["scene_audit"] = pipeline["scene_audit"]
        manifest["scene_cache"]["result"] = pipeline["cache"]
        manifest["render_scenes"] = pipeline["render_scenes"]
        manifest["report"] = generate_report_outputs(
            trace_path, output_dir, timeline, source.name)
        if output_video:
            manifest["sbs_video"] = {
                "path": os.fspath(output_video),
                "container": _container_for(output_video),
                "codec": args.codec,
                "sha256": sha256_file(output_video),
                "size_bytes": output_video.stat().st_size,
                "audio": "source-stream-copy" if source_video else "none",
                "timeline_validation": {
                    key: value
                    for key, value in pipeline["encoder"][
                        "timeline_validation"].items()
                    if key not in ("command", "log")
                },
                "hdr_validation": pipeline["encoder"]["hdr_validation"],
            }
        manifest["cleanup"] = {
            "whole_source_frame_spool": False,
            "whole_sbs_frame_spool": False,
            "work_kept": bool(args.keep_work),
        }
        manifest["status"] = "complete"
        write_json_atomic(manifest_path, manifest)
        succeeded = True
        return manifest
    except Exception as exc:
        partial_video_existed = bool(output_video and output_video.is_file())
        if partial_video_existed:
            assert output_video is not None
            output_video.unlink()
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["failure_cleanup"] = {
            "partial_sbs_video_removed": partial_video_existed,
            "work_kept_for_diagnostics": True,
        }
        write_json_atomic(manifest_path, manifest)
        if isinstance(exc, WholeClipError):
            raise
        raise WholeClipError(str(exc)) from exc
    finally:
        if analysis_producer is not None:
            analysis_producer.abort()
        if render_producer is not None:
            render_producer.abort()
        if succeeded and not args.keep_work:
            shutil.rmtree(work_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except WholeClipError as exc:
        print(f"run_whole_clip: {exc}", file=sys.stderr)
        return 2
    print(
        f"run_whole_clip: {result['status']} "
        f"({result['timeline']['frame_count']} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
