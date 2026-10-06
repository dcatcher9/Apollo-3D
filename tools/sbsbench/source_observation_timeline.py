"""Source observation timeline ABI shared by estimator evaluators and native replay."""

from pathlib import Path
import struct


OBSERVATION_TIMELINE_MAGIC = b"SBSOTL1\0"
OBSERVATION_TIMELINE_SCHEMA = 1
OBSERVATION_TIMELINE_HEADER_BYTES = 24


class TimelineError(RuntimeError):
    """Invalid or unauthenticated source-time evidence."""


def parse_positive_rational(value: str, option: str) -> tuple[int, int]:
    try:
        numerator_text, denominator_text = value.split("/", 1)
        numerator, denominator = int(numerator_text), int(denominator_text)
    except (ValueError, AttributeError) as exc:
        raise TimelineError(f"{option} must be a positive NUM/DEN rational") from exc
    if numerator <= 0 or denominator <= 0:
        raise TimelineError(f"{option} must be a positive NUM/DEN rational")
    return numerator, denominator


def prepared_observation_timestamps(frame_count: int, fps: str) -> list[int]:
    if type(frame_count) is not int or frame_count <= 0:
        raise TimelineError("prepared timeline requires a positive frame count")
    numerator, denominator = parse_positive_rational(fps, "prepared source cadence")
    timestamps = [1 + index * 1_000_000 * denominator // numerator
                  for index in range(frame_count)]
    if timestamps[-1] > (1 << 64) - 1:
        raise TimelineError("prepared source cadence exceeds uint64 microseconds")
    return timestamps


def observation_timeline_payload(timestamps: list[int]) -> bytes:
    if (not timestamps or
            any(type(value) is not int or not 0 < value <= (1 << 64) - 1
                for value in timestamps) or
            any(later < earlier for earlier, later in zip(timestamps, timestamps[1:]))):
        raise TimelineError("observation timeline is empty, zero, regressed, or outside uint64")
    payload = struct.pack("<8sIIQ", OBSERVATION_TIMELINE_MAGIC,
                          OBSERVATION_TIMELINE_SCHEMA,
                          OBSERVATION_TIMELINE_HEADER_BYTES, len(timestamps))
    return payload + struct.pack(f"<{len(timestamps)}Q", *timestamps)


def write_observation_timeline(path: Path, timestamps: list[int]) -> None:
    path.write_bytes(observation_timeline_payload(timestamps))


def read_observation_timeline(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) < OBSERVATION_TIMELINE_HEADER_BYTES:
        raise TimelineError("observation timeline is shorter than its header")
    magic, schema, header_bytes, count = struct.unpack("<8sIIQ", data[:24])
    if (magic != OBSERVATION_TIMELINE_MAGIC or schema != OBSERVATION_TIMELINE_SCHEMA or
            header_bytes != OBSERVATION_TIMELINE_HEADER_BYTES or count == 0 or
            len(data) != header_bytes + count * 8):
        raise TimelineError("observation timeline header or length is invalid")
    timestamps = list(struct.unpack(f"<{count}Q", data[header_bytes:]))
    if (any(value == 0 for value in timestamps) or
            any(later < earlier for earlier, later in zip(timestamps, timestamps[1:]))):
        raise TimelineError("observation timeline timestamps are zero or regressed")
    return timestamps
