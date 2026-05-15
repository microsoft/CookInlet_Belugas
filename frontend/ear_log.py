"""Optional EAR.LOG datetime lookup.

The Cook Inlet Belugas dataset ships with `frontend/EAR.LOG` — a text log
from the EAR recorder mapping numeric file IDs to recording start times.
Other projects can either supply an analogous log via the EAR_LOG_PATH env
var or ignore this module: every public function returns None gracefully
when the log is missing or empty.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import config

_RECORDING_LINE = re.compile(
    r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\s+Recording to\s+(\d+)\.",
    re.IGNORECASE,
)
_FILE_ID_FROM_BASENAME = re.compile(r"^\d+_(\d+)")
# Fallback: extract YYYYMMDD<sep>HHMMSS (optionally followed by Z for UTC) from
# anywhere in the basename, where <sep> is either `_` (e.g.
# "DFOCRP.KkHK0R2F-NML1.SM2M-3.20131011_000600Z.flac") or `T` for ISO-8601
# style (e.g. "AMAR779.20210914T160033Z.wav").
_DATETIME_FROM_BASENAME = re.compile(
    r"(\d{4})(\d{2})(\d{2})[_T](\d{2})(\d{2})(\d{2})(Z?)"
)
# SoundTrap names use a compact 2-digit-year timestamp with no separator,
# delimited by dots, e.g. "ST6249.220308193442.wav" → 2022-03-08 19:34:42.
# Field ranges are validated in the regex so random 12-digit runs don't match.
_SOUNDTRAP_DATETIME = re.compile(
    r"\.(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])"
    r"([01]\d|2[0-3])([0-5]\d)([0-5]\d)\."
)


@lru_cache(maxsize=1)
def _index() -> dict[int, str]:
    if not config.EAR_LOG_PATH.exists():
        return {}
    out: dict[int, str] = {}
    with config.EAR_LOG_PATH.open(errors="replace") as f:
        for line in f:
            m = _RECORDING_LINE.match(line)
            if m:
                out[int(m.group(2))] = m.group(1)
    return out


def recording_date(audio_basename: str) -> Optional[str]:
    """Return the recording start datetime for a CSV `audio` value, or None.

    Tries the EAR.LOG lookup first (Cook Inlet format: leading numeric file
    id). Falls back to parsing a `YYYYMMDD_HHMMSS[Z]` timestamp embedded in
    the basename (used by the orca dataset filenames).
    """
    m = _FILE_ID_FROM_BASENAME.match(audio_basename)
    if m:
        hit = _index().get(int(m.group(1)))
        if hit is not None:
            return hit

    m = _DATETIME_FROM_BASENAME.search(audio_basename)
    if m:
        y, mo, d, h, mi, s, z = m.groups()
        suffix = " UTC" if z else ""
        return f"{y}-{mo}-{d} {h}:{mi}:{s}{suffix}"

    m = _SOUNDTRAP_DATETIME.search(audio_basename)
    if m:
        yy, mo, d, h, mi, s = m.groups()
        return f"20{yy}-{mo}-{d} {h}:{mi}:{s}"
    return None
