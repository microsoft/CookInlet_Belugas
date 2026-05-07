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
    """Return the recording start datetime for a CSV `audio` value, or None."""
    m = _FILE_ID_FROM_BASENAME.match(audio_basename)
    if not m:
        return None
    return _index().get(int(m.group(1)))
