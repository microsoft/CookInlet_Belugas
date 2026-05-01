import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

EAR_LOG_PATH = Path(
    "/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/frontend/EAR.LOG"
)
RECORDING_LINE = re.compile(
    r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) Recording to (\d+)\.MOT"
)
AUDIO_INDEX = re.compile(r"^\d+_(\d+)")


@lru_cache(maxsize=1)
def _index_to_timestamp() -> dict:
    if not EAR_LOG_PATH.exists():
        return {}
    out = {}
    with EAR_LOG_PATH.open() as f:
        for line in f:
            m = RECORDING_LINE.match(line)
            if m:
                out[int(m.group(2))] = m.group(1)
    return out


def recording_date(audio_basename: str) -> Optional[str]:
    m = AUDIO_INDEX.match(audio_basename)
    if not m:
        return None
    return _index_to_timestamp().get(int(m.group(1)))
