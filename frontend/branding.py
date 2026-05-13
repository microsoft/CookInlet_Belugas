"""SPIRAL branding helper.

Loads `spiral_logo.svg` once per session and exposes a `render_logo()` helper
that embeds it inline in the page so it scales with the container width
instead of staying at its native 800x220 px size.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

_SVG_PATH = Path(__file__).parent / "spira_logo_long.svg"


@st.cache_data(show_spinner=False)
def _load_svg(path: str, mtime: float) -> str:
    """Load the SVG and strip its fixed width/height so CSS can size it.

    `mtime` is included in the cache key so any change to the file (or to
    the configured path) invalidates the cache automatically.
    """
    svg = Path(path).read_text()
    svg = re.sub(r'\swidth="\d+"', ' width="100%"', svg, count=1)
    svg = re.sub(r'\sheight="\d+"', "", svg, count=1)
    return svg


def render_logo(max_width: int | None = None) -> None:
    """Render the SPIRAL SVG logo as the page header.

    With `max_width=None` (default) the logo spans the full container width
    so it matches the spectrogram below it. Pass an integer pixel value to
    constrain it.
    """
    try:
        mtime = _SVG_PATH.stat().st_mtime
    except OSError:
        mtime = 0.0
    svg = _load_svg(str(_SVG_PATH), mtime)
    style = "width:100%; margin-bottom:0.5rem"
    if max_width is not None:
        style = f"max-width:{max_width}px; {style}"
    st.markdown(
        f"<div style='{style}'>{svg}</div>",
        unsafe_allow_html=True,
    )
