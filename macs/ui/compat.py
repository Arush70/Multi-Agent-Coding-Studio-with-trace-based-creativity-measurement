"""Small compatibility helpers for the dashboard across Streamlit versions.

Streamlit 1.46 added ``width="stretch"`` to ``st.dataframe`` and deprecated
``use_container_width`` (removed in later releases). ``FULL_WIDTH`` gives the right keyword for
the installed version, so tables fill the page on old and new Streamlit alike. The version is read
from the package metadata, so importing this module does not import Streamlit itself.
"""
from __future__ import annotations

from importlib import metadata


def _version(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _streamlit_version() -> tuple[int, ...]:
    try:
        return _version(metadata.version("streamlit"))
    except metadata.PackageNotFoundError:
        return (0,)


FULL_WIDTH: dict = ({"width": "stretch"} if _streamlit_version() >= (1, 46, 0)
                    else {"use_container_width": True})
