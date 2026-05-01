"""Streamlit entry point for the Cook Inlet Belugas review app."""

import streamlit as st

st.set_page_config(
    page_title="Cook Inlet Belugas — AL Review",
    page_icon="🐳",
    layout="wide",
)

st.title("🐳 Cook Inlet Belugas — Active Learning Review")
st.markdown(
    """
    Use the sidebar to navigate between tools.

    | Page | Purpose |
    |---|---|
    | **1 · Review** | Step through spectrogram predictions, listen to audio, and assign `manual_verif` labels. |
    | **2 · AL Targets** | Check whether any threshold/model meets your Precision / Recall / F1 requirements. |

    ---
    Your reviewed labels are saved to `frontend/reviews/<csv_name>_<username>_reviewed.csv`
    and never overwrite the source CSV.
    """
)
