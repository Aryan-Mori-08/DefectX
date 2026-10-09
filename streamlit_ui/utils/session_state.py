"""Session state management and persistence helpers for Streamlit."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import streamlit as st


def init_session_state() -> None:
    """Initializes standard session state keys if not already present."""
    defaults: Dict[str, Any] = {
        # Current active inspection frame and outputs
        "current_image_bgr": None,
        "current_anomaly_map": None,
        "current_anomaly_score": None,
        "current_latency_ms": 0.0,
        "current_quality_report": None,
        "current_profile": None,
        "current_sample_name": "capture",
        "current_decision": None,
        # History ledger for analytics and CSV downloads
        "inspection_history": [],
        # Operator feedback logs
        "feedback_logs": [],
        # Selected profile
        "selected_profile": "pcb_rev_b",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_inspection_result(
    profile: str,
    score: float,
    decision: str,
    latency_ms: float,
    num_regions: int,
    image_name: str = "capture",
) -> None:
    """Records an inspection decision into the session history ledger."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "profile": profile,
        "anomaly_score": round(score, 4),
        "decision": decision,
        "latency_ms": round(latency_ms, 1),
        "num_regions": num_regions,
    }
    st.session_state["inspection_history"].append(entry)


def log_operator_feedback(
    profile: str,
    action: str,
    category: str,
    notes: str,
    score: float,
) -> None:
    """Records an operator feedback action."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "action": action,
        "category": category,
        "notes": notes,
        "anomaly_score": round(score, 4),
    }
    st.session_state["feedback_logs"].append(entry)
