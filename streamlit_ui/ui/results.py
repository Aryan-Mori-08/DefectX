"""Results and Metrics UI module.

Displays production line KPI metrics (Yield %, PASS/REJECT counts, latency),
shows an audit log table of inspected items, and provides CSV export capabilities.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def render_results_page() -> None:
    """Renders the inspection metrics, historical table, and CSV download."""
    st.header("📊 Production Line Yield & Metrics")

    history: List[Dict[str, Any]] = st.session_state.get("inspection_history", [])

    total_inspections = len(history)
    pass_count = sum(1 for item in history if item.get("decision") == "PASS")
    reject_count = sum(1 for item in history if item.get("decision") == "REJECT")
    low_quality_count = sum(1 for item in history if item.get("decision") == "LOW QUALITY CAPTURE")

    yield_pct = (pass_count / total_inspections * 100.0) if total_inspections > 0 else 100.0
    avg_latency = (
        sum(item.get("latency_ms", 0.0) for item in history) / total_inspections
        if total_inspections > 0
        else 0.0
    )
    latest_decision = history[-1].get("decision", "N/A") if history else "N/A"

    # -------------------------------------------------------------------------
    # Top KPI Metric Cards
    # -------------------------------------------------------------------------
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)

    with kpi_c1:
        st.metric("Total Inspections", f"{total_inspections}")
    with kpi_c2:
        st.metric("PASS Count", f"{pass_count}", delta=f"{pass_count} units")
    with kpi_c3:
        st.metric("REJECT Count", f"{reject_count}", delta=f"-{reject_count} units", delta_color="inverse")
    with kpi_c4:
        st.metric("Line Yield %", f"{yield_pct:.1f}%")
    with kpi_c5:
        st.metric("Avg Latency", f"{avg_latency:.1f} ms")

    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        st.metric("Low-Quality / Retried", f"{low_quality_count}")
    with sub_c2:
        st.metric("Latest Result", latest_decision)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Recent Inspections Table
    # -------------------------------------------------------------------------
    st.subheader("📋 Recent Inspection Log")

    if not history:
        st.info("No inspection records logged in this session yet. Run an inspection in Tab 1 to see data.")
    else:
        df = pd.DataFrame(history)
        # Reorder columns for readability
        desired_cols = [
            "timestamp",
            "image_name",
            "profile",
            "anomaly_score",
            "decision",
            "latency_ms",
            "num_regions",
        ]
        available_cols = [c for c in desired_cols if c in df.columns]
        df_display = df[available_cols]

        st.dataframe(df_display, use_container_width=True)

        # ---------------------------------------------------------------------
        # CSV Export Button
        # ---------------------------------------------------------------------
        csv_buffer = io.StringIO()
        df_display.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="📥 Download Inspection Log (CSV)",
            data=csv_data,
            file_name="inspection_results.csv",
            mime="text/csv",
            type="primary",
        )
