"""Operator Feedback UI module.

Handles operator validation for REJECT events:
- Confirming genuine defects with category taxonomy tags.
- Marking false alarms into a quarantined staging buffer via submit_false_alarm_feedback().
- Preserving models without direct mutation.
"""

from __future__ import annotations

import streamlit as st

from utils.model_interface import submit_false_alarm_feedback
from utils.session_state import log_operator_feedback


DEFECT_CATEGORIES = [
    "Scratch",
    "Dent",
    "Crack",
    "Contamination",
    "Deformation",
    "Discoloration",
    "Missing component",
    "Other",
]


def render_feedback_page(selected_profile: str) -> None:
    """Renders the operator feedback workflow for rejected parts."""
    st.header("🔄 Operator Quality Feedback")
    st.markdown(
        "Review flagged units and submit feedback. Genuine defects are tagged with taxonomy "
        "categories for scrap logs, while false alarms are quarantined into a staging buffer for QA review."
    )

    current_decision = st.session_state.get("current_decision")
    current_image = st.session_state.get("current_image_bgr")
    current_map = st.session_state.get("current_anomaly_map")
    current_score = st.session_state.get("current_anomaly_score", 0.0)

    # -------------------------------------------------------------------------
    # Active Rejection Action Panel
    # -------------------------------------------------------------------------
    if current_decision == "REJECT" and current_image is not None:
        st.subheader("⚠️ Active Unit Flagged as REJECT")

        f_col1, f_col2 = st.columns([1, 1])

        with f_col1:
            st.metric("Anomaly Score", f"{current_score:.4f}")
            defect_category = st.selectbox(
                "Defect Category Classification",
                DEFECT_CATEGORIES,
                help="Categorize the defect root cause for production quality tracking.",
            )
            operator_notes = st.text_area(
                "Operator Comments",
                placeholder="e.g. Scuffed protective film, hairline scratch along pad 3...",
            )

        with f_col2:
            st.markdown("### Action Dispatch")

            # 1. Confirm Defect
            if st.button("✅ Confirm Genuine Defect", type="primary"):
                log_operator_feedback(
                    profile=selected_profile,
                    action="CONFIRM_DEFECT",
                    category=defect_category,
                    notes=operator_notes,
                    score=current_score,
                )
                st.success(f"Defect confirmed as '{defect_category}'. Unit logged for rework/scrap.")

            st.markdown("---")

            # 2. Mark as False Alarm
            if st.button("🚩 Mark as False Alarm"):
                success = submit_false_alarm_feedback(
                    image_bgr=current_image,
                    anomaly_map=current_map,
                    profile_name=selected_profile,
                    defect_category="False Alarm",
                    notes=operator_notes,
                )
                if success:
                    log_operator_feedback(
                        profile=selected_profile,
                        action="FALSE_ALARM",
                        category=defect_category,
                        notes=operator_notes,
                        score=current_score,
                    )
                    st.success(
                        "✅ Marked as False Alarm! Candidate patches have been quarantined "
                        "into the staging buffer. The production memory bank remains untouched."
                    )
                else:
                    st.error("Failed to submit false alarm feedback.")

    elif current_decision == "PASS":
        st.info("The latest inspected unit was evaluated as PASS. Feedback actions are only required for REJECT units.")
    else:
        st.info("No active inspection frame loaded. Capture an image on the Inspection tab to begin.")

    # -------------------------------------------------------------------------
    # Feedback Audit Ledger
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📜 Operator Feedback Ledger")

    feedback_logs = st.session_state.get("feedback_logs", [])
    if not feedback_logs:
        st.caption("No operator feedback actions submitted in this session.")
    else:
        st.dataframe(feedback_logs[::-1], use_container_width=True)
