"""Industrial Defect Inspection Assistant - Streamlit Application Entrypoint.

Main orchestrator providing sidebar controls (Profile selector, Dynamic threshold sliders,
Device health indicator) and multi-tab workflows (Inspection, Onboarding, Results, Feedback).
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure streamlit_ui directory is at the top of sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ui.feedback import render_feedback_page
from ui.inspection import render_inspection_page
from ui.onboarding import render_onboarding_page
from ui.results import render_results_page
from utils.model_interface import list_product_profiles, load_product_profile
from utils.session_state import init_session_state


# Page configuration
st.set_page_config(
    page_title="Industrial Defect Inspection Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar():
    """Renders persistent global controls in the Streamlit sidebar."""
    st.sidebar.title("🛠️ System Controls")
    st.sidebar.markdown("**PatchCore Anomaly Engine**")

    # 1. Product Profile Selection
    disk_profiles = list_product_profiles()
    state_profiles = st.session_state.get("enrolled_profiles", [])
    merged_profiles = sorted(list(set(disk_profiles + state_profiles)))
    if not merged_profiles:
        merged_profiles = ["pcb_rev_b", "metal_casting_part_a", "injection_mold_04"]
    st.session_state["enrolled_profiles"] = merged_profiles

    current_selected = st.session_state.get("selected_profile")
    if not current_selected or current_selected not in merged_profiles:
        current_selected = merged_profiles[0]
        st.session_state["selected_profile"] = current_selected

    selected_idx = merged_profiles.index(current_selected) if current_selected in merged_profiles else 0

    selected_profile = st.sidebar.selectbox(
        "Product Profile",
        merged_profiles,
        index=selected_idx,
        help="Select the product profile memory bank to test against.",
    )
    st.session_state["selected_profile"] = selected_profile

    # Load product profile on change
    load_product_profile(selected_profile)

    st.sidebar.markdown("---")

    # 2. Dynamic Threshold Controls (Triggering instant re-evaluations without re-running inference)
    st.sidebar.subheader("🎯 Dynamic Thresholds")
    st.sidebar.caption("Modifying sliders recomputes decisions instantly without re-running model inference.")

    image_threshold = st.sidebar.slider(
        "Image Threshold (PASS/REJECT)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Global distance threshold: units scoring above this value are marked REJECT.",
    )

    pixel_threshold = st.sidebar.slider(
        "Pixel Threshold (Segmentation)",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
        help="Pixel anomaly threshold for generating the binary mask.",
    )

    min_defect_area = st.sidebar.slider(
        "Min Defect Area (pixels)",
        min_value=1,
        max_value=500,
        value=15,
        step=1,
        help="Defect regions smaller than this area are filtered out as optical noise.",
    )

    st.sidebar.markdown("---")

    # 3. Model & Engine Status
    st.sidebar.subheader("🔌 Engine Status")
    st.sidebar.success("● Engine: ONLINE")
    st.sidebar.info(f"Active Profile: `{selected_profile}`")
    st.sidebar.caption("Inference Mode: Zero-Re-Inference Reactive Cache")

    return selected_profile, image_threshold, pixel_threshold, min_defect_area


def main():
    # Initialize session state cache
    init_session_state()

    # Application Header
    st.title("Industrial Defect Inspection Assistant")
    st.caption("Product-agnostic few-shot PatchCore anomaly detection for real-time visual quality assurance.")

    # Render Sidebar
    selected_profile, image_thresh, pixel_thresh, min_area = render_sidebar()

    # Main Navigation Tabs
    tab_inspect, tab_onboard, tab_results, tab_feedback = st.tabs([
        "🔬 1. Inspection",
        "📦 2. Product Onboarding",
        "📊 3. Results & Yield",
        "🔄 4. Operator Feedback",
    ])

    with tab_inspect:
        render_inspection_page(
            selected_profile=selected_profile,
            image_threshold=image_thresh,
            pixel_threshold=pixel_thresh,
            min_defect_area=min_area,
        )

    with tab_onboard:
        render_onboarding_page()

    with tab_results:
        render_results_page()

    with tab_feedback:
        render_feedback_page(selected_profile=selected_profile)


if __name__ == "__main__":
    main()
