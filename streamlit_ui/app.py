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
from utils.auth import (
    init_session_state as init_auth_state,
    render_login_page,
    render_navbar,
    logout,
    check_permission,
    render_user_management
)


# Page configuration with custom theme
st.set_page_config(
    page_title="DefectX - AI-Powered Visual Inspection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_custom_css():
    """Load custom CSS styling based on reference design."""
    css_file = Path(__file__).parent / "assets" / "custom_theme.css"
    if css_file.exists():
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()


def render_sidebar():
    """Renders persistent global controls in the Streamlit sidebar."""
    st.sidebar.markdown("# 🛠️ System Controls")
    st.sidebar.caption("**PatchCore Anomaly Engine**")

    # 1. Product Profile Selection
    disk_profiles = list_product_profiles()
    state_profiles = st.session_state.get("enrolled_profiles", [])
    merged_profiles = sorted(list(set(disk_profiles + state_profiles)))
    if not merged_profiles:
        merged_profiles = []  # Empty - no default products
    st.session_state["enrolled_profiles"] = merged_profiles

    current_selected = st.session_state.get("selected_profile")
    if not current_selected or current_selected not in merged_profiles:
        current_selected = merged_profiles[0] if merged_profiles else None
        st.session_state["selected_profile"] = current_selected

    if not merged_profiles:
        st.sidebar.warning("⚠️ No product profiles found. Please onboard a product first.")
        selected_profile = None
    else:
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

    # 2. Dynamic Threshold Controls
    st.sidebar.markdown("### 🎯 Dynamic Thresholds")
    st.sidebar.caption("Adjust sliders to recompute decisions instantly without re-running inference.")

    image_threshold = st.sidebar.slider(
        "Image Threshold (PASS/REJECT)",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.01,
        help="Global distance threshold: units scoring above this value are marked REJECT.",
    )

    pixel_threshold = st.sidebar.slider(
        "Pixel Threshold (Segmentation)",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Pixel anomaly threshold for generating the binary mask.",
    )

    min_defect_area = st.sidebar.slider(
        "Min Defect Area (pixels)",
        min_value=1,
        max_value=500,
        value=2,
        step=1,
        help="Defect regions smaller than this area are filtered out as optical noise.",
    )

    st.sidebar.markdown("---")

    # 3. Engine Status
    st.sidebar.markdown("### 🔌 Engine Status")
    if selected_profile:
        st.sidebar.success("● Engine: ONLINE")
        st.sidebar.info(f"Active Profile: `{selected_profile}`")
    else:
        st.sidebar.error("● Engine: NO PROFILE")
        st.sidebar.warning("Please onboard a product")

    st.sidebar.caption("Inference Mode: Zero-Re-Inference Reactive Cache")

    return selected_profile, image_threshold, pixel_threshold, min_defect_area


def main():
    # Initialize both session states
    init_auth_state()
    init_session_state()

    # Check authentication
    if not st.session_state.authenticated:
        render_login_page()
        return

    # Render navbar for authenticated users
    render_navbar()

    # Logout button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            logout()

    # Application Header with brand styling
    st.markdown("""
        <div style="padding: 20px 0; margin-bottom: 30px;">
            <h1 style="margin: 0; display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 48px;">◈</span>
                <span>Defect<span style="color: #ff791b;">X</span></span>
            </h1>
            <p style="color: #64748b; font-size: 13px; margin-top: 8px; letter-spacing: 2.5px; font-weight: 600;">
                DETECT&nbsp; | &nbsp;INSPECT&nbsp; | &nbsp;ENSURE
            </p>
            <p style="color: #64748b; font-size: 14px; margin-top: 12px;">
                Product-agnostic few-shot PatchCore anomaly detection for real-time visual quality assurance.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Render Sidebar
    selected_profile, image_thresh, pixel_thresh, min_area = render_sidebar()

    # Main Navigation Tabs based on user role
    user_role = st.session_state.user["role"]

    if user_role == "admin":
        # Admin has access to all tabs + user management
        tabs = st.tabs([
            "🔬 Inspection",
            "📦 Product Onboarding",
            "📊 Results & Yield",
            "🔄 Operator Feedback",
            "👥 User Management"
        ])

        with tabs[0]:
            render_inspection_page(
                selected_profile=selected_profile,
                image_threshold=image_thresh,
                pixel_threshold=pixel_thresh,
                min_defect_area=min_area,
            )

        with tabs[1]:
            render_onboarding_page()

        with tabs[2]:
            render_results_page()

        with tabs[3]:
            render_feedback_page(selected_profile=selected_profile)

        with tabs[4]:
            render_user_management()

    elif user_role == "operator":
        # Operator has access to inspection, results, and feedback
        tabs = st.tabs([
            "🔬 Inspection",
            "📦 Product Onboarding",
            "📊 Results & Yield",
            "🔄 Operator Feedback"
        ])

        with tabs[0]:
            render_inspection_page(
                selected_profile=selected_profile,
                image_threshold=image_thresh,
                pixel_threshold=pixel_thresh,
                min_defect_area=min_area,
            )

        with tabs[1]:
            render_onboarding_page()

        with tabs[2]:
            render_results_page()

        with tabs[3]:
            render_feedback_page(selected_profile=selected_profile)

    else:  # employee
        # Employee has access to inspection only
        tabs = st.tabs([
            "🔬 Inspection",
            "📊 Results View"
        ])

        with tabs[0]:
            render_inspection_page(
                selected_profile=selected_profile,
                image_threshold=image_thresh,
                pixel_threshold=pixel_thresh,
                min_defect_area=min_area,
            )

        with tabs[1]:
            render_results_page()


if __name__ == "__main__":
    main()
