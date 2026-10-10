"""Enhanced inspection page with DefectX design styling."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import streamlit as st

from utils.image_utils import bgr_to_rgb, draw_bounding_boxes, pil_to_bgr
from utils.model_interface import (
    check_capture_quality,
    extract_defect_regions,
    generate_heatmap_overlay,
    predict,
    preprocess_image,
)
from utils.session_state import log_inspection_result


def render_decision_badge(decision: str) -> None:
    """Renders a prominent styled decision badge."""
    if decision == "PASS":
        st.markdown(
            """
            <div style="
                background: #e7f7f0;
                color: #12805c;
                padding: 18px 24px;
                border-radius: 10px;
                font-size: 22px;
                font-weight: 800;
                text-align: center;
                margin: 20px 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 4px 15px rgba(18, 128, 92, 0.1);
            ">
                <div>
                    <small style="font-size: 9px; letter-spacing: 1px; font-weight: 800;">INSPECTION STATUS</small>
                    <h2 style="margin: 5px 0 0 0; font-size: 22px;">PASS - NO DEFECTS</h2>
                </div>
                <span style="font-size: 32px;">✓</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif decision == "REJECT":
        st.markdown(
            """
            <div style="
                background: #fff0ed;
                color: #c83939;
                padding: 18px 24px;
                border-radius: 10px;
                font-size: 22px;
                font-weight: 800;
                text-align: center;
                margin: 20px 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 4px 15px rgba(200, 57, 57, 0.1);
            ">
                <div>
                    <small style="font-size: 9px; letter-spacing: 1px; font-weight: 800;">INSPECTION STATUS</small>
                    <h2 style="margin: 5px 0 0 0; font-size: 22px;">REJECT - DEFECT DETECTED</h2>
                </div>
                <span style="font-size: 32px;">!</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background: #fff9e6;
                color: #b8860b;
                padding: 18px 24px;
                border-radius: 10px;
                font-size: 22px;
                font-weight: 800;
                text-align: center;
                margin: 20px 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <div>
                    <small style="font-size: 9px; letter-spacing: 1px; font-weight: 800;">INSPECTION STATUS</small>
                    <h2 style="margin: 5px 0 0 0; font-size: 22px;">LOW QUALITY - RETRY CAPTURE</h2>
                </div>
                <span style="font-size: 32px;">⚠</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_inspection_page(
    selected_profile: str,
    image_threshold: float,
    pixel_threshold: float,
    min_defect_area: int,
) -> None:
    """Renders the main industrial inspection interface with DefectX styling."""

    # Header with styling
    st.markdown(f"""
        <div style="margin-bottom: 25px;">
            <p style="color: #104a8a; font-size: 11px; font-weight: 800; letter-spacing: 2.5px; margin-bottom: 8px;">
                INSPECTION WORKSPACE
            </p>
            <h2 style="margin: 0;">Optical Inspection</h2>
            <p style="color: #64748b; font-size: 13px; margin-top: 8px;">
                Active Profile: <code>{selected_profile if selected_profile else 'None'}</code>
            </p>
        </div>
    """, unsafe_allow_html=True)

    if selected_profile is None:
        st.error("⚠️ No product profile selected. Please onboard a product first in the '📦 Product Onboarding' tab.")
        return

    # -------------------------------------------------------------------------
    # Input Selection: Live Video, Camera Snapshot, or File Upload
    # -------------------------------------------------------------------------
    input_tab1, input_tab2, input_tab3 = st.tabs([
        "📹 Live Video Stream",
        "📷 Camera Snapshot",
        "📁 File Upload"
    ])

    # Tab 1: Live Video Inspection
    with input_tab1:
        try:
            # Try to use WebRTC version if available
            from ui.video_inspection import render_video_inspection_page
            render_video_inspection_page(
                selected_profile=selected_profile,
                image_threshold=image_threshold,
                pixel_threshold=pixel_threshold,
                min_defect_area=min_defect_area,
            )
        except ImportError:
            # Fall back to OpenCV version if streamlit-webrtc is not installed
            st.warning("⚠️ Full real-time video streaming requires `streamlit-webrtc`. Using simplified camera mode.")
            st.info("💡 For full features, run: `pip install streamlit-webrtc av`")
            from ui.video_inspection_opencv import render_video_inspection_opencv
            render_video_inspection_opencv(
                selected_profile=selected_profile,
                image_threshold=image_threshold,
                pixel_threshold=pixel_threshold,
                min_defect_area=min_defect_area,
            )
        return  # Exit early when video mode is active

    # -------------------------------------------------------------------------
    # Existing Snapshot and Upload Modes (Tab 2 and 3)
    # -------------------------------------------------------------------------
    input_tab1, input_tab2 = input_tab2, input_tab3  # Reassign for compatibility

    raw_bgr: Optional[np.ndarray] = None
    sample_name = "live_capture"

    with input_tab1:
        st.markdown("### 📷 Camera Capture")
        cam_frame = st.camera_input("Take a snapshot of the unit under test")
        if cam_frame is not None:
            pil_img = Image.open(cam_frame)
            raw_bgr = pil_to_bgr(pil_img)
            sample_name = "camera_snapshot.jpg"

    with input_tab2:
        st.markdown("### 📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload an inspection image (PNG, JPG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="inspection_uploader",
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            raw_bgr = pil_to_bgr(pil_img)
            sample_name = uploaded_file.name

    # -------------------------------------------------------------------------
    # Optical Quality Check & Inference Execution
    # -------------------------------------------------------------------------
    if raw_bgr is not None:
        st.markdown("---")
        st.markdown("### 🔬 Quality Gate Check")

        # Evaluate capture quality
        quality_report = check_capture_quality(raw_bgr)

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            st.metric("Blur Score (Laplacian)", f"{quality_report['blur_score']:.1f}")
        with q_col2:
            st.metric("Brightness", f"{quality_report['brightness_score']:.1f}")
        with q_col3:
            st.metric("Contrast (RMS)", f"{quality_report['contrast_score']:.1f}")
        with q_col4:
            if quality_report["status"] == "PASS":
                st.success("✅ Quality: PASS")
            else:
                st.warning("⚠️ Quality: FAIL")

        if quality_report["status"] != "PASS":
            render_decision_badge("LOW QUALITY CAPTURE")
            st.warning("⚠️ Capture failed quality checks. Please adjust lighting, focus, or camera alignment.")
            for reason in quality_report.get("rejection_reasons", []):
                st.caption(f"• {reason}")
            return

        # ---------------------------------------------------------------------
        # Anomaly Inference
        # ---------------------------------------------------------------------
        cached_bgr = st.session_state.get("current_image_bgr")
        cached_profile = st.session_state.get("current_profile")
        need_inference = (
            cached_bgr is None
            or not np.array_equal(cached_bgr, raw_bgr)
            or cached_profile != selected_profile
        )

        if need_inference:
            with st.spinner("🔍 Extracting patch embeddings and querying memory bank..."):
                proc_bgr = preprocess_image(raw_bgr)
                score, anomaly_map, latency, used_profile = predict(proc_bgr, selected_profile)

                st.session_state["current_image_bgr"] = raw_bgr
                st.session_state["current_preprocessed_bgr"] = proc_bgr
                st.session_state["current_anomaly_map"] = anomaly_map
                st.session_state["current_anomaly_score"] = score
                st.session_state["current_latency_ms"] = latency
                st.session_state["current_profile"] = used_profile
                st.session_state["current_sample_name"] = sample_name

    # -------------------------------------------------------------------------
    # Dynamic Threshold Post-Processing
    # -------------------------------------------------------------------------
    if st.session_state.get("current_anomaly_map") is not None:
        anomaly_map = st.session_state["current_anomaly_map"]
        score = st.session_state["current_anomaly_score"]
        latency = st.session_state["current_latency_ms"]
        proc_bgr = st.session_state["current_preprocessed_bgr"]

        binary_mask, defect_regions = extract_defect_regions(
            anomaly_map=anomaly_map,
            pixel_threshold=pixel_threshold,
            min_area=min_defect_area,
        )

        heatmap_overlay = generate_heatmap_overlay(proc_bgr, anomaly_map, alpha=0.45)
        proc_rgb = bgr_to_rgb(proc_bgr)
        annotated_image = draw_bounding_boxes(proc_rgb, defect_regions)

        is_defect = (score >= image_threshold) or (len(defect_regions) > 0)
        decision = "REJECT" if is_defect else "PASS"
        st.session_state["current_decision"] = decision

        # ---------------------------------------------------------------------
        # Results Display
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📊 Inspection Results")

        render_decision_badge(decision)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(
                "Anomaly Score",
                f"{score:.4f}",
                delta=f"{score - image_threshold:+.4f} vs threshold",
                delta_color="inverse",
            )
        with m_col2:
            st.metric("Image Threshold", f"{image_threshold:.4f}")
        with m_col3:
            st.metric("Detected Defects", f"{len(defect_regions)}")
        with m_col4:
            st.metric("Processing Time", f"{latency:.1f} ms")

        # 4-Way Visual Display
        st.markdown("---")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("**1. Original Preprocessed Frame**")
            st.image(proc_rgb, use_container_width=True)

            st.markdown("**3. Binary Anomaly Mask**")
            st.image(binary_mask, use_container_width=True, clamp=True)

        with v_col2:
            st.markdown("**2. Continuous Heatmap Overlay**")
            st.image(heatmap_overlay, use_container_width=True)

            st.markdown("**4. Annotated Bounding Boxes**")
            st.image(annotated_image, use_container_width=True)

        if defect_regions:
            st.markdown("#### 📍 Isolated Defect Candidates")
            st.dataframe(defect_regions, use_container_width=True)

        if st.button("💾 Save Inspection to Log", type="primary"):
            log_inspection_result(
                profile=selected_profile,
                score=score,
                decision=decision,
                latency_ms=latency,
                num_regions=len(defect_regions),
                image_name=st.session_state.get("current_sample_name", "capture"),
            )
            st.success("✅ Successfully logged inspection record to session ledger.")
