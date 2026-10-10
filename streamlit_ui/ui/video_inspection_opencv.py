"""Alternative video inspection using OpenCV and st.camera_input for video frames.

This version uses native Streamlit components without streamlit-webrtc dependency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from utils.image_utils import bgr_to_rgb, draw_bounding_boxes, pil_to_bgr
from utils.model_interface import (
    check_capture_quality,
    extract_defect_regions,
    generate_heatmap_overlay,
    predict,
    preprocess_image,
)


def render_video_inspection_opencv(
    selected_profile: str,
    image_threshold: float,
    pixel_threshold: float,
    min_defect_area: int,
) -> None:
    """Renders video inspection using OpenCV camera capture with Streamlit."""

    if selected_profile is None:
        st.error("⚠️ No product profile selected. Please onboard a product first in the '📦 2. Product Onboarding' tab.")
        return

    st.header(f"📹 Live Video Inspection - Profile: `{selected_profile}`")
    st.caption("Real-time anomaly detection using continuous camera feed")

    # Instructions
    st.info("📸 **How to use:** Use the camera input below to continuously capture and inspect frames. Click 'Take photo' repeatedly for live inspection.")

    # Video processing controls
    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### ⚙️ Settings")
        auto_refresh = st.checkbox("Auto-refresh", value=False, help="Automatically refresh the page for continuous capture")

        if auto_refresh:
            refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 3)
            st.caption(f"Page will refresh every {refresh_rate} seconds")

    with col1:
        st.markdown("### 🎥 Camera Feed")

        # Camera input
        camera_photo = st.camera_input("Capture frame for inspection")

        if camera_photo is not None:
            # Convert to numpy array
            pil_img = Image.open(camera_photo)
            raw_bgr = pil_to_bgr(pil_img)

            # Start processing
            with st.spinner("🔍 Processing frame..."):
                start_time = time.time()

                # 1. Quality gate check
                quality_report = check_capture_quality(raw_bgr)

                # Display quality metrics
                q_col1, q_col2, q_col3, q_col4 = st.columns(4)
                with q_col1:
                    st.metric("Blur Score", f"{quality_report['blur_score']:.1f}")
                with q_col2:
                    st.metric("Brightness", f"{quality_report['brightness_score']:.1f}")
                with q_col3:
                    st.metric("Contrast", f"{quality_report['contrast_score']:.1f}")
                with q_col4:
                    if quality_report["status"] == "PASS":
                        st.success("Quality: ✅ PASS")
                    else:
                        st.warning("Quality: ⚠️ FAIL")

                # Check if quality passed
                if quality_report["status"] != "PASS":
                    st.error("❌ **QUALITY CHECK FAILED** - Retake photo with better lighting/focus")
                    for reason in quality_report.get("rejection_reasons", []):
                        st.caption(f"• {reason}")
                else:
                    # 2. Run anomaly detection
                    proc_img = preprocess_image(raw_bgr)
                    score, anomaly_map, latency, _ = predict(proc_img, selected_profile)

                    # 3. Extract defects
                    binary_mask, defect_regions = extract_defect_regions(
                        anomaly_map=anomaly_map,
                        pixel_threshold=pixel_threshold,
                        min_area=min_defect_area,
                    )

                    # 4. Generate visualizations
                    heatmap_overlay = generate_heatmap_overlay(proc_img, anomaly_map, alpha=0.45)
                    proc_rgb = bgr_to_rgb(proc_img)
                    annotated_image = draw_bounding_boxes(proc_rgb, defect_regions)

                    # 5. Determine decision
                    is_defect = (score >= image_threshold) or (len(defect_regions) > 0)
                    decision = "REJECT" if is_defect else "PASS"

                    end_time = time.time()
                    total_latency = (end_time - start_time) * 1000

                    # Display decision banner
                    if decision == "PASS":
                        st.success("✅ **PASS** - No defects detected")
                    else:
                        st.error("❌ **REJECT** - Defect detected!")

                    # Display metrics
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    with m_col1:
                        st.metric("Anomaly Score", f"{score:.4f}")
                    with m_col2:
                        st.metric("Threshold", f"{image_threshold:.4f}")
                    with m_col3:
                        st.metric("Defect Regions", len(defect_regions))
                    with m_col4:
                        st.metric("Total Latency", f"{total_latency:.0f} ms")

                    # 4-way visual display
                    st.markdown("---")
                    st.subheader("🔬 Inspection Results")

                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.markdown("**1. Original Frame**")
                        st.image(proc_rgb, use_container_width=True)

                        st.markdown("**3. Binary Anomaly Mask**")
                        st.image(binary_mask, use_container_width=True, clamp=True)

                    with v_col2:
                        st.markdown("**2. Heatmap Overlay**")
                        st.image(heatmap_overlay, use_container_width=True)

                        st.markdown("**4. Defect Localization**")
                        st.image(annotated_image, use_container_width=True)

                    # Defect details
                    if defect_regions:
                        st.markdown("#### 📍 Detected Defects")
                        st.dataframe(defect_regions, use_container_width=True)

        else:
            st.info("👆 Click 'Take photo' above to capture and inspect a frame")

    # Auto-refresh logic
    if auto_refresh and camera_photo is not None:
        time.sleep(refresh_rate)
        st.rerun()

    # Statistics section
    st.markdown("---")
    st.subheader("📊 Session Statistics")

    # Initialize session state for statistics
    if "video_stats" not in st.session_state:
        st.session_state["video_stats"] = {
            "total_frames": 0,
            "pass_count": 0,
            "reject_count": 0,
            "quality_fail_count": 0,
        }

    # Update stats if frame was processed
    if camera_photo is not None:
        stats = st.session_state["video_stats"]

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Frames", stats["total_frames"])
        with stat_col2:
            st.metric("Pass", stats["pass_count"], delta_color="normal")
        with stat_col3:
            st.metric("Reject", stats["reject_count"], delta_color="inverse")
        with stat_col4:
            pass_rate = (stats["pass_count"] / stats["total_frames"] * 100) if stats["total_frames"] > 0 else 0
            st.metric("Pass Rate", f"{pass_rate:.1f}%")

    # Instructions
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Click 'Take photo'** above to capture a frame from your camera
    2. **Wait for processing** - the system will analyze the frame
    3. **View results** - Green = PASS, Red = REJECT
    4. **Enable 'Auto-refresh'** for continuous inspection (experimental)
    5. **Adjust thresholds** in the sidebar to fine-tune sensitivity

    **Note:** This mode processes one frame at a time. For true real-time video streaming,
    install `streamlit-webrtc` by running: `pip install streamlit-webrtc av`
    """)
