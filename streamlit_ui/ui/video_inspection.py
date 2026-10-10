"""Video-based live inspection page UI module.

Provides real-time video stream inspection with continuous frame processing,
live statistics, and defect detection visualization.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

from utils.image_utils import bgr_to_rgb
from utils.model_interface import (
    check_capture_quality,
    extract_defect_regions,
    generate_heatmap_overlay,
    predict,
    preprocess_image,
)
from utils.video_stream import FrameInspectionResult, VideoAnnotator


class AnomalyDetectionTransformer(VideoTransformerBase):
    """Video transformer for real-time anomaly detection on webcam stream."""

    def __init__(self):
        self.frame_count = 0
        self.process_every_n = 3  # Process every 3rd frame for performance
        self.selected_profile = None  # Will be set from sidebar selection
        self.image_threshold = 0.40
        self.pixel_threshold = 0.25
        self.min_defect_area = 2
        self.latest_result: Optional[Dict[str, Any]] = None
        self.annotator = VideoAnnotator()

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each video frame through the anomaly detection pipeline."""
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        # Process only every Nth frame to maintain performance
        if self.frame_count % self.process_every_n != 0:
            # Draw cached results if available
            if self.latest_result is not None:
                img = self._draw_cached_result(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Run inspection pipeline
        try:
            start_time = time.time()

            # 1. Quality gate check
            quality_report = check_capture_quality(img)

            if quality_report["status"] != "PASS":
                # Draw quality fail overlay
                img = self.annotator.draw_inspection_overlay(
                    img,
                    decision="QUALITY_FAIL",
                    anomaly_score=0.0,
                    num_defects=0,
                    latency_ms=0.0,
                    quality_status="FAIL",
                )
                self.latest_result = {
                    "decision": "QUALITY_FAIL",
                    "score": 0.0,
                    "defects": 0,
                    "latency": 0.0,
                }
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # 2. Preprocess and run inference
            proc_img = preprocess_image(img)
            score, anomaly_map, latency, _ = predict(proc_img, self.selected_profile)

            # 3. Extract defect regions
            _, defect_regions = extract_defect_regions(
                anomaly_map=anomaly_map,
                pixel_threshold=self.pixel_threshold,
                min_area=self.min_defect_area,
            )

            # 4. Determine decision
            is_defect = (score >= self.image_threshold) or (len(defect_regions) > 0)
            decision = "REJECT" if is_defect else "PASS"

            end_time = time.time()
            total_latency = (end_time - start_time) * 1000

            # Cache result
            self.latest_result = {
                "decision": decision,
                "score": score,
                "defects": len(defect_regions),
                "latency": total_latency,
                "regions": defect_regions,
            }

            # 5. Annotate frame
            img = self.annotator.draw_inspection_overlay(
                img,
                decision=decision,
                anomaly_score=score,
                num_defects=len(defect_regions),
                latency_ms=total_latency,
                quality_status="PASS",
            )

            if defect_regions:
                img = self.annotator.draw_defect_regions(img, defect_regions)

        except Exception as e:
            # Draw error overlay
            import cv2
            cv2.putText(
                img, f"Error: {str(e)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _draw_cached_result(self, img: np.ndarray) -> np.ndarray:
        """Draw the most recent inspection result on frame."""
        return self.annotator.draw_inspection_overlay(
            img,
            decision=self.latest_result["decision"],
            anomaly_score=self.latest_result["score"],
            num_defects=self.latest_result["defects"],
            latency_ms=self.latest_result["latency"],
            quality_status="PASS",
        )


def render_video_inspection_page(
    selected_profile: str,
    image_threshold: float,
    pixel_threshold: float,
    min_defect_area: int,
) -> None:
    """Renders the live video inspection interface."""
    st.header(f"📹 Live Video Inspection - Profile: `{selected_profile}`")
    st.caption("Real-time anomaly detection on continuous video stream")

    # Video stream controls
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 🎥 Live Camera Feed")
        st.info("Grant camera permission when prompted. The system will analyze frames in real-time.")

    with col2:
        st.markdown("### ⚙️ Stream Settings")
        process_rate = st.select_slider(
            "Processing Rate",
            options=["Every Frame", "Every 2nd", "Every 3rd", "Every 5th"],
            value="Every 3rd",
            help="Process fewer frames to improve performance"
        )

        show_heatmap = st.checkbox("Show Heatmap Overlay", value=False)

    st.markdown("---")

    # WebRTC streamer for live video
    webrtc_ctx = webrtc_streamer(
        key="anomaly-detection-stream",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=AnomalyDetectionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Update transformer settings when sidebar changes
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.selected_profile = selected_profile
        webrtc_ctx.video_transformer.image_threshold = image_threshold
        webrtc_ctx.video_transformer.pixel_threshold = pixel_threshold
        webrtc_ctx.video_transformer.min_defect_area = min_defect_area

    st.markdown("---")

    # Live statistics dashboard
    st.subheader("📊 Live Inspection Statistics")

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Stream Status", "🟢 ACTIVE" if webrtc_ctx.state.playing else "🔴 STOPPED")

    with stat_col2:
        if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.latest_result:
            result = webrtc_ctx.video_transformer.latest_result
            st.metric("Latest Decision", result["decision"])

    with stat_col3:
        if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.latest_result:
            result = webrtc_ctx.video_transformer.latest_result
            st.metric("Anomaly Score", f"{result['score']:.4f}")

    with stat_col4:
        if webrtc_ctx.video_transformer:
            st.metric("Frames Processed", webrtc_ctx.video_transformer.frame_count)

    # Instructions
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Click 'START' above** to begin the video stream
    2. **Grant camera permission** when your browser requests it
    3. **Position the product** in front of the camera
    4. **Watch real-time results** overlaid on the video feed
    5. Green banner = PASS, Red banner = REJECT (defect detected)
    6. Adjust thresholds in the sidebar to fine-tune sensitivity
    """)

    # Technical notes
    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
        - **Active Profile**: `{selected_profile}`
        - **Image Threshold**: {image_threshold:.4f}
        - **Pixel Threshold**: {pixel_threshold:.4f}
        - **Min Defect Area**: {min_defect_area} pixels
        - **Processing Mode**: Real-time frame-by-frame analysis
        - **Quality Gating**: Automatic blur/exposure/contrast checks
        """)
