"""Real-time video stream handler for continuous anomaly detection.

Provides webcam/USB camera streaming with frame-by-frame quality gating
and anomaly inference, supporting live industrial inspection workflows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from collections import deque


@dataclass
class FrameInspectionResult:
    """Result from a single frame inspection."""
    frame_id: int
    timestamp: float
    decision: str  # "PASS" | "REJECT" | "QUALITY_FAIL"
    anomaly_score: float
    num_defects: int
    latency_ms: float
    quality_metrics: Dict[str, float]


class VideoStreamProcessor:
    """Handles live video stream capture and frame-by-frame processing.

    Attributes:
        camera_id: Camera device index (0 for default webcam)
        target_fps: Target frame processing rate
        skip_frames: Process every Nth frame to reduce load
        result_buffer_size: Number of recent results to keep in memory
    """

    def __init__(
        self,
        camera_id: int = 0,
        target_fps: int = 10,
        skip_frames: int = 2,
        result_buffer_size: int = 100,
    ):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.skip_frames = skip_frames
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        self.frame_count = 0
        self.results_buffer: deque = deque(maxlen=result_buffer_size)

    def start_stream(self) -> bool:
        """Initialize video capture device."""
        if self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        self.is_streaming = True
        self.frame_count = 0
        return True

    def stop_stream(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_streaming = False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the video stream.

        Returns:
            BGR numpy array if successful, None otherwise.
        """
        if not self.is_streaming or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_count += 1
        return frame

    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on skip_frames."""
        return (self.frame_count % self.skip_frames) == 0

    def add_result(self, result: FrameInspectionResult) -> None:
        """Add inspection result to the circular buffer."""
        self.results_buffer.append(result)

    def get_recent_results(self, n: int = 10) -> List[FrameInspectionResult]:
        """Get the N most recent inspection results."""
        return list(self.results_buffer)[-n:]

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate aggregate statistics from recent results."""
        if not self.results_buffer:
            return {
                "total_frames": 0,
                "pass_count": 0,
                "reject_count": 0,
                "quality_fail_count": 0,
                "pass_rate": 0.0,
                "avg_anomaly_score": 0.0,
                "avg_latency_ms": 0.0,
            }

        results = list(self.results_buffer)
        total = len(results)
        pass_count = sum(1 for r in results if r.decision == "PASS")
        reject_count = sum(1 for r in results if r.decision == "REJECT")
        quality_fail_count = sum(1 for r in results if r.decision == "QUALITY_FAIL")

        avg_score = np.mean([r.anomaly_score for r in results])
        avg_latency = np.mean([r.latency_ms for r in results])

        return {
            "total_frames": total,
            "pass_count": pass_count,
            "reject_count": reject_count,
            "quality_fail_count": quality_fail_count,
            "pass_rate": (pass_count / total * 100) if total > 0 else 0.0,
            "avg_anomaly_score": float(avg_score),
            "avg_latency_ms": float(avg_latency),
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_stream()


class VideoAnnotator:
    """Utility class for annotating video frames with inspection results."""

    @staticmethod
    def draw_inspection_overlay(
        frame: np.ndarray,
        decision: str,
        anomaly_score: float,
        num_defects: int,
        latency_ms: float,
        quality_status: str = "OK",
    ) -> np.ndarray:
        """Draw inspection information overlay on frame.

        Args:
            frame: BGR image array
            decision: "PASS" | "REJECT" | "QUALITY_FAIL"
            anomaly_score: Anomaly score value
            num_defects: Number of detected defects
            latency_ms: Processing latency
            quality_status: Quality gate status

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Decision banner at top
        if decision == "PASS":
            color = (0, 255, 0)  # Green
            text = "PASS"
        elif decision == "REJECT":
            color = (0, 0, 255)  # Red
            text = "REJECT - DEFECT DETECTED"
        else:
            color = (0, 165, 255)  # Orange
            text = "QUALITY CHECK FAILED"

        cv2.rectangle(annotated, (0, 0), (w, 60), color, -1)
        cv2.putText(
            annotated, text, (20, 40),
            cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3
        )

        # Metrics overlay at bottom
        overlay_y = h - 100
        cv2.rectangle(annotated, (0, overlay_y), (w, h), (0, 0, 0), -1)

        metrics_text = [
            f"Score: {anomaly_score:.4f}",
            f"Defects: {num_defects}",
            f"Latency: {latency_ms:.1f}ms",
            f"Quality: {quality_status}",
        ]

        x_offset = 20
        for i, text in enumerate(metrics_text):
            cv2.putText(
                annotated, text, (x_offset + i * 250, overlay_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return annotated

    @staticmethod
    def draw_defect_regions(
        frame: np.ndarray,
        defect_regions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw bounding boxes around detected defect regions.

        Args:
            frame: BGR image array
            defect_regions: List of defect region dictionaries with x, y, w, h keys

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for i, region in enumerate(defect_regions):
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("w", 0))
            h = int(region.get("h", 0))

            # Draw red bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Draw label
            label = f"Defect {i+1}"
            cv2.putText(
                annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        return annotated
