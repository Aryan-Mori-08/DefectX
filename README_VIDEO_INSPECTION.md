# Live Video Inspection Feature

## Overview
The DEFECTX system now supports **real-time video stream inspection** for continuous anomaly detection on live camera feeds. This feature complements the existing snapshot and file upload modes.

## Features

### 🎥 Real-Time Video Processing
- **Continuous frame analysis** - Process video frames in real-time from webcam/USB camera
- **Adaptive frame skipping** - Configurable processing rate (every frame, every 2nd, 3rd, or 5th frame)
- **Live quality gating** - Automatic blur, exposure, and contrast checks on each frame
- **Instant feedback** - Visual overlays showing PASS/REJECT decisions directly on video

### 🔍 Detection Capabilities
- **Frame-by-frame anomaly scoring** - Each frame analyzed independently
- **Defect localization** - Bounding boxes drawn around detected defects in real-time
- **Quality metrics display** - Live statistics overlaid on video stream
- **Decision tracking** - Historical buffer of recent inspection results

### 📊 Live Statistics Dashboard
- Stream status (active/stopped)
- Latest inspection decision
- Current anomaly score
- Total frames processed
- Pass/reject/quality-fail counts

## Installation

Install the additional required packages:

```bash
pip install streamlit-webrtc>=0.47.0 av>=10.0.0
```

Or install from the updated requirements file:

```bash
pip install -r streamlit_ui/requirements.txt
```

## Usage

### Starting the Application

```bash
streamlit run streamlit_ui/app.py
```

### Using Live Video Inspection

1. **Navigate to Inspection Tab** - Open the "🔬 1. Inspection" tab
2. **Select Live Video Mode** - Click on the "📹 Live Video Stream" tab
3. **Grant Camera Permission** - Allow your browser to access the camera when prompted
4. **Click START** - Begin the video stream
5. **Position Product** - Place the product in front of the camera
6. **View Real-Time Results** - Watch inspection results overlaid on the video

### Visual Indicators

- **🟢 Green Banner** - PASS (no defects detected)
- **🔴 Red Banner** - REJECT (defect detected with bounding boxes)
- **🟠 Orange Banner** - QUALITY CHECK FAILED (improve lighting/focus)

### Adjusting Sensitivity

Use the sidebar controls to fine-tune detection:
- **Image Threshold** - Overall anomaly score cutoff
- **Pixel Threshold** - Per-pixel anomaly sensitivity
- **Min Defect Area** - Minimum defect size to report

## Architecture

### New Components

#### `streamlit_ui/utils/video_stream.py`
- `VideoStreamProcessor` - Manages video capture and frame buffering
- `FrameInspectionResult` - Data class for inspection results
- `VideoAnnotator` - Utilities for drawing overlays and bounding boxes

#### `streamlit_ui/ui/video_inspection.py`
- `AnomalyDetectionTransformer` - WebRTC video transformer for real-time processing
- `render_video_inspection_page()` - UI rendering for video inspection mode

### Modified Components

#### `streamlit_ui/ui/inspection.py`
- Added third tab for "Live Video Stream"
- Integrated video inspection page rendering
- Maintained backward compatibility with snapshot and upload modes

## Configuration

### Video Stream Settings

Modify in `streamlit_ui/ui/video_inspection.py`:

```python
# In AnomalyDetectionTransformer.__init__()
self.process_every_n = 3  # Process every Nth frame (1-10)
```

### Camera Resolution

Modify in `streamlit_ui/utils/video_stream.py`:

```python
# In VideoStreamProcessor.start_stream()
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Default: 1280
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Default: 720
```

## Performance Optimization

### Frame Rate Control
- **Every Frame** - Highest accuracy, highest CPU load (~30 FPS processing)
- **Every 2nd Frame** - Balanced performance (~15 FPS processing)
- **Every 3rd Frame** - Recommended default (~10 FPS processing)
- **Every 5th Frame** - Lower load for slower systems (~6 FPS processing)

### Memory Management
- Results buffer maintains last 100 inspection results
- Automatic frame caching prevents re-processing between model runs
- Circular buffer prevents memory overflow

## Quality Gate Integration

The live video inspection uses the same quality gate thresholds defined in:
- `anomaly_engine/config/default_config.yaml`
- `anomaly_engine/src/capture_quality.py`

Updated thresholds (stricter):
- Laplacian variance: 50.0 (sharpness)
- Mean intensity: 30.0 - 235.0 (exposure range)
- RMS contrast: 15.0 (minimum contrast)
- Max overexposed pixels: 5%
- Max underexposed pixels: 5%

## Troubleshooting

### Camera Not Detected
- Ensure camera is connected and not in use by another application
- Grant browser camera permissions when prompted
- Try a different browser (Chrome/Edge recommended)

### Low Frame Rate
- Reduce processing rate (use "Every 5th" frame)
- Lower camera resolution in `video_stream.py`
- Ensure sufficient CPU/GPU resources

### False Rejections
- Adjust sidebar thresholds (increase image threshold)
- Improve lighting conditions
- Check quality gate metrics

### Quality Check Failures
- Improve lighting (avoid overexposure)
- Ensure proper focus (reduce blur)
- Check contrast levels

## Technical Details

### Processing Pipeline
1. **Frame Capture** - WebRTC captures frame from browser camera
2. **Quality Gate** - Blur, exposure, contrast checks
3. **Preprocessing** - CLAHE, white balance, resizing
4. **Feature Extraction** - PatchCore embedding generation
5. **Anomaly Scoring** - Memory bank nearest neighbor search
6. **Defect Localization** - Heatmap thresholding and region extraction
7. **Visualization** - Overlay generation and bounding box drawing

### Latency Breakdown
- Frame capture: ~1-5ms
- Quality checks: ~2-5ms
- Preprocessing: ~10-20ms
- Model inference: ~50-150ms (GPU) / ~200-500ms (CPU)
- Post-processing: ~5-10ms
- **Total**: ~70-190ms per frame (GPU) / ~220-540ms (CPU)

## Future Enhancements

- [ ] Multi-camera support
- [ ] Video recording with annotations
- [ ] Real-time statistics export
- [ ] Automated alert triggers
- [ ] Batch video file processing
- [ ] GPU acceleration for preprocessing
- [ ] RTSP camera stream support

## Comparison: Video vs Snapshot Mode

| Feature | Live Video | Snapshot | File Upload |
|---------|-----------|----------|-------------|
| **Real-time** | ✅ Yes | ⚠️ Manual | ❌ No |
| **Continuous** | ✅ Yes | ❌ No | ❌ No |
| **Statistics** | ✅ Live | ❌ No | ❌ No |
| **Defect Tracking** | ✅ Yes | ❌ No | ❌ No |
| **Best For** | Production line | Quality audit | Batch analysis |

---

**Last Updated**: 2026-10-10  
**Version**: 1.0.0  
**Requires**: streamlit-webrtc>=0.47.0, av>=10.0.0
