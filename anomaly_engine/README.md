# PatchCore Industrial Anomaly Detection Engine

A product-agnostic, unsupervised anomaly detection system built specifically for high-speed industrial optical inspection lines. The engine profiles a product type using only **10–30 defect-free reference images**, eliminating the need for defective samples during initial onboarding.

---

## Key Capabilities

1. **Pre-Inference Capture Quality Gate**:
   - Rejects blurred captures via discrete Laplacian variance $\text{Var}(\nabla^2 I) < T_{\text{blur}}$.
   - Guards against underexposure, overexposure, specular reflections, and low-contrast illumination shifts before invoking neural backbones.

2. **Deterministic Optical Preprocessing**:
   - Gray-World white balancing for illumination color constancy.
   - Contrast-Limited Adaptive Histogram Equalization (CLAHE) in LAB color space.
   - Spatial resizing to configurable dimensions ($256\times 256$ or $384\times 384$) with ImageNet normalization.

3. **Multi-Scale Deep Feature Extraction**:
   - Standard industrial backbone: **WideResNet-50-2** extracting intermediate `layer2` and `layer3` representations.
   - Receptive field expansion via local neighborhood average pooling (`AvgPool2d(k=3, s=1, p=1)`).
   - Optional **DINOv2** Vision Transformer (`dinov2_vits14` / `dinov2_vitb14`) patch tokens for stochastic textures and textiles.

4. **Greedy Coreset Subsampling**:
   - Johnson-Lindenstrauss random projection to $d=128$ dimensions.
   - Minimax facility location algorithm retaining $1\%\text{--}5\%$ of the most representative patch vectors while discarding spatial redundancy.

5. **Nearest-Neighbor Querying**:
   - Direct FAISS `IndexFlatL2` acceleration with vectorized PyTorch CUDA / CPU distance matrix fallback.

6. **Defect Localization & Severity Quantification**:
   - Bilinear upsampling to native capture resolution with 2D Gaussian kernel smoothing ($\sigma = 4.0$).
   - Jet colormap overlay blending ($\alpha = 0.45$).
   - Connected component analysis isolating bounding boxes `[x, y, w, h]`, centroids, area, and logarithmic severity index:
     $$\text{Severity}(C_k) = \frac{\max_{(x,y) \in C_k} H(x,y)}{T_{\text{pixel}}} \cdot \log_{10}(1 + \text{Area}(C_k))$$

7. **Zero-Shot Product Routing**:
   - Global L2-normalized image descriptor cosine similarity against prototype centroids.
   - Automatic routing with fallback to `UNKNOWN_PRODUCT` on low confidence or narrow margins. Manual override supported.

8. **Dual-Threshold Calibration**:
   - Completely decoupled image threshold $T_{\text{image}}$ (PASS/REJECT) and pixel threshold $T_{\text{pixel}}$ (segmentation).
   - Optimizable via Target Recall (Zero Escape Policy) or Youden's J index in $O(1)$ time without re-extracting features.

9. **Operator False-Alarm Feedback Loop**:
   - Quarantined staging buffer storing candidate false alarm patches.
   - Defect contamination safety checks.
   - Atomic version promotion (`v1.0.0` $\to$ `v1.1.0`) with immediate one-step rollback.

---

## Directory Layout

```text
anomaly_engine/
├── config/
│   └── default_config.yaml           # Centralized configuration
├── data/
│   ├── reference/                    # 10-30 defect-free images per product
│   └── validation/                   # Split into good/ and defect/
├── artifacts/
│   ├── memory_banks/                 # Versioned snapshots & audit_log.jsonl
│   └── models/                       # Product prototype registry
├── src/
│   ├── __init__.py
│   ├── capture_quality.py            # Optical sharpness & exposure gate
│   ├── preprocessing.py              # CLAHE, White-balance, Normalization
│   ├── feature_extractor.py          # WideResNet-50-2 & DINOv2
│   ├── memory_bank.py                # Coreset sampler & nearest-neighbor index
│   ├── patchcore_engine.py           # Master inference orchestrator
│   ├── product_classifier.py         # Zero-shot prototype router
│   ├── localization.py               # Heatmap synthesis & connected components
│   ├── thresholding.py               # Decoupled threshold calibration
│   ├── feedback.py                   # Staged false-alarm quarantine & rollback
│   └── evaluation.py                 # AUROC, AUPR, F1, FPR/FNR, latency benchmarks
├── scripts/
│   ├── build_memory_bank.py          # Profile product from 10-30 normal images
│   ├── run_inference.py              # CLI batch/single image inspector
│   ├── evaluate_model.py             # Validation benchmark reporter
│   └── calibrate_thresholds.py       # Threshold calibration script
└── tests/                            # PyTest unit verification suite
```

---

## Quickstart Guide

### 1. Requirements Installation
```bash
pip install torch torchvision numpy opencv-python scikit-learn pyyaml scipy
# Optional for maximum GPU index speed:
pip install faiss-gpu  # or faiss-cpu
```

### 2. Build a Product Memory Bank (Onboarding 10–30 Normal Images)
```bash
python scripts/build_memory_bank.py \
    --product pcb_rev_b \
    --ref-dir data/reference/pcb_rev_b/ \
    --version v1.0.0
```

### 3. Run Inference on Test Images
```bash
python scripts/run_inference.py \
    --image path/to/sample.png \
    --product pcb_rev_b \
    --save-vis \
    --output-dir inspection_results/
```

### 4. Calibrate Decision Thresholds
```bash
python scripts/calibrate_thresholds.py \
    --product pcb_rev_b \
    --val-dir data/validation/pcb_rev_b/ \
    --method target_recall \
    --target-recall 0.999 \
    --update-bank
```

### 5. Evaluate Accuracy & Throughput
```bash
python scripts/evaluate_model.py \
    --product pcb_rev_b \
    --val-dir data/validation/pcb_rev_b/ \
    --report-file evaluation_report.json
```

### 6. Run Unit Test Suite
```bash
pytest tests/ -v
```
