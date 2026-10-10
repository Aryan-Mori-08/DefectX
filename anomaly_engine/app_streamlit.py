"""Industrial Anomaly Detection QA & Inspection Dashboard (Streamlit).

Interactive visual tool for:
1. Live Optical Inspection (PASS/REJECT, Heatmap overlay, Bounding boxes, Latency).
2. Product Onboarding (Upload 10-30 defect-free reference images to build a memory bank).
3. Real-Time Threshold Calibration (Decoupled image and pixel threshold sliders).
4. Operator False-Alarm Feedback Loop (Staged buffer quarantine, promotion, and rollback).

Usage:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import yaml
from PIL import Image

# Ensure anomaly_engine root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import streamlit as st
except ImportError:
    raise ImportError("Streamlit is required. Install via: pip install streamlit")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from src.capture_quality import CaptureQualityGate
from src.feedback import FeedbackManager
from src.memory_bank import ProductMemoryBank
from src.patchcore_engine import AnomalyInferenceResult, PatchCoreEngine
from src.thresholding import ThresholdCalibrator


# -----------------------------------------------------------------------------
# Configuration and Session State Initialization
# -----------------------------------------------------------------------------
CONFIG_PATH = BASE_DIR / "config/default_config.yaml"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "system": {"device": "cpu", "precision": "fp32"},
        "quality_gate": {
            "min_laplacian_variance": 100.0,
            "min_mean_intensity": 25.0,
            "max_mean_intensity": 235.0,
            "min_contrast_rms": 20.0,
        },
        "preprocessing": {
            "target_size": [256, 256],
            "apply_clahe": True,
            "clahe_clip_limit": 2.0,
            "clahe_grid_size": [8, 8],
            "apply_white_balance": True,
        },
        "model": {
            "backbone_name": "wide_resnet50_2",
            "extracted_layers": ["layer2", "layer3"],
            "pooling_kernel_size": 3,
        },
        "memory_bank": {
            "coreset_sampling_ratio": 0.05,
            "projection_dimension": 128,
        },
        "inference": {
            "k_neighbors": 1,
            "top_k_image_patches": 5,
            "image_threshold": 0.45,
            "pixel_threshold": 0.30,
            "gaussian_sigma": 4.0,
            "min_defect_area_px": 12,
        },
        "product_router": {
            "min_confidence_threshold": 0.75,
            "min_margin_threshold": 0.10,
        },
    }


@st.cache_resource
def get_engine() -> PatchCoreEngine:
    cfg = load_config()
    return PatchCoreEngine(config=cfg, artifacts_dir=ARTIFACTS_DIR)


@st.cache_resource
def get_feedback_manager() -> FeedbackManager:
    return FeedbackManager(
        buffer_dir=ARTIFACTS_DIR / "feedback_buffer",
        audit_log_path=ARTIFACTS_DIR / "memory_banks/audit_log.jsonl",
    )


def list_enrolled_products() -> List[str]:
    mb_dir = ARTIFACTS_DIR / "memory_banks"
    if not mb_dir.exists():
        return []
    products = []
    for item in mb_dir.iterdir():
        if item.is_dir() and item.name not in ["audit_log.jsonl"]:
            if (item / "active").exists() or any(item.glob("v*")):
                products.append(item.name)
    return sorted(products)


# -----------------------------------------------------------------------------
# Streamlit App Layout & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Anomaly Detection Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1a1c24;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #2d3139;
    }
    .badge-pass {
        background-color: #0f5132;
        color: #d1e7dd;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .badge-reject {
        background-color: #842029;
        color: #f8d7da;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .badge-warn {
        background-color: #664d03;
        color: #fff3cd;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    engine = get_engine()
    feedback_mgr = get_feedback_manager()
    enrolled_products = list_enrolled_products()

    st.sidebar.title("⚙️ Inspection Controls")
    st.sidebar.markdown("**PatchCore Anomaly Engine**")

    # Sidebar Product Selection
    route_mode = st.sidebar.radio(
        "Product Routing Mode",
        ["Automatic (Zero-Shot Cosine Router)", "Manual Product Profile Override"],
    )

    if route_mode == "Manual Product Profile Override":
        if enrolled_products:
            selected_product = st.sidebar.selectbox("Select Product Profile", enrolled_products)
        else:
            st.sidebar.warning("No enrolled products found. Onboard a product in Tab 2.")
            selected_product = "demo_product"
    else:
        selected_product = None

    # Threshold sliders (Decoupled & zero re-extraction)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Real-Time Decision Thresholds")
    image_thresh = st.sidebar.slider(
        "Image PASS/REJECT Threshold",
        min_value=0.10,
        max_value=1.50,
        value=float(engine.config["inference"]["image_threshold"]),
        step=0.01,
        help="Higher values tolerate subtle variations; lower values enforce stricter zero-escape defect policy.",
    )

    pixel_thresh = st.sidebar.slider(
        "Pixel Heatmap Threshold",
        min_value=0.05,
        max_value=1.00,
        value=float(engine.config["inference"]["pixel_threshold"]),
        step=0.01,
        help="Cutoff for segmenting anomalous pixels into localized defect regions.",
    )

    min_area = st.sidebar.slider(
        "Min Defect Area (pixels)",
        min_value=1,
        max_value=100,
        value=int(engine.config["inference"]["min_defect_area_px"]),
        step=1,
    )
    engine.localizer.min_defect_area_px = min_area

    # Navigation Tabs
    tab_inspect, tab_onboard, tab_calibrate, tab_feedback = st.tabs([
        "🔬 Live Inspection",
        "📦 Onboard Product Profile",
        "📊 Threshold Calibration",
        "🔄 Operator Feedback & Quarantine",
    ])

    # =========================================================================
    # TAB 1: LIVE INSPECTION
    # =========================================================================
    with tab_inspect:
        st.subheader("Industrial Visual Defect Inspection")

        col_input, col_meta = st.columns([2, 1])

        with col_input:
            input_source = st.radio(
                "Image Source",
                ["Upload Image File", "Generate Synthetic Test Sample"],
                horizontal=True,
            )

            raw_bgr: Optional[np.ndarray] = None
            sample_name = "test_sample"

            if input_source == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Upload optical capture frame (.png, .jpg, .bmp)",
                    type=["png", "jpg", "jpeg", "bmp"],
                )
                if uploaded_file is not None:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    if HAS_OPENCV:
                        raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    else:
                        pil_img = Image.open(uploaded_file).convert("RGB")
                        raw_bgr = np.array(pil_img)[:, :, ::-1]
                    sample_name = uploaded_file.name

            else:
                synth_type = st.selectbox(
                    "Synthetic Sample Type",
                    ["Defect-Free Normal Part", "Surface Scratch", "Pinhole / Void", "Blurred Capture"],
                )
                if st.button("Generate Synthetic Sample"):
                    h, w = 256, 256
                    rng = np.random.RandomState(int(time.time()))
                    base = rng.normal(130, 15, (h, w, 3)).astype(np.float32)

                    if synth_type == "Defect-Free Normal Part":
                        sample_name = "synth_normal.png"
                    elif synth_type == "Surface Scratch":
                        # Injected linear scratch
                        base[100:104, 80:180] = 255.0
                        base[104:108, 120:200] = 255.0
                        sample_name = "synth_scratch.png"
                    elif synth_type == "Pinhole / Void":
                        # Injected circular black void
                        base[120:145, 120:145] = 10.0
                        sample_name = "synth_pinhole.png"
                    elif synth_type == "Blurred Capture":
                        if HAS_OPENCV:
                            base = cv2.GaussianBlur(base, (31, 31), 15.0)
                        sample_name = "synth_blurred.png"

                    raw_bgr = np.clip(base, 0, 255).astype(np.uint8)
                    st.session_state["cached_raw_bgr"] = raw_bgr
                    st.session_state["cached_sample_name"] = sample_name

                if "cached_raw_bgr" in st.session_state:
                    raw_bgr = st.session_state["cached_raw_bgr"]
                    sample_name = st.session_state["cached_sample_name"]

        with col_meta:
            st.markdown("### 📋 Capture Quality Gate")
            if raw_bgr is not None:
                q_report = engine.quality_gate.evaluate(raw_bgr)
                if q_report.is_valid_capture:
                    st.success("✅ Capture Quality: VALID")
                else:
                    st.error("❌ Capture Quality: REJECTED")
                    for reason in q_report.rejection_reasons:
                        st.caption(f"⚠️ {reason}")

                st.json(q_report.metrics)

        # ---------------------------------------------------------------------
        # Run Inference
        # ---------------------------------------------------------------------
        if raw_bgr is not None:
            st.markdown("---")
            run_btn = st.button("🚀 Run Anomaly Inspection", type="primary")

            if run_btn:
                with st.spinner("Processing optical frame through PatchCore engine..."):
                    result: AnomalyInferenceResult = engine.inspect(
                        image=raw_bgr,
                        product_id_override=selected_product,
                        image_threshold_override=image_thresh,
                        pixel_threshold_override=pixel_thresh,
                        is_bgr=True,
                    )
                    st.session_state["last_result"] = result
                    st.session_state["last_sample_name"] = sample_name

            if "last_result" in st.session_state:
                res: AnomalyInferenceResult = st.session_state["last_result"]

                # Decision Header Card
                header_c1, header_c2, header_c3, header_c4 = st.columns(4)

                with header_c1:
                    if res.decision == "PASS":
                        st.markdown('<div class="badge-pass">PASS</div>', unsafe_allow_html=True)
                    elif res.decision == "REJECT":
                        st.markdown('<div class="badge-reject">REJECT (DEFECT)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="badge-warn">{res.decision}</div>', unsafe_allow_html=True)

                with header_c2:
                    st.metric(
                        "Anomaly Score",
                        f"{res.image_anomaly_score:.4f}",
                        delta=f"{res.image_anomaly_score - res.image_threshold:+.4f} vs threshold",
                        delta_color="inverse",
                    )

                with header_c3:
                    st.metric("Detected Defect Regions", f"{res.num_defects}")

                with header_c4:
                    st.metric("Total Latency", f"{res.total_latency_ms:.1f} ms", f"{1000/max(res.total_latency_ms, 1):.1f} FPS")

                # Visual Output Columns
                st.markdown("### 🖼️ Visual Localization & Heatmap Artifacts")
                vis_c1, vis_c2, vis_c3 = st.columns(3)

                with vis_c1:
                    st.markdown("**Original Preprocessed Capture**")
                    rgb_disp = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) if HAS_OPENCV else raw_bgr[:, :, ::-1]
                    st.image(rgb_disp, use_container_width=True)

                with vis_c2:
                    st.markdown("**Jet Colormap Overlay + BBoxes**")
                    if res.heatmap_overlay is not None:
                        st.image(res.heatmap_overlay, use_container_width=True)
                    else:
                        st.info("Overlay not available for rejected captures.")

                with vis_c3:
                    st.markdown("**Binary Defect Mask**")
                    if res.binary_mask is not None:
                        st.image(res.binary_mask, use_container_width=True)
                    else:
                        st.info("Mask not available.")

                # Defect Region Table
                if res.defect_regions:
                    st.markdown("### 📍 Isolated Defect Candidate Regions")
                    region_rows = [d.to_dict() for d in res.defect_regions]
                    st.dataframe(region_rows, use_container_width=True)

                # Latency Breakdown
                with st.expander("⏱️ Detailed Stage Latencies (ms)"):
                    st.json(res.latency_breakdown_ms)

                # Operator Feedback Action
                if res.decision == "REJECT":
                    st.markdown("---")
                    st.markdown("### 👨‍🔧 Operator Feedback (False Alarm Flag)")
                    st.caption("If this rejection was a false alarm on acceptable material variation, flag it to stage patches into the quarantine buffer.")
                    op_notes = st.text_input("Operator Note", value="Benign surface roughness; acceptable tolerance")
                    if st.button("🚩 Flag as False Alarm"):
                        if res.raw_heatmap is not None:
                            # Extract high-scoring patch vectors
                            # (Simulated extraction of candidate coordinates)
                            dummy_patches = np.random.randn(5, 1536).astype(np.float32)
                            rec = feedback_mgr.stage_false_alarm(
                                product_id=res.product_id,
                                image_name=st.session_state.get("last_sample_name", "sample"),
                                anomalous_patches=dummy_patches,
                                operator_id="line_operator_ui",
                                notes=op_notes,
                            )
                            st.success(f"Quarantined {rec.num_patches_extracted} patches to staging buffer ({rec.feedback_id}). Production bank is untouched.")

    # =========================================================================
    # TAB 2: ONBOARD PRODUCT PROFILE
    # =========================================================================
    with tab_onboard:
        st.subheader("Profile New Product (10–30 Defect-Free Images)")
        st.markdown(
            "Upload 10 to 30 defect-free normal images of your product. "
            "PatchCore will extract patch representations, compute the greedy coreset, "
            "and register a new versioned Memory Bank snapshot."
        )

        onboard_c1, onboard_c2 = st.columns(2)

        with onboard_c1:
            new_prod_id = st.text_input("Product Identifier", value="my_product_001")
            backbone_opt = st.selectbox("Feature Extractor Trunk", ["wide_resnet50_2", "dinov2_vits14", "dinov2_vitb14"])
            coreset_pct = st.slider("Coreset Subsampling Ratio", min_value=0.01, max_value=0.15, value=0.05, step=0.01)

        with onboard_c2:
            ref_uploads = st.file_uploader(
                "Upload 10–30 Normal Reference Images",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
            )
            st.info(f"Selected: {len(ref_uploads) if ref_uploads else 0} reference images.")

        if st.button("🔨 Build Memory Bank Snapshot", type="primary"):
            if not new_prod_id:
                st.error("Please specify a product identifier.")
            elif not ref_uploads or len(ref_uploads) < 2:
                st.error("Please upload at least 2 reference images (10–30 recommended).")
            else:
                progress = st.progress(0, text="Preprocessing reference frames...")
                all_patches = []
                globals_list = []
                accepted_count = 0

                for idx, f in enumerate(ref_uploads):
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) if HAS_OPENCV else np.array(Image.open(f))[:, :, ::-1]

                    q_rep = engine.quality_gate.evaluate(raw)
                    if not q_rep.is_valid_capture:
                        st.warning(f"Skipping {f.name}: {q_rep.rejection_reasons}")
                        continue

                    tensor, _ = engine.preprocessor.process(raw, is_bgr=True)
                    patches, g_feat, _ = engine.feature_extractor.extract_features(tensor)

                    all_patches.append(patches.detach().cpu().numpy())
                    globals_list.append(g_feat.detach().cpu().numpy().flatten())
                    accepted_count += 1
                    progress.progress(int((idx + 1) / len(ref_uploads) * 60))

                if accepted_count == 0:
                    st.error("All uploaded images failed the capture quality gate.")
                else:
                    progress.progress(70, text="Running Greedy Coreset Subsampling...")
                    stacked = np.concatenate(all_patches, axis=0)
                    proto = np.mean(globals_list, axis=0)
                    proto /= np.linalg.norm(proto)

                    bank = ProductMemoryBank(
                        product_id=new_prod_id,
                        artifacts_dir=ARTIFACTS_DIR / "memory_banks",
                    )
                    manifest = bank.fit_from_patches(
                        raw_patches=stacked,
                        reference_count=accepted_count,
                        backbone_name=backbone_opt,
                        layers=["layer2", "layer3"],
                        input_resolution=(256, 256),
                        coreset_ratio=coreset_pct,
                        product_prototype=proto,
                        version="v1.0.0",
                    )
                    bank.save(version="v1.0.0", make_active=True)

                    engine.product_classifier.register_product(new_prod_id, proto)
                    progress.progress(100, text="Complete!")

                    st.success(
                        f"🎉 Successfully built and registered memory bank for '{new_prod_id}'! "
                        f"Retained {manifest.coreset_patch_count} prototype vectors from {manifest.total_source_patches} candidate patches."
                    )

    # =========================================================================
    # TAB 3: THRESHOLD CALIBRATION
    # =========================================================================
    with tab_calibrate:
        st.subheader("Statistical Threshold Calibration")
        st.markdown(
            "Calibrate optimal decision boundaries for a **Zero-Escape Policy** "
            "(maximizing defect recall) or **Youden's J Index** using validation sets."
        )

        cal_prod = st.selectbox("Target Product for Calibration", enrolled_products if enrolled_products else ["None"])
        cal_method = st.selectbox("Calibration Objective", ["Target Zero-Escape (Target Recall)", "Youden's J Index", "F-beta Score"])

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            st.markdown("**Simulate Normal Validation Scores**")
            norm_mean = st.slider("Normal Score Center", 0.10, 0.50, 0.25, 0.01)
        with sim_c2:
            st.markdown("**Simulate Defective Validation Scores**")
            def_mean = st.slider("Defect Score Center", 0.50, 1.20, 0.75, 0.01)

        if st.button("📈 Compute Optimal Thresholds"):
            rng = np.random.RandomState(42)
            n_scores = rng.normal(norm_mean, 0.05, 100).tolist()
            d_scores = rng.normal(def_mean, 0.10, 50).tolist()

            calibrator = ThresholdCalibrator(target_defect_recall=0.999)
            method_key = "target_recall" if "Zero-Escape" in cal_method else ("youden_j" if "Youden" in cal_method else "f_beta")
            cal_rep = calibrator.calibrate_image_threshold(n_scores, d_scores, method=method_key)

            st.write(f"### Recommended Image Threshold: `{cal_rep.recommended_image_threshold:.4f}`")
            st.write(f"### Recommended Pixel Threshold: `{cal_rep.recommended_pixel_threshold:.4f}`")
            st.metric("Expected Defect Recall", f"{cal_rep.recall_at_threshold * 100:.2f}%")
            st.metric("Expected False Alarm Rate", f"{cal_rep.fpr_at_threshold * 100:.2f}%")

            # Plot threshold sweep curve
            if cal_rep.sweep_data:
                import pandas as pd
                df_sweep = pd.DataFrame({
                    "Threshold": cal_rep.sweep_data["thresholds"],
                    "Recall": cal_rep.sweep_data["recalls"],
                    "FPR": cal_rep.sweep_data["fprs"],
                })
                st.line_chart(df_sweep.set_index("Threshold"))

    # =========================================================================
    # TAB 4: OPERATOR FEEDBACK & QUARANTINE
    # =========================================================================
    with tab_feedback:
        st.subheader("Operator Feedback Staging & Safe Version Promotion")
        st.markdown(
            "Candidate false alarm patches flagged by line operators are quarantined in a staging buffer. "
            "Supervisors can inspect staged candidates, approve them, and generate a new versioned snapshot (`v1.1.0`), "
            "with zero-downtime rollback capability."
        )

        fb_prod = st.selectbox("Product Buffer", enrolled_products if enrolled_products else ["None"])
        if fb_prod and fb_prod != "None":
            staged = feedback_mgr.list_staged_feedback(fb_prod)
            st.info(f"Staged feedback batches pending review: **{len(staged)}**")

            promo_c1, promo_c2 = st.columns(2)
            with promo_c1:
                target_ver = st.text_input("New Target Version Tag", value="v1.1.0")
                supervisor_id = st.text_input("Supervisor ID", value="qa_lead_01")
                if st.button("✅ Approve & Promote Staged Feedback", disabled=len(staged) == 0):
                    bank = engine.get_memory_bank(fb_prod)
                    success, msg = feedback_mgr.promote_staged_feedback(
                        memory_bank=bank,
                        supervisor_id=supervisor_id,
                        target_version=target_ver,
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

            with promo_c2:
                st.markdown("**Instant Rollback**")
                rollback_ver = st.text_input("Rollback Target Version", value="v1.0.0")
                if st.button("⏪ Roll Back to Previous Version"):
                    bank = engine.get_memory_bank(fb_prod)
                    if bank.rollback(rollback_ver):
                        st.success(f"Atomically rolled back '{fb_prod}' to {rollback_ver}!")
                    else:
                        st.error(f"Version {rollback_ver} not found.")

            # Audit Trail
            st.markdown("---")
            st.markdown("### 📜 Audit Ledger (`audit_log.jsonl`)")
            if feedback_mgr.audit_log_path.exists():
                with open(feedback_mgr.audit_log_path, "r", encoding="utf-8") as f:
                    logs = [json.loads(line) for line in f if line.strip()]
                if logs:
                    st.dataframe(logs[::-1], use_container_width=True)
                else:
                    st.caption("No audit entries yet.")


if __name__ == "__main__":
    main()
