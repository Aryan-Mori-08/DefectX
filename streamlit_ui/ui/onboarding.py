"""Product Onboarding UI module.

Enables uploading 10-30 defect-free reference captures to profile a new product,
previews image thumbnails, enforces sample size guards, and invokes the
placeholder memory bank construction pipeline with progress indicators.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import List
from PIL import Image
import streamlit as st

from utils.model_interface import build_memory_bank


def render_onboarding_page() -> None:
    """Renders the product onboarding interface."""
    st.header("📦 Product Profile Onboarding")
    st.markdown(
        "Onboard a new product profile using **10–30 defect-free reference images**. "
        "The system constructs a representative coreset memory bank without needing defective samples."
    )

    col_form, col_info = st.columns([2, 1])

    with col_form:
        profile_name = st.text_input(
            "New Product Profile Identifier",
            placeholder="e.g. pcb_assembly_rev_c",
            help="Alphanumeric identifier for the new production line product.",
        )

        uploaded_files = st.file_uploader(
            "Upload 10–30 Defect-Free Reference Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="onboarding_uploader",
        )

    with col_info:
        st.markdown("### 📋 Onboarding Guidelines")
        st.markdown(
            """
            - **Defect-Free Only**: All reference images must be confirmed acceptable parts.
            - **Lighting Consistency**: Use standard production line illumination.
            - **Sample Size**:
              - Minimum: 10 images
              - Recommended: 20–30 images
            - **Variability**: Include typical acceptable cosmetic tolerances.
            """
        )

    # -------------------------------------------------------------------------
    # Image Count & Warning Verification
    # -------------------------------------------------------------------------
    if uploaded_files:
        num_images = len(uploaded_files)
        st.markdown(f"**Uploaded Reference Images:** `{num_images}`")

        if num_images < 10:
            st.warning(
                f"⚠️ Only {num_images} images uploaded. At least 10 defect-free images are "
                "required to construct a reliable coreset memory bank."
            )
        elif num_images > 30:
            st.info(
                f"ℹ️ {num_images} images uploaded. Coreset subsampling will automatically "
                "compress the memory footprint to ~1,024 prototype patches."
            )
        else:
            st.success(f"✅ Image count ({num_images}) satisfies the 10–30 reference standard.")

        # Preview Thumbnails in expandable section
        with st.expander("🖼️ Preview Reference Images", expanded=False):
            cols = st.columns(min(5, max(1, num_images)))
            for idx, file in enumerate(uploaded_files[:15]):  # Preview up to 15
                col = cols[idx % len(cols)]
                img = Image.open(file)
                col.image(img, caption=file.name, use_container_width=True)

        st.markdown("---")
        # ---------------------------------------------------------------------
        # Build Memory Bank Action
        # ---------------------------------------------------------------------
        build_btn = st.button("🔨 Build Memory Bank", type="primary")

        if build_btn:
            if not profile_name or not profile_name.strip():
                st.error("Please enter a valid product profile name.")
                return

            if num_images < 2:
                st.error("Please upload at least 2 reference images to initiate profiling.")
                return

            # Temporary file staging
            temp_paths: List[str] = []
            progress_bar = st.progress(0, text="Staging reference images...")

            try:
                temp_dir = tempfile.mkdtemp(prefix="patchcore_ref_")
                for i, file in enumerate(uploaded_files):
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    temp_paths.append(file_path)
                    progress_bar.progress(int((i + 1) / len(uploaded_files) * 30))

                progress_bar.progress(40, text="Extracting patch representations...")
                # Call placeholder memory bank function
                success = build_memory_bank(profile_name.strip(), temp_paths)

                progress_bar.progress(85, text="Applying greedy coreset subsampling...")
                progress_bar.progress(100, text="Memory bank snapshot serialized!")

                if success:
                    clean_name = profile_name.strip()
                    # Ensure profile is added to session state list
                    existing = st.session_state.get("enrolled_profiles", [])
                    if clean_name not in existing:
                        existing.append(clean_name)
                    st.session_state["enrolled_profiles"] = sorted(list(set(existing)))
                    st.session_state["selected_profile"] = clean_name

                    st.success(
                        f"🎉 Successfully built memory bank for profile `{clean_name}`! "
                        f"Profile is now active and selected in the system controls. Refreshing dashboard..."
                    )
                    time.sleep(1.2)
                    st.rerun()
                else:
                    st.error("Failed to build memory bank. Check log outputs.")

            except Exception as e:
                st.error(f"Error during memory bank construction: {str(e)}")
