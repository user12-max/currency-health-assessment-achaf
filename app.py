"""ACHAF — Adaptive Currency Health Assessment Framework.

Main Streamlit Research Dashboard for Currency Note Physical Condition
Assessment and Spatial Damage Localization using Texture, Edge, Color, and
Morphological Digital Image Processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    ANALYSIS_MODES,
    CORE_MODULE_NAME,
    DAMAGE_CATEGORY_LABELS,
    DAMAGE_COLOR_LEGEND_RGB,
    DEFAULT_CHI_WEIGHTS,
    FRAMEWORK_NAME,
    MAX_FILE_SIZE_MB,
    PROTOTYPE_DISCLAIMER,
    SUPPORTED_DENOMINATIONS,
    SUPPORTED_SIDES,
    AnalysisSettings,
)
from src.evaluation import (
    evaluate_mask_against_ground_truth,
    generate_synthetic_test_note,
)
from src.image_io import load_and_validate_image
from src.pipeline import ACHAFPipelineOutput, run_achaf_pipeline
from src.report_generation import (
    build_component_scores_dataframe,
    build_indicators_dataframe,
    encode_rgb_to_png_bytes,
    generate_csv_summary,
    generate_json_report,
    generate_pdf_report,
)
from src.visualization import create_chi_gauge_chart, create_component_scores_chart

logger = logging.getLogger(__name__)


def _inject_custom_css() -> None:
    """Apply clean, modern research-dashboard styling."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }
        .achaf-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
            color: #F8FAFC;
            padding: 1.35rem 1.75rem;
            border-radius: 10px;
            margin-bottom: 1.25rem;
            border-left: 6px solid #38BDF8;
        }
        .achaf-header h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .achaf-header p {
            margin: 0.35rem 0 0 0;
            font-size: 0.98rem;
            color: #CBD5E1;
        }
        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            height: 100%;
        }
        .kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .kpi-sub {
            font-size: 0.80rem;
            color: #475569;
            margin-top: 0.3rem;
        }
        .legend-box {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0.65rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.85rem;
            font-weight: 500;
            color: #1E293B;
        }
        .legend-swatch {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            display: inline-block;
            border: 1px solid rgba(0,0,0,0.25);
        }
        .disclaimer-banner {
            background-color: #F1F5F9;
            border-left: 4px solid #64748B;
            padding: 0.7rem 1rem;
            border-radius: 4px;
            font-size: 0.84rem;
            color: #334155;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    """Initialize Streamlit session state keys safely."""
    if "achaf_output" not in st.session_state:
        st.session_state["achaf_output"] = None
    if "active_sample_mode" not in st.session_state:
        st.session_state["active_sample_mode"] = "None (Upload Image)"


def _reset_analysis_state() -> None:
    """Clear stored analysis output from session state."""
    st.session_state["achaf_output"] = None


def _render_color_legend() -> None:
    """Render the consistent 4-category damage localization color legend."""
    items_html = []
    for key, label in DAMAGE_CATEGORY_LABELS.items():
        r, g, b = DAMAGE_COLOR_LEGEND_RGB[key]
        items_html.append(
            f'<div class="legend-item">'
            f'<span class="legend-swatch" style="background-color: rgb({r},{g},{b});"></span>'
            f"<span>{label}</span>"
            f"</div>"
        )
    st.markdown(
        f'<div class="legend-box"><strong>Damage Overlay Color Legend:</strong> {"".join(items_html)}</div>',
        unsafe_allow_html=True,
    )


def _category_badge_color(category: str) -> str:
    """Return hex color associated with a CHI category label."""
    mapping = {
        "Excellent": "#16A34A",
        "Good": "#2563EB",
        "Fair": "#D97706",
        "Poor": "#EA580C",
        "Critical": "#DC2626",
    }
    return mapping.get(category, "#64748B")


def main() -> None:
    """Run the ACHAF Streamlit application."""
    st.set_page_config(
        page_title="ACHAF — Currency Note Health Assessment",
        page_icon="💵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_css()
    _init_session_state()

    # Page Header
    st.markdown(
        """
        <div class="achaf-header">
            <h1>ACHAF — Currency Note Health Assessment</h1>
            <p>Adaptive Multi-Feature Analysis for Physical Damage Assessment and Localization</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("1. Note Image Input")

        input_source_choice = st.radio(
            "Input Source",
            options=[
                "Upload Currency Image",
                "Synthetic Specimen: Clean Note",
                "Synthetic Specimen: Damaged Note",
                "Synthetic Specimen: Angled & Damaged",
            ],
            index=0,
            help=(
                "Upload a real currency note photograph (PNG, JPG, JPEG, WebP) or load a "
                "deterministic synthetic test specimen to inspect pipeline behavior."
            ),
        )

        uploaded_file = None
        if input_source_choice == "Upload Currency Image":
            uploaded_file = st.file_uploader(
                f"Upload Currency Note (Max {MAX_FILE_SIZE_MB:.0f} MB)",
                type=["png", "jpg", "jpeg", "webp"],
                key="primary_note_uploader",
                help="Accepts PNG, JPG, JPEG, and WebP photographs of genuine currency notes.",
            )

        st.subheader("2. Note Metadata & Mode")
        col_den, col_side = st.columns(2)
        with col_den:
            selected_denomination = st.selectbox(
                "Denomination",
                options=SUPPORTED_DENOMINATIONS,
                index=0,
            )
        with col_side:
            selected_side = st.selectbox(
                "Visible Side",
                options=SUPPORTED_SIDES,
                index=0,
            )

        selected_mode = st.selectbox(
            "Analysis Mode",
            options=ANALYSIS_MODES,
            index=0,
            help=(
                "Feature-based analysis evaluates a single note image. "
                "Reference-assisted analysis aligns the test note against a cleaner note "
                "of the same denomination, side, and design."
            ),
        )

        reference_file = None
        use_synthetic_ref = False
        if selected_mode == "Reference-assisted analysis":
            st.info(
                "**Reference Image Guidance:** Upload a cleaner note of the **exact same "
                "denomination, visible side, and series design**. Reference comparison can be "
                "unreliable if the images cannot be geometrically aligned via ORB+RANSAC; "
                "if alignment fails, ACHAF automatically falls back to feature-based analysis."
            )
            if input_source_choice.startswith("Synthetic Specimen"):
                use_synthetic_ref = st.checkbox(
                    "Use Clean Synthetic Specimen as Reference",
                    value=True,
                )
            if not use_synthetic_ref:
                reference_file = st.file_uploader(
                    "Upload Cleaner Reference Note (Optional)",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="reference_note_uploader",
                )

        # Expandable Advanced Settings
        with st.expander("Advanced Settings", expanded=False):
            st.markdown("##### Preprocessing")
            denoise_method = st.selectbox(
                "Denoising Filter",
                options=["bilateral", "gaussian", "median", "none"],
                index=0,
                help="Bilateral filtering preserves thin crease edges while reducing sensor noise.",
            )
            denoise_strength = st.slider(
                "Denoising Kernel Size", min_value=3, max_value=9, value=5, step=2
            )
            clahe_clip = st.slider(
                "CLAHE Contrast Clip Limit", min_value=1.0, max_value=4.5, value=2.0, step=0.25
            )
            apply_illum = st.checkbox("Apply Illumination Normalization", value=True)

            st.markdown("##### Segmentation & Manual Crop Fallback")
            enable_auto_seg = st.checkbox(
                "Automatic Note Segmentation & Perspective Rectification",
                value=True,
                help="Uncheck to use manual crop margins if background clutter prevents contour detection.",
            )
            canny_low = st.slider("Canny Low Threshold", 20, 100, 40, 5)
            canny_high = st.slider("Canny High Threshold", 80, 220, 140, 5)

            st.caption("Manual Crop Margins (% of frame to trim from each side):")
            mc1, mc2 = st.columns(2)
            with mc1:
                crop_top = st.slider("Top Margin %", 0, 40, 0, 1) / 100.0
                crop_left = st.slider("Left Margin %", 0, 40, 0, 1) / 100.0
            with mc2:
                crop_bottom = st.slider("Bottom Margin %", 0, 40, 0, 1) / 100.0
                crop_right = st.slider("Right Margin %", 0, 40, 0, 1) / 100.0

            st.markdown("##### Feature Sensitivity Multipliers")
            edge_sens = st.slider("Edge & Corner Sensitivity", 0.5, 2.0, 1.0, 0.1)
            fold_sens = st.slider("Fold & Crease Sensitivity", 0.5, 2.0, 1.0, 0.1)
            tex_sens = st.slider("Texture Irregularity Sensitivity", 0.5, 2.0, 1.0, 0.1)
            stain_sens = st.slider("Stain & Color Sensitivity", 0.5, 2.0, 1.0, 0.1)

            st.markdown("##### CHI Component Weights")
            w_edge = st.slider("Edge Weight", 0.05, 0.60, DEFAULT_CHI_WEIGHTS["edge"], 0.05)
            w_fold = st.slider("Fold Weight", 0.05, 0.60, DEFAULT_CHI_WEIGHTS["fold"], 0.05)
            w_tex = st.slider("Texture Weight", 0.05, 0.60, DEFAULT_CHI_WEIGHTS["texture"], 0.05)
            w_col = st.slider("Color/Stain Weight", 0.05, 0.60, DEFAULT_CHI_WEIGHTS["color"], 0.05)
            w_cor = st.slider("Corner Weight", 0.05, 0.60, DEFAULT_CHI_WEIGHTS["corner"], 0.05)

        settings_obj = AnalysisSettings(
            denoise_method=denoise_method,
            denoise_strength=denoise_strength,
            clahe_clip_limit=clahe_clip,
            apply_illumination_norm=apply_illum,
            enable_auto_segmentation=enable_auto_seg,
            canny_low_threshold=canny_low,
            canny_high_threshold=canny_high,
            manual_crop_margins=(crop_top, crop_bottom, crop_left, crop_right),
            edge_sensitivity=edge_sens,
            fold_sensitivity=fold_sens,
            texture_sensitivity=tex_sens,
            stain_sensitivity=stain_sens,
            chi_weights={
                "edge": w_edge,
                "fold": w_fold,
                "texture": w_tex,
                "color": w_col,
                "corner": w_cor,
            },
        )

        st.divider()
        analyze_clicked = st.button(
            "🔍 Analyze Currency Note",
            type="primary",
            width="stretch",
        )
        reset_clicked = st.button(
            "🗑️ Reset / Clear Analysis",
            width="stretch",
        )
        if reset_clicked:
            _reset_analysis_state()
            st.rerun()

    # -----------------------------------------------------------------------
    # Resolve Current Input Image for Preview & Execution
    # -----------------------------------------------------------------------
    active_input_source = None
    active_ref_source = None
    is_synthetic_specimen = False

    if input_source_choice == "Upload Currency Image":
        active_input_source = uploaded_file
        active_ref_source = reference_file
    elif input_source_choice == "Synthetic Specimen: Clean Note":
        active_input_source, _ = generate_synthetic_test_note(variant="clean")
        is_synthetic_specimen = True
    elif input_source_choice == "Synthetic Specimen: Damaged Note":
        active_input_source, _ = generate_synthetic_test_note(variant="damaged")
        is_synthetic_specimen = True
    elif input_source_choice == "Synthetic Specimen: Angled & Damaged":
        active_input_source, _ = generate_synthetic_test_note(variant="angled_damaged")
        is_synthetic_specimen = True

    if selected_mode == "Reference-assisted analysis":
        if use_synthetic_ref and is_synthetic_specimen:
            active_ref_source, _ = generate_synthetic_test_note(variant="clean")
        else:
            active_ref_source = reference_file

    # Show input preview before analysis
    if active_input_source is None:
        st.info(
            "👈 **Get Started:** Upload a currency note photograph (PNG, JPG, JPEG, or WebP) "
            "in the sidebar—or select one of the built-in synthetic test specimens—adjust "
            "any optional parameters, and click **Analyze Currency Note**."
        )
        with st.expander("About the ACHAF Pipeline & Methodology", expanded=True):
            st.markdown(
                """
                **ACHAF (Adaptive Currency Health Assessment Framework)** evaluates physical
                deterioration indicators on currency notes using explainable Digital Image Processing:
                1. **Image Acquisition & Capture Quality**: Format/EXIF validation, Laplacian blur variance, exposure clipping, and illumination uniformity checks.
                2. **Preprocessing**: Aspect-preserving scaling, edge-preserving bilateral denoising, Lab L-channel illumination normalization, and CLAHE enhancement.
                3. **Note Segmentation & Rectification**: Multi-cue foreground extraction, 4-corner quadrilateral fitting, perspective homography rectification, and manual-crop fallback.
                4. **Multi-Feature Extraction**:
                   - **Edge & Corner Analysis**: Canny edges, Sobel gradients, 4-edge continuity/roughness, and 4-corner area retention.
                   - **Morphological Analysis**: Scale-aware top-hat, black-hat, morphological gradient, and directional line opening + probabilistic Hough line detection for fold/crease candidates.
                   - **Texture Analysis**: Patch-based Gray Level Co-occurrence Matrix (GLCM) and uniform Local Binary Patterns (LBP).
                   - **Color & Stain Analysis**: Illumination-normalized HSV and CIE Lab local chromatic/luminance anomaly detection.
                5. **Adaptive Multi-Feature Fusion Module (AMFM) & Currency Health Index (CHI)**: Combines normalized component condition scores (`0–100`) while decoupling poor photographic quality from physical damage penalties.
                """
            )
        return

    # Validate & preview uploaded image prior to running full pipeline
    preview_load = load_and_validate_image(active_input_source)
    if not preview_load.is_valid or preview_load.rgb_image is None:
        st.error(f"**Input Validation Error:** {preview_load.error_message}")
        return

    if st.session_state["achaf_output"] is None and not analyze_clicked:
        st.subheader("Uploaded Input Preview (Ready for Analysis)")
        if is_synthetic_specimen:
            st.warning(
                "**Synthetic Test Specimen Selected:** This is a deterministic synthetic image "
                "generated for testing image-processing functions, not a real currency note."
            )
        pcol1, pcol2 = st.columns([2, 1])
        with pcol1:
            st.image(
                preview_load.rgb_image,
                caption=f"Input Image ({preview_load.width}×{preview_load.height} px, Format: {preview_load.format_name}, Mode: {preview_load.original_mode})",
                width="stretch",
            )
        with pcol2:
            st.markdown("#### Input Image Properties")
            st.write(f"- **Dimensions:** `{preview_load.width} × {preview_load.height}` px")
            st.write(f"- **Color Mode:** `{preview_load.original_mode}`")
            st.write(f"- **Denomination:** `{selected_denomination}`")
            st.write(f"- **Visible Side:** `{selected_side}`")
            st.write(f"- **Analysis Mode:** `{selected_mode}`")
            if preview_load.warnings:
                for pw in preview_load.warnings:
                    st.warning(pw)
            st.info("Click **🔍 Analyze Currency Note** in the sidebar to run the full pipeline.")
        return

    # -----------------------------------------------------------------------
    # Execute Pipeline When Button Clicked
    # -----------------------------------------------------------------------
    if analyze_clicked:
        with st.spinner("Running ACHAF multi-feature image processing pipeline..."):
            try:
                pipeline_out = run_achaf_pipeline(
                    image_source=active_input_source,
                    denomination=selected_denomination,
                    note_side=selected_side,
                    analysis_mode=selected_mode,
                    reference_source=active_ref_source,
                    settings=settings_obj,
                )
                st.session_state["achaf_output"] = pipeline_out
            except Exception as exc:
                logger.exception("Unhandled error in ACHAF pipeline")
                st.error(
                    f"An unexpected error occurred during image analysis: {exc}. "
                    "Please verify your input image and settings."
                )
                return

    output: Optional[ACHAFPipelineOutput] = st.session_state.get("achaf_output")
    if output is None:
        return

    if not output.is_success:
        st.error(f"**Analysis Failed:** {output.error_message}")
        return

    # Unpack validated pipeline outputs
    q_rep = output.quality_report
    pre_res = output.preprocessing_result
    seg_res = output.segmentation_result
    ref_res = output.reference_result
    tex_res = output.texture_result
    edge_res = output.edge_result
    morph_res = output.morphology_result
    col_res = output.color_result
    indicators = output.indicators
    amfm_res = output.amfm_result
    chi_res = output.chi_result
    loc_res = output.localization_result
    report_dict = output.report_dict

    assert (
        q_rep is not None
        and pre_res is not None
        and seg_res is not None
        and ref_res is not None
        and tex_res is not None
        and edge_res is not None
        and morph_res is not None
        and col_res is not None
        and indicators is not None
        and amfm_res is not None
        and chi_res is not None
        and loc_res is not None
        and report_dict is not None
    )

    if is_synthetic_specimen:
        st.info(
            "🧪 **Synthetic Verification Mode:** Results below are computed live by the "
            "ACHAF pipeline on a synthetic test pattern (not real-world currency data)."
        )

    # -----------------------------------------------------------------------
    # Status Banners & Capture Quality Warnings
    # -----------------------------------------------------------------------
    if q_rep.is_provisional or chi_res.is_provisional:
        st.warning(
            "⚠️ **Provisional Assessment Notice:** Capture quality or missing feature "
            "channels limit analysis reliability. Review the specific warnings below."
        )

    if not seg_res.segmentation_succeeded:
        st.warning(
            f"⚠️ **Segmentation Fallback Active:** {seg_res.status_message} "
            "You can adjust **Manual Crop Margins** under *Advanced Settings* in the sidebar "
            "or upload an already cropped note image."
        )

    if selected_mode == "Reference-assisted analysis":
        if ref_res.alignment_succeeded:
            st.success(f"✅ **Reference-Assisted Mode Active:** {ref_res.status_message}")
        else:
            st.warning(f"⚠️ **Reference Registration Fallback:** {ref_res.status_message}")

    # -----------------------------------------------------------------------
    # Top KPI Dashboard Cards
    # -----------------------------------------------------------------------
    cat_color = _category_badge_color(chi_res.health_category)
    chi_str = f"{chi_res.chi_score:.1f} / 100" if chi_res.chi_score is not None else "Withheld"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid {cat_color};">
                <div class="kpi-label">Currency Health Index (CHI)</div>
                <div class="kpi-value">{chi_str}</div>
                <div class="kpi-sub">{chi_res.scoring_approach}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        prov_badge = " (Provisional)" if chi_res.is_provisional else ""
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid {cat_color};">
                <div class="kpi-label">Estimated Condition Category</div>
                <div class="kpi-value" style="color: {cat_color};">{chi_res.health_category}{prov_badge}</div>
                <div class="kpi-sub">Prototype academic classification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        n_det = sum(1 for ind in indicators.values() if ind.is_detected)
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid #3B82F6;">
                <div class="kpi-label">Detected Damage Indicators</div>
                <div class="kpi-value">{n_det} / {len(indicators)} Categories</div>
                <div class="kpi-sub">Localized Area: {loc_res.total_damaged_area_pct:.2f}% of note</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid #64748B;">
                <div class="kpi-label">Capture Quality & Runtime</div>
                <div class="kpi-value">{q_rep.quality_rating}</div>
                <div class="kpi-sub">Blur Var: {q_rep.blur_laplacian_variance:.1f} | Time: {output.processing_time_ms:.0f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # -----------------------------------------------------------------------
    # Section 1: Before-and-After Comparison & Damage Localization Panels
    # -----------------------------------------------------------------------
    st.subheader("1. Visual Inspection & Spatial Damage Localization")
    _render_color_legend()

    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        st.image(
            pre_res.resized_rgb,
            caption=f"Original Input Image ({q_rep.width}×{q_rep.height} px)",
            width="stretch",
        )
    with vcol2:
        rect_caption = (
            f"Rectified & Preprocessed Note ({seg_res.method_used})"
            if seg_res.segmentation_succeeded
            else "Preprocessed Note (Manual / Cropped Fallback)"
        )
        st.image(
            seg_res.rectified_rgb,
            caption=rect_caption,
            width="stretch",
        )
    with vcol3:
        st.image(
            loc_res.annotated_rectified_rgb,
            caption=(
                f"Annotated Damage Localization ({len(loc_res.localized_regions)} region(s))"
                if loc_res.has_detected_damage
                else "Annotated Note (No Prominent Damage Regions Detected)"
            ),
            width="stretch",
        )

    with st.expander("View Combined Multi-Category Damage Map & Original-Perspective Overlay", expanded=False):
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.image(
                loc_res.combined_damage_map_rgb,
                caption="Combined Multi-Category Damage Map (Dark Diagnostic Canvas)",
                width="stretch",
            )
        with mcol2:
            st.image(
                loc_res.annotated_original_rgb,
                caption="Damage Overlay Mapped Back to Original Perspective (via Inverse Homography)",
                width="stretch",
            )

    # -----------------------------------------------------------------------
    # Section 2: CHI Gauge & AMFM Component Condition Breakdown
    # -----------------------------------------------------------------------
    st.subheader(f"2. {CORE_MODULE_NAME} & CHI Scoring")

    gcol1, gcol2 = st.columns([1.05, 1.0])
    with gcol1:
        fig_gauge = create_chi_gauge_chart(chi_res)
        st.pyplot(fig_gauge, width="stretch")
        plt.close(fig_gauge)

        fig_bars = create_component_scores_chart(amfm_res, chi_res)
        st.pyplot(fig_bars, width="stretch")
        plt.close(fig_bars)

    with gcol2:
        st.markdown("##### Component-Wise Condition Scores & Weights")
        comp_df = build_component_scores_dataframe(amfm_res, chi_res)
        st.dataframe(comp_df, width="stretch", hide_index=True)

        st.markdown("##### Scoring Formula & How Assessment Was Reached")
        st.latex(
            r"\text{CHI} = w_{\text{edge}} S_{\text{edge}} + w_{\text{fold}} S_{\text{fold}} "
            r"+ w_{\text{texture}} S_{\text{texture}} + w_{\text{color}} S_{\text{color}} "
            r"+ w_{\text{corner}} S_{\text{corner}}"
        )
        st.info(report_dict["prototype_assessment_summary"])

    # -----------------------------------------------------------------------
    # Section 3: Detected Visual Indicators & Region Coordinates
    # -----------------------------------------------------------------------
    st.subheader("3. Damage-Indicator Analysis & Affected Regions")
    ind_df = build_indicators_dataframe(indicators)
    st.dataframe(ind_df, width="stretch", hide_index=True)

    if loc_res.localized_regions:
        with st.expander(f"Inspect {len(loc_res.localized_regions)} Localized Region Coordinate(s)", expanded=False):
            reg_df = pd.DataFrame(report_dict["affected_regions"])
            st.dataframe(reg_df, width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # Section 4: Separate Diagnostic Masks & Intermediate Pipeline Views
    # -----------------------------------------------------------------------
    st.subheader("4. Diagnostic Masks & Intermediate Feature Maps")
    diag_tabs = st.tabs(
        [
            "Category Damage Masks",
            "Preprocessing Stages",
            "Edge & Boundary Analysis",
            "Morphology & Fold Analysis",
            "Texture & Color Analysis",
            "Reference & Evaluation",
        ]
    )

    with diag_tabs[0]:
        st.caption("Binary masks isolated per damage category in rectified note coordinates:")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.image(
                loc_res.category_masks["edge_corner"],
                caption="Edge & Corner Mask (Red class)",
                width="stretch",
            )
        with d2:
            st.image(
                loc_res.category_masks["fold_line"],
                caption="Fold-like Line Mask (Blue class)",
                width="stretch",
            )
        with d3:
            st.image(
                loc_res.category_masks["stain_color"],
                caption="Stain & Color Mask (Yellow class)",
                width="stretch",
            )
        with d4:
            st.image(
                loc_res.category_masks["texture"],
                caption="Texture Anomaly Mask (Purple class)",
                width="stretch",
            )

    with diag_tabs[1]:
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.image(pre_res.denoised_rgb, caption="1. Edge-Preserving Denoised RGB", width="stretch")
        with p2:
            st.image(
                pre_res.illumination_corrected_rgb,
                caption="2. Lab Illumination Normalized RGB",
                width="stretch",
            )
        with p3:
            st.image(pre_res.enhanced_rgb, caption="3. CLAHE Contrast Enhanced RGB", width="stretch")
        with p4:
            st.image(pre_res.clahe_gray, caption="4. CLAHE Grayscale Channel", width="stretch")

    with diag_tabs[2]:
        e1, e2, e3 = st.columns(3)
        with e1:
            st.image(
                edge_res.boundary_visualization_rgb,
                caption="4-Edge & 4-Corner Partition + Defect Highlight",
                width="stretch",
            )
        with e2:
            st.image(edge_res.canny_edges, caption="Canny Edge Detector Map", width="stretch")
        with e3:
            st.image(edge_res.sobel_magnitude_vis, caption="Sobel Gradient Magnitude", width="stretch")

        st.markdown(f"**Boundary Summary:** {edge_res.explanation}")

    with diag_tabs[3]:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.image(
                morph_res.directional_fold_response,
                caption="Anisotropic Directional Fold Response",
                width="stretch",
            )
        with m2:
            st.image(morph_res.blackhat_map, caption="Morphological Black-Hat (Troughs)", width="stretch")
        with m3:
            st.image(morph_res.tophat_map, caption="Morphological White Top-Hat (Ridges)", width="stretch")
        with m4:
            st.image(morph_res.morph_gradient, caption="Morphological Gradient", width="stretch")

        st.markdown(f"**Morphology Summary:** {morph_res.explanation}")

    with diag_tabs[4]:
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.image(tex_res.lbp_image, caption="Uniform Local Binary Pattern (LBP)", width="stretch")
        with tc2:
            st.image(tex_res.heatmap_gray, caption="Patch Texture Anomaly Heatmap", width="stretch")
        with tc3:
            st.image(
                col_res.color_anomaly_heatmap,
                caption="CIE Lab / HSV Stain Anomaly Heatmap",
                width="stretch",
            )
        with tc4:
            st.image(col_res.stain_mask, caption="Thresholded Stain Mask", width="stretch")

        st.markdown(
            f"- **Texture Summary:** {tex_res.explanation}\n"
            f"- **Color Summary:** {col_res.explanation}"
        )

    with diag_tabs[5]:
        if ref_res.reference_supplied:
            st.markdown("##### Reference-Assisted Registration Diagnostics")
            st.write(ref_res.status_message)
            if ref_res.alignment_succeeded and ref_res.match_visualization_rgb is not None:
                st.image(
                    ref_res.match_visualization_rgb,
                    caption=f"ORB + RANSAC Inlier Matches ({ref_res.inlier_count} inliers, SSIM={ref_res.global_ssim:.2f})",
                    width="stretch",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if ref_res.structural_diff_map is not None:
                        st.image(ref_res.structural_diff_map, caption="Structural Difference Map (1 - SSIM)", width="stretch")
                with rc2:
                    if ref_res.color_diff_map is not None:
                        st.image(ref_res.color_diff_map, caption="CIE Lab Color Difference Map", width="stretch")
        else:
            st.info("No reference image was supplied for this run.")

        st.divider()
        st.markdown("##### Quantitative Ground-Truth Mask Evaluation (Optional)")
        st.caption(
            "Upload a binary ground-truth defect mask (white = defect, black = clean) "
            "matching the rectified note to compute Precision, Recall, F1-Score, and IoU."
        )
        gt_upload = st.file_uploader(
            "Upload Ground-Truth Defect Mask (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="gt_mask_uploader",
        )
        if gt_upload is not None:
            gt_load = load_and_validate_image(gt_upload)
            if gt_load.is_valid and gt_load.rgb_image is not None:
                gt_gray = cv2.cvtColor(gt_load.rgb_image, cv2.COLOR_RGB2GRAY)
                pred_union = np.zeros_like(loc_res.category_masks["edge_corner"])
                for cm in loc_res.category_masks.values():
                    pred_union = cv2.bitwise_or(pred_union, cm)
                eval_metrics = evaluate_mask_against_ground_truth(
                    pred_union, gt_gray, processing_time_ms=output.processing_time_ms
                )
                ecol1, ecol2, ecol3, ecol4 = st.columns(4)
                ecol1.metric("Precision", f"{eval_metrics.precision:.3f}")
                ecol2.metric("Recall", f"{eval_metrics.recall:.3f}")
                ecol3.metric("F1-Score", f"{eval_metrics.f1_score:.3f}")
                ecol4.metric("IoU (Jaccard)", f"{eval_metrics.iou:.3f}")

    # -----------------------------------------------------------------------
    # Section 5: Warnings, Limitations, and Downloadable Reports
    # -----------------------------------------------------------------------
    st.subheader("5. Warnings, Limitations & Downloadable Outputs")

    warnings_all = report_dict.get("limitations_and_warnings", [])
    if warnings_all:
        with st.expander(f"Capture Quality Warnings & Methodological Notes ({len(warnings_all)})", expanded=True):
            for w_txt in warnings_all:
                st.warning(w_txt)

    # Generate downloadable bytes from actual analysis results
    annotated_png_bytes = encode_rgb_to_png_bytes(loc_res.annotated_rectified_rgb)
    processed_png_bytes = encode_rgb_to_png_bytes(seg_res.rectified_rgb)
    json_str = generate_json_report(report_dict)
    csv_str = generate_csv_summary(report_dict)
    pdf_bytes = generate_pdf_report(
        report_dict,
        processed_rgb=seg_res.rectified_rgb,
        annotated_rgb=loc_res.annotated_rectified_rgb,
    )

    dl1, dl2, dl3, dl4, dl5 = st.columns(5)
    with dl1:
        st.download_button(
            label="📥 Annotated Map (PNG)",
            data=annotated_png_bytes,
            file_name="achaf_annotated_damage_map.png",
            mime="image/png",
            width="stretch",
        )
    with dl2:
        st.download_button(
            label="📥 Processed Note (PNG)",
            data=processed_png_bytes,
            file_name="achaf_processed_note.png",
            mime="image/png",
            width="stretch",
        )
    with dl3:
        st.download_button(
            label="📥 Full Report (JSON)",
            data=json_str.encode("utf-8"),
            file_name="achaf_assessment_report.json",
            mime="application/json",
            width="stretch",
        )
    with dl4:
        st.download_button(
            label="📥 Score Summary (CSV)",
            data=csv_str.encode("utf-8"),
            file_name="achaf_summary_scores.csv",
            mime="text/csv",
            width="stretch",
        )
    with dl5:
        st.download_button(
            label="📥 Health Report (PDF)",
            data=pdf_bytes,
            file_name="achaf_health_report.pdf",
            mime="application/pdf",
            width="stretch",
        )

    st.markdown(
        f'<div class="disclaimer-banner"><strong>Academic Prototype Notice:</strong> {PROTOTYPE_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
