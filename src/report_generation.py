"""Module 19: Health Report Generation and Multi-Format Export (PNG, JSON, CSV, PDF).

Generates structured assessment reports and downloadable artifacts from actual
pipeline outputs:
1. Annotated damage-map PNG (`encode_rgb_to_png_bytes`)
2. Processed/rectified note PNG (`encode_rgb_to_png_bytes`)
3. Structured JSON report (`generate_json_report`)
4. Tabular CSV summary of component scores and damage indicators (`generate_csv_summary`)
5. Formatted academic PDF report using ReportLab (`generate_pdf_report`)
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config.settings import (
    CORE_MODULE_NAME,
    FRAMEWORK_NAME,
    PROJECT_TITLE,
    PROTOTYPE_DISCLAIMER,
    SCORING_METRIC_NAME,
)
from src.damage_detection import DamageIndicatorResult
from src.damage_localization import DamageLocalizationResult
from src.feature_fusion import AMFMFusionResult
from src.health_scoring import CHIScoringResult
from src.image_quality import CaptureQualityReport
from src.reference_comparison import ReferenceComparisonResult
from src.segmentation import SegmentationResult


def encode_rgb_to_png_bytes(rgb_image: np.ndarray) -> bytes:
    """Encode an RGB uint8 NumPy array into lossless PNG bytes."""
    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("Cannot encode an empty image to PNG.")
    arr = np.clip(rgb_image, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def build_health_report_dict(
    denomination: str,
    note_side: str,
    analysis_mode: str,
    quality_report: CaptureQualityReport,
    segmentation_result: SegmentationResult,
    reference_result: ReferenceComparisonResult,
    amfm_result: AMFMFusionResult,
    chi_result: CHIScoringResult,
    indicators: Dict[str, DamageIndicatorResult],
    localization_result: DamageLocalizationResult,
    processing_time_ms: float = 0.0,
) -> Dict[str, Any]:
    """Assemble the complete, structured ACHAF assessment report dictionary."""
    timestamp_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # Collect all warnings & limitations without duplicates
    all_warnings: List[str] = []
    for w in (
        quality_report.warnings
        + segmentation_result.warnings
        + reference_result.warnings
        + chi_result.limitations_and_notes
    ):
        if w and w not in all_warnings:
            all_warnings.append(w)

    detected_list = [
        ind.to_dict() for ind in indicators.values() if ind.is_detected
    ]
    all_indicators_list = [ind.to_dict() for ind in indicators.values()]

    regions_list = [
        {
            "region_id": r.region_id,
            "category_key": r.category_key,
            "category_name": r.category_name,
            "bbox_xywh": list(r.bbox),
            "centroid_xy": [round(r.centroid[0], 1), round(r.centroid[1], 1)],
            "area_px": r.area_px,
            "area_pct": round(r.area_pct, 3),
            "anatomical_location": r.anatomical_location,
            "confidence_label": r.confidence_label,
        }
        for r in localization_result.localized_regions
    ]

    # Compose executive prototype summary
    if chi_result.chi_score is not None:
        summary_text = (
            f"The analyzed currency note ({denomination}, Side: {note_side}) achieved a "
            f"Currency Health Index (CHI) of {chi_result.chi_score:.1f}/100, placing it in "
            f"the '{chi_result.health_category}' estimated physical-condition category "
            f"({chi_result.scoring_approach}). "
        )
    else:
        summary_text = (
            f"The analyzed currency note ({denomination}, Side: {note_side}) could not be "
            f"assigned a reliable overall CHI score ({chi_result.scoring_approach}). "
        )

    if detected_list:
        det_names = [ str(d["category_name"]) for d in detected_list ]
        summary_text += (
            f"Detected visual deterioration indicators ({len(detected_list)}): "
            f"{', '.join(det_names)}, covering approximately "
            f"{localization_result.total_damaged_area_pct:.2f}% of the rectified note area. "
        )
    else:
        summary_text += "No prominent localized physical damage indicators were detected. "

    summary_text += PROTOTYPE_DISCLAIMER

    return {
        "project_title": PROJECT_TITLE,
        "framework_name": FRAMEWORK_NAME,
        "core_module_name": CORE_MODULE_NAME,
        "scoring_metric_name": SCORING_METRIC_NAME,
        "analysis_timestamp": timestamp_iso,
        "processing_time_ms": round(processing_time_ms, 2),
        "note_metadata": {
            "selected_denomination": denomination,
            "selected_side": note_side,
            "requested_analysis_mode": analysis_mode,
            "effective_analysis_mode": (
                "Reference-assisted analysis"
                if reference_result.alignment_succeeded
                else "Feature-based analysis"
            ),
        },
        "input_dimensions": {
            "width": quality_report.width,
            "height": quality_report.height,
        },
        "capture_quality": quality_report.to_dict(),
        "segmentation": {
            "segmentation_succeeded": segmentation_result.segmentation_succeeded,
            "perspective_corrected": segmentation_result.perspective_corrected,
            "method_used": segmentation_result.method_used,
            "contour_area_ratio": round(segmentation_result.contour_area_ratio, 4),
            "estimated_aspect_ratio": round(segmentation_result.estimated_aspect_ratio, 3),
            "status_message": segmentation_result.status_message,
        },
        "reference_alignment": reference_result.to_summary_dict(),
        "currency_health_index": chi_result.to_dict(),
        "component_condition_scores": {
            k: v.to_dict() for k, v in amfm_result.components.items()
        },
        "damage_indicators": all_indicators_list,
        "detected_visual_indicators": detected_list,
        "affected_regions": regions_list,
        "total_damaged_area_pct": round(localization_result.total_damaged_area_pct, 3),
        "limitations_and_warnings": all_warnings,
        "prototype_assessment_summary": summary_text,
        "disclaimer": PROTOTYPE_DISCLAIMER,
    }


def generate_json_report(report_dict: Dict[str, Any]) -> str:
    """Serialize the health report dictionary into a formatted, valid JSON string."""
    return json.dumps(report_dict, indent=2, ensure_ascii=False)


def generate_csv_summary(report_dict: Dict[str, Any]) -> str:
    """Generate a readable CSV summary containing metadata, component scores, and indicators."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    writer.writerow(["SECTION", "PARAMETER / CATEGORY", "VALUE / SCORE", "STATUS / RELIABILITY", "DETAILS"])
    writer.writerow(["Metadata", "Framework", report_dict.get("framework_name", ""), "Info", ""])
    writer.writerow(["Metadata", "Timestamp", report_dict.get("analysis_timestamp", ""), "Info", ""])

    meta = report_dict.get("note_metadata", {})
    writer.writerow(["Metadata", "Denomination", meta.get("selected_denomination", "Unknown"), "Info", ""])
    writer.writerow(["Metadata", "Note Side", meta.get("selected_side", "Unknown"), "Info", ""])
    writer.writerow(["Metadata", "Effective Mode", meta.get("effective_analysis_mode", ""), "Info", ""])

    cq = report_dict.get("capture_quality", {})
    writer.writerow([
        "Capture Quality",
        "Dimensions & Rating",
        f"{cq.get('width', 0)}x{cq.get('height', 0)}",
        cq.get("quality_rating", ""),
        f"Brightness={cq.get('mean_brightness', 0)}, BlurVar={cq.get('blur_laplacian_variance', 0)}",
    ])

    seg = report_dict.get("segmentation", {})
    writer.writerow([
        "Segmentation",
        "Note Boundary Isolation",
        str(seg.get("segmentation_succeeded", False)),
        seg.get("method_used", ""),
        seg.get("status_message", ""),
    ])

    chi = report_dict.get("currency_health_index", {})
    writer.writerow([
        "Overall Assessment",
        "Currency Health Index (CHI)",
        chi.get("chi_score_rounded", "Withheld"),
        chi.get("health_category", "Unable to Assess"),
        chi.get("scoring_approach", ""),
    ])

    comps = report_dict.get("component_condition_scores", {})
    eff_weights = chi.get("effective_weights", {})
    for ckey, cdata in comps.items():
        w_val = eff_weights.get(ckey, 0.0)
        writer.writerow([
            "AMFM Component",
            cdata.get("display_name", ckey),
            cdata.get("condition_score", "Unavailable"),
            cdata.get("reliability_label", ""),
            f"EffectiveWeight={w_val:.2f} | {cdata.get('status_explanation', '')}",
        ])

    for ind in report_dict.get("damage_indicators", []):
        locs = "; ".join(ind.get("affected_locations", [])) or "None"
        writer.writerow([
            "Damage Indicator",
            ind.get("category_name", ""),
            ind.get("severity_score", 0.0),
            f"{ind.get('severity_label', '')} (Confidence: {ind.get('confidence_label', '')})",
            f"Regions={ind.get('region_count', 0)}, Area={ind.get('affected_area_pct', 0.0)}% [{locs}] | {ind.get('explanation', '')}",
        ])

    return output.getvalue()


def build_component_scores_dataframe(amfm_result: AMFMFusionResult, chi_result: CHIScoringResult) -> pd.DataFrame:
    """Create a pandas DataFrame of AMFM component condition scores for tabular display."""
    rows = []
    for k, comp in amfm_result.components.items():
        cfg_w = chi_result.configured_weights.get(k, 0.0)
        eff_w = chi_result.effective_weights.get(k, 0.0)
        contrib = chi_result.weighted_contributions.get(k, 0.0)
        rows.append(
            {
                "Component": comp.display_name,
                "Available": "Yes" if comp.is_available else "No",
                "Condition Score (0–100)": (
                    round(comp.condition_score, 1)
                    if comp.condition_score is not None
                    else np.nan
                ),
                "Defect Severity (0–100)": round(comp.raw_defect_severity, 1),
                "Configured Weight": round(cfg_w, 2),
                "Effective Weight": round(eff_w, 2),
                "CHI Contribution": round(contrib, 2),
                "Reliability": comp.reliability_label,
                "Ref Used": "Yes" if comp.used_reference else "No",
            }
        )
    return pd.DataFrame(rows)


def build_indicators_dataframe(indicators: Dict[str, DamageIndicatorResult]) -> pd.DataFrame:
    """Create a pandas DataFrame summarizing the 6 damage-indicator detectors."""
    rows = []
    for ind in indicators.values():
        locs = ", ".join(ind.affected_locations) if ind.affected_locations else "—"
        rows.append(
            {
                "Defect Category": ind.category_name,
                "Status / Severity": ind.severity_label,
                "Indicator Score (0–100)": round(ind.severity_score, 1) if ind.is_assessed else np.nan,
                "Affected Regions": ind.region_count if ind.is_assessed else 0,
                "Affected Area (%)": round(ind.affected_area_pct, 2) if ind.is_assessed else 0.0,
                "Locations": locs,
                "Reliability": ind.confidence_label,
                "Ref Assisted": "Yes" if ind.used_reference else "No",
                "Explanation": ind.explanation,
            }
        )
    return pd.DataFrame(rows)


def generate_pdf_report(
    report_dict: Dict[str, Any],
    processed_rgb: Optional[np.ndarray] = None,
    annotated_rgb: Optional[np.ndarray] = None,
) -> bytes:
    """Generate a multi-section PDF assessment report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="ACHAF Currency Note Health Assessment Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ACHAFTitle",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ACHAFSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "ACHAFH2",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ACHAFBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )
    disclaimer_style = ParagraphStyle(
        "ACHAFDisclaimer",
        parent=styles["Italic"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )

    elements: List[Any] = []

    # Header
    elements.append(Paragraph(FRAMEWORK_NAME, title_style))
    elements.append(Paragraph(PROJECT_TITLE, subtitle_style))

    # Executive Summary Box
    chi_info = report_dict.get("currency_health_index", {})
    meta = report_dict.get("note_metadata", {})
    cq = report_dict.get("capture_quality", {})
    seg = report_dict.get("segmentation", {})

    summary_rows = [
        [
            "Analysis Timestamp:",
            str(report_dict.get("analysis_timestamp", "")),
            "Currency Health Index (CHI):",
            str(chi_info.get("chi_display", "N/A")),
        ],
        [
            "Denomination / Side:",
            f"{meta.get('selected_denomination', 'Unknown')} / {meta.get('selected_side', 'Unknown')}",
            "Estimated Condition Category:",
            str(chi_info.get("health_category", "N/A")),
        ],
        [
            "Analysis Mode:",
            str(meta.get("effective_analysis_mode", "")),
            "Capture Quality Rating:",
            f"{cq.get('quality_rating', '')} ({cq.get('width', 0)}x{cq.get('height', 0)} px)",
        ],
        [
            "Segmentation Status:",
            f"{'Succeeded' if seg.get('segmentation_succeeded') else 'Fallback'} ({seg.get('method_used', '')})",
            "Damaged Surface Area:",
            f"{report_dict.get('total_damaged_area_pct', 0.0):.2f}%",
        ],
    ]

    summary_table = Table(summary_rows, colWidths=[38 * mm, 52 * mm, 45 * mm, 45 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 6))

    # Prototype summary paragraph
    elements.append(Paragraph("<b>Assessment Summary:</b> " + str(report_dict.get("prototype_assessment_summary", "")), body_style))
    elements.append(Spacer(1, 6))

    # Embedded visual comparison if images are provided
    img_cells = []
    for label, arr in [("Rectified / Processed Note", processed_rgb), ("Annotated Damage Localization", annotated_rgb)]:
        if arr is not None and arr.size > 0:
            png_bytes = encode_rgb_to_png_bytes(arr)
            img_bio = io.BytesIO(png_bytes)
            h_img, w_img = arr.shape[:2]
            disp_w = 84 * mm
            disp_h = disp_w * (float(h_img) / float(max(1, w_img)))
            disp_h = min(disp_h, 48 * mm)
            rl_img = RLImage(img_bio, width=disp_w, height=disp_h)
            img_cells.append([Paragraph(f"<b>{label}</b>", body_style), rl_img])

    if len(img_cells) == 2:
        vis_table = Table(
            [[img_cells[0][0], img_cells[1][0]], [img_cells[0][1], img_cells[1][1]]],
            colWidths=[90 * mm, 90 * mm],
        )
        vis_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(vis_table)
        elements.append(Spacer(1, 6))

    # Section: AMFM Component Condition Scores
    elements.append(Paragraph("1. AMFM Component-Wise Condition Scores", h2_style))
    comp_rows = [["Component", "Condition (0-100)", "Severity", "Eff. Weight", "Contribution", "Reliability"]]
    comps = report_dict.get("component_condition_scores", {})
    eff_w = chi_info.get("effective_weights", {})
    contribs = chi_info.get("weighted_contributions", {})

    for ckey, cdata in comps.items():
        cscore = cdata.get("condition_score")
        cscore_str = f"{cscore:.1f}" if cscore is not None else "Unavailable"
        comp_rows.append(
            [
                str(cdata.get("display_name", ckey)),
                cscore_str,
                f"{cdata.get('raw_defect_severity', 0.0):.1f}",
                f"{eff_w.get(ckey, 0.0):.2f}",
                f"{contribs.get(ckey, 0.0):.2f}",
                str(cdata.get("reliability_label", "")),
            ]
        )

    comp_table = Table(comp_rows, colWidths=[52 * mm, 30 * mm, 24 * mm, 24 * mm, 25 * mm, 25 * mm])
    comp_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(comp_table)

    # Section: Damage Indicators
    elements.append(Paragraph("2. Visual Damage Indicators & Localization", h2_style))
    ind_rows = [["Indicator Category", "Severity Status", "Score", "Regions", "Area %", "Confidence"]]
    for ind in report_dict.get("damage_indicators", []):
        ind_rows.append(
            [
                str(ind.get("category_name", "")),
                str(ind.get("severity_label", "")),
                f"{ind.get('severity_score', 0.0):.1f}",
                str(ind.get("region_count", 0)),
                f"{ind.get('affected_area_pct', 0.0):.2f}%",
                str(ind.get("confidence_label", "")),
            ]
        )

    ind_table = Table(ind_rows, colWidths=[58 * mm, 32 * mm, 20 * mm, 20 * mm, 22 * mm, 28 * mm])
    ind_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(ind_table)

    # Section: Warnings & Limitations
    warnings_list = report_dict.get("limitations_and_warnings", [])
    if warnings_list:
        elements.append(Paragraph("3. Capture Warnings & Methodological Limitations", h2_style))
        for w_item in warnings_list:
            elements.append(Paragraph(f"• {w_item}", body_style))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Disclaimer: {PROTOTYPE_DISCLAIMER}", disclaimer_style))

    doc.build(elements)
    return buffer.getvalue()
