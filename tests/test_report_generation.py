"""Unit tests for Module 19: Health Report Generation and Exports (PNG, JSON, CSV, PDF)."""

from __future__ import annotations

import csv
import io
import json

from PIL import Image

from src.evaluation import generate_synthetic_test_note
from src.pipeline import run_achaf_pipeline
from src.report_generation import (
    encode_rgb_to_png_bytes,
    generate_csv_summary,
    generate_json_report,
    generate_pdf_report,
)


def test_end_to_end_report_and_exports_validity() -> None:
    """Verify full pipeline produces valid JSON, readable CSV, valid PNGs, and valid PDF."""
    dmg_img, _ = generate_synthetic_test_note(variant="damaged")
    clean_img, _ = generate_synthetic_test_note(variant="clean")

    out = run_achaf_pipeline(
        image_source=dmg_img,
        denomination="₹100",
        note_side="Front",
        analysis_mode="Reference-assisted analysis",
        reference_source=clean_img,
    )

    assert out.is_success is True
    assert out.report_dict is not None
    assert out.segmentation_result is not None
    assert out.localization_result is not None

    # 1. Validate JSON report
    json_str = generate_json_report(out.report_dict)
    parsed = json.loads(json_str)
    assert parsed["framework_name"].startswith("ACHAF")
    assert parsed["note_metadata"]["selected_denomination"] == "₹100"
    assert "currency_health_index" in parsed
    assert "component_condition_scores" in parsed
    assert "damage_indicators" in parsed
    assert len(parsed["damage_indicators"]) == 6

    # 2. Validate CSV summary
    csv_str = generate_csv_summary(out.report_dict)
    reader = list(csv.reader(io.StringIO(csv_str)))
    assert len(reader) >= 12
    assert reader[0][0] == "SECTION"

    # 3. Validate PNG encodings
    proc_png = encode_rgb_to_png_bytes(out.segmentation_result.rectified_rgb)
    ann_png = encode_rgb_to_png_bytes(out.localization_result.annotated_rectified_rgb)
    with Image.open(io.BytesIO(proc_png)) as pil_p:
        assert pil_p.format == "PNG"
        assert pil_p.size == (
            out.segmentation_result.rectified_rgb.shape[1],
            out.segmentation_result.rectified_rgb.shape[0],
        )
    with Image.open(io.BytesIO(ann_png)) as pil_a:
        assert pil_a.format == "PNG"

    # 4. Validate ReportLab PDF generation
    pdf_bytes = generate_pdf_report(
        out.report_dict,
        processed_rgb=out.segmentation_result.rectified_rgb,
        annotated_rgb=out.localization_result.annotated_rectified_rgb,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1500
    assert pdf_bytes.startswith(b"%PDF-")


def test_streamlit_app_ui_execution() -> None:
    """Verify the Streamlit application loads, previews, and executes analysis without errors."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception

    # Select synthetic damaged note and click Analyze Currency Note
    at.radio[0].set_value("Synthetic Specimen: Damaged Note").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) >= 2

