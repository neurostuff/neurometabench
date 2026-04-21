#!/usr/bin/env python3
"""Convert Tahmasian dementia spreadsheets to merged NiMADS studyset + annotation.

This script is intentionally dataset-specific and uses hardcoded input/output paths:
- raw/meta-datasets/4. Dementia - Tahmasian/bvFTD_ALL.xlsx
- raw/meta-datasets/4. Dementia - Tahmasian/dementia_analysis_manual_annotation.xlsx

Outputs:
- data/nimads/dementia/merged/nimads_studyset.json
- data/nimads/dementia/merged/nimads_annotation.json
"""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
COORDINATES_XLSX = BASE_DIR / "raw/meta-datasets/4. Dementia - Tahmasian/bvFTD_ALL.xlsx"
ANNOTATION_XLSX = (
    BASE_DIR
    / "raw/meta-datasets/4. Dementia - Tahmasian/dementia_analysis_manual_annotation.xlsx"
)
OUTPUT_DIR = BASE_DIR / "data/nimads/dementia/merged"
STUDYSET_JSON = OUTPUT_DIR / "nimads_studyset.json"
ANNOTATION_JSON = OUTPUT_DIR / "nimads_annotation.json"

STUDYSET_ID = "dementia_merged_studyset"
ANNOTATION_ID = f"annotation_{STUDYSET_ID}"

ANNOTATION_COLUMNS = ["all", "decrease", "structural", "functional"]


def _normalize_minus(value: str) -> str:
    return value.replace("−", "-").replace("–", "-")


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = _normalize_minus(text)
    return text.strip()


def _normalize_header(value: Any) -> str:
    text = _clean_text(value)
    text = re.sub(r"\s+", "_", text)
    return text.lower()


def _safe_int(value: Any) -> int:
    text = _clean_text(value)
    if text == "":
        raise ValueError("Expected integer-like value but found empty string")
    return int(float(text))


def _safe_float(value: Any) -> float:
    text = _clean_text(value)
    if text == "":
        raise ValueError("Expected numeric coordinate but found empty string")
    parsed = float(text)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite coordinate value encountered: {value!r}")
    return parsed


def _parse_space(value: Any) -> str:
    text = _clean_text(value).upper()
    if text in {"MNI", "MNI152"}:
        return "MNI"
    if text in {"TAL", "TALAIRACH"}:
        return "TAL"
    raise ValueError(f"Unsupported coordinate space: {value!r}")


def _parse_bool_01(value: Any, *, col_name: str, study_id: str) -> bool:
    text = _clean_text(value)
    if text in {"0", "0.0"}:
        return False
    if text in {"1", "1.0"}:
        return True
    if isinstance(value, (int, bool)):
        if int(value) == 0:
            return False
        if int(value) == 1:
            return True
    if isinstance(value, float) and not pd.isna(value):
        if value == 0.0:
            return False
        if value == 1.0:
            return True
    raise ValueError(
        f"Invalid 0/1 value for annotation column {col_name!r} in {study_id!r}: {value!r}"
    )


def _require_columns(df: pd.DataFrame, required_keys: Iterable[str], file_path: Path) -> None:
    missing = [key for key in required_keys if key not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _load_coordinates_df() -> pd.DataFrame:
    df = pd.read_excel(COORDINATES_XLSX, engine="openpyxl", dtype=object)
    df = df.rename(columns={col: _normalize_header(col) for col in df.columns})

    _require_columns(
        df,
        [
            "experiment",
            "subjects",
            "x",
            "y",
            "z",
            "space",
            "n_bvftd",
            "n_con",
            "contrast",
            "modality",
        ],
        COORDINATES_XLSX,
    )

    cleaned_rows: List[Dict[str, Any]] = []
    dropped_rows = 0

    for _, row in df.iterrows():
        study_id = _clean_text(row["experiment"])
        x_text = _clean_text(row["x"])
        y_text = _clean_text(row["y"])
        z_text = _clean_text(row["z"])

        if study_id == "" or x_text == "" or y_text == "" or z_text == "":
            dropped_rows += 1
            continue

        cleaned_rows.append(
            {
                "study_id": study_id,
                "subjects": _safe_int(row["subjects"]),
                "n_bvftd": _safe_int(row["n_bvftd"]),
                "n_control": _safe_int(row["n_con"]),
                "x": _safe_float(x_text),
                "y": _safe_float(y_text),
                "z": _safe_float(z_text),
                "space": _parse_space(row["space"]),
                "contrast": _clean_text(row["contrast"]),
                "modality": _clean_text(row["modality"]),
            }
        )

    if not cleaned_rows:
        raise ValueError("No valid coordinate rows remained after cleaning")

    cleaned_df = pd.DataFrame(cleaned_rows)
    cleaned_df.attrs["dropped_rows"] = dropped_rows
    cleaned_df.attrs["raw_rows"] = len(df)
    return cleaned_df


def _build_studyset(coords_df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    studies_ordered: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    study_to_analysis_ids: Dict[str, List[str]] = {}

    for study_id, study_rows in coords_df.groupby("study_id", sort=False):
        # Deterministic per-study split by first-seen key tuple.
        group_map: "OrderedDict[Tuple[Any, ...], List[Dict[str, Any]]]" = OrderedDict()
        for row in study_rows.to_dict(orient="records"):
            key = (
                row["subjects"],
                row["n_bvftd"],
                row["n_control"],
                row["contrast"],
                row["modality"],
                row["space"],
            )
            group_map.setdefault(key, []).append(row)

        analyses: List[Dict[str, Any]] = []
        analysis_ids: List[str] = []
        for idx, (group_key, rows) in enumerate(group_map.items(), start=1):
            subjects, n_bvftd, n_control, contrast, modality, space = group_key
            analysis_id = f"{study_id}_{idx}"

            points = [
                {
                    "space": space,
                    "coordinates": [r["x"], r["y"], r["z"]],
                }
                for r in rows
            ]

            analysis = {
                "id": analysis_id,
                "name": analysis_id,
                "conditions": [],
                "weights": [],
                "images": [],
                "points": points,
                "metadata": {
                    "sample_sizes": [subjects],
                    "n_bvFTD": n_bvftd,
                    "n_control": n_control,
                    "contrast": contrast,
                    "modality": modality,
                    "space": space,
                },
                "study_id": study_id,
            }
            analyses.append(analysis)
            analysis_ids.append(analysis_id)

        studies_ordered[study_id] = {
            "id": study_id,
            "name": study_id,
            "authors": "",
            "publication": "",
            "metadata": {},
            "analyses": analyses,
        }
        study_to_analysis_ids[study_id] = analysis_ids

    studyset = {
        "id": STUDYSET_ID,
        "name": "dementia merged studyset",
        "description": "Merged from Tahmasian dementia spreadsheet pair.",
        "studies": list(studies_ordered.values()),
    }
    return studyset, study_to_analysis_ids


def _load_annotation_rows() -> Tuple[Dict[str, Dict[str, bool]], Dict[str, str], int]:
    df = pd.read_excel(ANNOTATION_XLSX, engine="openpyxl", dtype=object)
    df = df.rename(columns={col: _normalize_header(col) for col in df.columns})

    _require_columns(df, ["analysis_id", "pmids", "notes", *ANNOTATION_COLUMNS], ANNOTATION_XLSX)

    deduped: Dict[str, Dict[str, bool]] = {}
    pmids_by_study: Dict[str, str] = {}
    duplicate_rows_removed = 0

    for _, row in df.iterrows():
        study_id = _clean_text(row["analysis_id"])
        if study_id == "":
            continue

        note_values = {
            col: _parse_bool_01(row[col], col_name=col, study_id=study_id)
            for col in ANNOTATION_COLUMNS
        }

        if study_id in deduped:
            if deduped[study_id] != note_values:
                raise ValueError(
                    f"Conflicting duplicate annotation rows for {study_id!r}: "
                    f"existing={deduped[study_id]} new={note_values}"
                )
            duplicate_rows_removed += 1
            continue

        deduped[study_id] = note_values
        pmids_by_study[study_id] = _clean_text(row["pmids"])

    if not deduped:
        raise ValueError("No annotation rows found in manual annotation sheet")

    return deduped, pmids_by_study, duplicate_rows_removed


def _build_annotation(
    studyset: Dict[str, Any],
    study_to_analysis_ids: Dict[str, List[str]],
    note_by_study: Dict[str, Dict[str, bool]],
    pmids_by_study: Dict[str, str],
) -> Dict[str, Any]:
    study_ids_from_studyset = set(study_to_analysis_ids.keys())
    study_ids_from_ann = set(note_by_study.keys())

    missing_in_ann = sorted(study_ids_from_studyset - study_ids_from_ann)
    unknown_in_ann = sorted(study_ids_from_ann - study_ids_from_studyset)

    if missing_in_ann:
        raise ValueError(f"Missing annotation rows for study IDs: {missing_in_ann}")
    if unknown_in_ann:
        raise ValueError(f"Annotation rows reference unknown study IDs: {unknown_in_ann}")

    notes: List[Dict[str, Any]] = []
    for study in studyset["studies"]:
        sid = study["id"]
        note_values = note_by_study[sid]
        pmids = pmids_by_study.get(sid, "")
        for analysis_id in study_to_analysis_ids[sid]:
            notes.append(
                {
                    "note": dict(note_values),
                    "analysis": analysis_id,
                    "annotation": ANNOTATION_ID,
                }
            )
            # Keep pmids on study metadata to preserve source context.
            if pmids:
                study["metadata"]["pmids"] = pmids

    annotation = {
        "id": ANNOTATION_ID,
        "name": "dementia_merged_annotations",
        "description": "Manual boolean annotations for Tahmasian dementia studyset.",
        "metadata": {
            "project": "dementia",
            "source_files": [COORDINATES_XLSX.name, ANNOTATION_XLSX.name],
        },
        "note_keys": {key: "boolean" for key in ANNOTATION_COLUMNS},
        "studyset": STUDYSET_ID,
        "notes": notes,
    }
    return annotation


def _validate_outputs(studyset: Dict[str, Any], annotation: Dict[str, Any]) -> None:
    analysis_ids = {
        analysis["id"]
        for study in studyset.get("studies", [])
        for analysis in study.get("analyses", [])
    }
    if not analysis_ids:
        raise ValueError("No analysis IDs found in generated studyset")

    if annotation.get("note_keys") != {key: "boolean" for key in ANNOTATION_COLUMNS}:
        raise ValueError(
            f"Unexpected note_keys in annotation: {annotation.get('note_keys')}"
        )

    notes = annotation.get("notes", [])
    for item in notes:
        aid = item.get("analysis")
        if aid not in analysis_ids:
            raise ValueError(f"Annotation references missing analysis ID: {aid!r}")

    for study in studyset.get("studies", []):
        for analysis in study.get("analyses", []):
            for point in analysis.get("points", []):
                space = point.get("space")
                if space not in {"MNI", "TAL"}:
                    raise ValueError(f"Invalid space in points: {space!r}")
                coords = point.get("coordinates", [])
                if len(coords) != 3:
                    raise ValueError(f"Invalid coordinate length for point: {coords!r}")
                if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in coords):
                    raise ValueError(f"Invalid coordinate values for point: {coords!r}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coords_df = _load_coordinates_df()
    studyset, study_to_analysis_ids = _build_studyset(coords_df)

    note_by_study, pmids_by_study, duplicate_rows_removed = _load_annotation_rows()
    annotation = _build_annotation(
        studyset=studyset,
        study_to_analysis_ids=study_to_analysis_ids,
        note_by_study=note_by_study,
        pmids_by_study=pmids_by_study,
    )

    _validate_outputs(studyset, annotation)

    STUDYSET_JSON.write_text(json.dumps(studyset, indent=2), encoding="utf-8")
    ANNOTATION_JSON.write_text(json.dumps(annotation, indent=2), encoding="utf-8")

    num_studies = len(studyset["studies"])
    num_analyses = sum(len(study["analyses"]) for study in studyset["studies"])
    num_notes = len(annotation["notes"])

    print("Conversion complete")
    print(f"Coordinates source: {COORDINATES_XLSX}")
    print(f"Annotation source:  {ANNOTATION_XLSX}")
    print(f"Wrote studyset:     {STUDYSET_JSON}")
    print(f"Wrote annotation:   {ANNOTATION_JSON}")
    print(
        "Summary: "
        f"studies={num_studies}, analyses={num_analyses}, notes={num_notes}, "
        f"raw_coord_rows={coords_df.attrs.get('raw_rows', 0)}, "
        f"dropped_coord_rows={coords_df.attrs.get('dropped_rows', 0)}, "
        f"deduped_annotation_rows={duplicate_rows_removed}"
    )


if __name__ == "__main__":
    main()
