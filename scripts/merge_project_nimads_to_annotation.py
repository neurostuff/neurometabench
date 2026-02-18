#!/usr/bin/env python3
"""Merge project NiMADS files into one studyset + annotation with fuzzy deduplication."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


NAME_WEIGHT = 0.30
COORD_WEIGHT = 0.70
EXACT_BONUS = 0.05

GENERATED_FILENAMES = {
    "nimads_studyset.json",
    "nimads_annotation.json",
    "fuzzy_merge_diagnostics.json",
    "match_results_all_annotations.json",
}


def clean_text(value: Any) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower().strip()
    text = text.replace(">", " > ")
    text = re.sub(r"\s+", " ", text)
    return text


def split_name_base(name: str) -> str:
    return normalize_text(name).split(";", 1)[0].strip()


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def slugify_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", clean_text(value).strip())
    token = token.strip("_")
    return token or "study"


def canonicalize_study_label(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\bet\s+al\.?\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "study"


def sanitize_key(name: str, used: set[str]) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not key:
        key = "source"
    if key not in used:
        used.add(key)
        return key

    i = 2
    while f"{key}_{i}" in used:
        i += 1
    out = f"{key}_{i}"
    used.add(out)
    return out


def derive_study_id_token(study: dict[str, Any], fallback_study_id: str) -> str:
    candidates: list[Any] = [fallback_study_id, study.get("id"), study.get("publication"), study.get("name")]
    metadata = study.get("metadata")
    if isinstance(metadata, dict):
        for key in ("pmid", "pubmed_id", "pubmed", "publication_id"):
            if key in metadata:
                candidates.append(metadata.get(key))
        candidates.extend(metadata.values())

    for candidate in candidates:
        if candidate is None:
            continue
        text = clean_text(candidate).strip()
        if text.isdigit():
            return text
        match = re.search(r"(?<!\d)(\d{6,9})(?!\d)", text)
        if match:
            return match.group(1)

    return slugify_token(fallback_study_id)


def derive_study_match_key(study: dict[str, Any]) -> tuple[str, str]:
    candidates: list[Any] = [study.get("id"), study.get("name"), study.get("publication")]
    metadata = study.get("metadata")
    if isinstance(metadata, dict):
        for key in ("pmid", "pubmed_id", "pubmed", "publication_id"):
            if key in metadata:
                candidates.append(metadata.get(key))
        candidates.extend(metadata.values())

    for candidate in candidates:
        if candidate is None:
            continue
        text = clean_text(candidate).strip()
        if not text:
            continue
        if text.isdigit():
            return text, text
        match = re.search(r"(?<!\d)(\d{6,9})(?!\d)", text)
        if match:
            pmid = match.group(1)
            return pmid, pmid

    raw = clean_text(study.get("id") or study.get("name") or study.get("publication") or "study").strip()
    if ";" in raw:
        raw = raw.split(";", 1)[0].strip()
    if not raw:
        raw = "study"
    return canonicalize_study_label(raw), raw


def parse_points(points: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    parsed: list[tuple[float, float, float]] = []
    for point in points or []:
        coords = point.get("coordinates", [])
        if not isinstance(coords, (list, tuple)) or len(coords) != 3:
            continue
        try:
            parsed.append((float(coords[0]), float(coords[1]), float(coords[2])))
        except Exception:
            continue
    return parsed


def rounded_coords(coords: list[tuple[float, float, float]], decimals: int = 1) -> list[tuple[float, float, float]]:
    return sorted((round(x, decimals), round(y, decimals), round(z, decimals)) for x, y, z in coords)


def distance_to_similarity(distance: float) -> float:
    if distance <= 1.0:
        return 1.0
    if distance <= 2.0:
        return 0.9
    if distance <= 4.0:
        return 0.9 - ((distance - 2.0) * (0.3 / 2.0))
    if distance <= 8.0:
        return 0.6 - ((distance - 4.0) * (0.4 / 4.0))
    return 0.0


def _hungarian_minimize(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Solve rectangular assignment with Hungarian algorithm (min-cost)."""
    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return []

    transposed = False
    cost = cost_matrix
    if n_rows > n_cols:
        transposed = True
        cost = [[cost_matrix[i][j] for i in range(n_rows)] for j in range(n_cols)]
        n_rows, n_cols = n_cols, n_rows

    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    pairs = [(i, assignment[i]) for i in range(n_rows) if assignment[i] != -1]
    if not transposed:
        return pairs
    return [(j, i) for i, j in pairs]


def rectangular_assignment(scores: list[list[float]]) -> tuple[list[tuple[int, int]], str]:
    n_rows = len(scores)
    n_cols = len(scores[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return [], "empty"

    if linear_sum_assignment is not None:
        costs = [[1.0 - scores[i][j] for j in range(n_cols)] for i in range(n_rows)]
        row_ind, col_ind = linear_sum_assignment(costs)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist())]
        return pairs, "scipy_hungarian"

    # Pure-Python Hungarian fallback keeps one-to-one optimal assignment.
    costs = [[1.0 - scores[i][j] for j in range(n_cols)] for i in range(n_rows)]
    pairs = _hungarian_minimize(costs)
    return pairs, "python_hungarian"


def compute_name_score(name_a: str, name_b: str) -> float:
    a_full = normalize_text(name_a)
    b_full = normalize_text(name_b)
    a_base = split_name_base(name_a)
    b_base = split_name_base(name_b)

    scores = [
        SequenceMatcher(None, a_full, b_full).ratio(),
        SequenceMatcher(None, a_base, b_base).ratio(),
        SequenceMatcher(None, a_full, b_base).ratio(),
        SequenceMatcher(None, a_base, b_full).ratio(),
    ]
    return max(scores)


def compute_coord_score(
    coords_a: list[tuple[float, float, float]],
    coords_b: list[tuple[float, float, float]],
) -> tuple[float, bool, float, float]:
    if not coords_a or not coords_b:
        return 0.0, False, 0.0, 0.0

    sim_matrix: list[list[float]] = []
    for ax, ay, az in coords_a:
        row: list[float] = []
        for bx, by, bz in coords_b:
            dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            row.append(distance_to_similarity(dist))
        sim_matrix.append(row)

    pairs, _backend = rectangular_assignment(sim_matrix)
    if not pairs:
        return 0.0, False, 0.0, 0.0

    matched = [sim_matrix[i][j] for i, j in pairs]
    match_quality = sum(matched) / len(matched)
    coverage_penalty = min(len(coords_a), len(coords_b)) / max(len(coords_a), len(coords_b))
    exact_coord_set = len(coords_a) == len(coords_b) and rounded_coords(coords_a) == rounded_coords(coords_b)

    score = (match_quality * coverage_penalty) + (EXACT_BONUS if exact_coord_set else 0.0)
    score = max(0.0, min(1.0, score))
    return score, exact_coord_set, coverage_penalty, match_quality


def score_analysis_pair(source_member: dict[str, Any], canonical_member: dict[str, Any]) -> dict[str, Any]:
    name_score = compute_name_score(source_member["name"], canonical_member["name"])
    coord_score, exact_coord_set, coverage_penalty, match_quality = compute_coord_score(
        source_member["coords"], canonical_member["coords"]
    )
    combined = (COORD_WEIGHT * coord_score) + (NAME_WEIGHT * name_score)

    reason_codes: list[str] = []
    if exact_coord_set:
        reason_codes.append("exact_coord_set")
    if coord_score >= 0.75:
        reason_codes.append("high_coord_match")
    if len(source_member["coords"]) != len(canonical_member["coords"]):
        reason_codes.append("coord_count_mismatch")
    if not source_member["coords"] or not canonical_member["coords"]:
        reason_codes.append("missing_coords_on_one_side")
    if coord_score < 0.4 and name_score >= 0.75:
        reason_codes.append("low_coord_high_name")
    if coord_score == 0.0 and name_score >= 0.6:
        reason_codes.append("name_only_signal")

    return {
        "name_score": float(name_score),
        "coord_score": float(coord_score),
        "combined_score": float(combined),
        "exact_coord_set": bool(exact_coord_set),
        "coverage_penalty": float(coverage_penalty),
        "match_quality": float(match_quality),
        "reason_codes": sorted(set(reason_codes)),
    }


def get_representative_member(group: dict[str, Any]) -> dict[str, Any]:
    return max(
        group["members"],
        key=lambda m: (
            len(m["coords"]),
            len(m["name"]),
            -m["source_order"],
            -m["global_order"],
        ),
    )


def build_analysis_object_from_group(group: dict[str, Any], study_id: str) -> dict[str, Any]:
    rep = get_representative_member(group)
    out = copy.deepcopy(rep["analysis_obj"])

    metadata = out.get("metadata")
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        metadata = {"_original_metadata": metadata}

    merged_sources = sorted({m["source_key"] for m in group["members"]})
    merged_member_ids = [clean_text(m["id"]) for m in group["members"]]
    merged_member_names = [clean_text(m["name"]) for m in group["members"]]

    metadata["merged_sources"] = merged_sources
    metadata["merged_member_count"] = len(group["members"])
    metadata["merged_member_ids"] = merged_member_ids
    metadata["merged_member_names"] = merged_member_names
    metadata["representative_source"] = rep["source_key"]

    out["metadata"] = metadata
    out["name"] = rep["name"]
    out["id"] = rep["id"] if rep["id"] else "analysis"
    if "study_id" not in out:
        out["study_id"] = study_id

    return out


def assign_pmid_index_ids(study_token: str, analyses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pmid = slugify_token(study_token)
    ops: list[dict[str, Any]] = []

    for idx, analysis in enumerate(analyses, start=1):
        new_id = f"{pmid}_{idx}"
        old_id = analysis.get("id")
        analysis["id"] = new_id
        if str(old_id) != new_id:
            ops.append({"old_id": old_id, "new_id": new_id})

    return analyses, ops


def discover_source_files(project_dir: Path, output_dir: Path) -> list[Path]:
    files = []
    for path in sorted(project_dir.glob("*.json")):
        if path.name in GENERATED_FILENAMES:
            continue
        if path.parent == output_dir:
            continue
        files.append(path)
    return files


def load_source_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "studies" not in data:
        raise ValueError(f"Invalid NiMADS file (missing studies): {path}")
    return data


def merge_study_metadata(base: dict[str, Any], incoming: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out = copy.deepcopy(base)
    conflicts: list[dict[str, Any]] = []

    for key, incoming_value in incoming.items():
        if key == "analyses":
            continue
        if key not in out:
            out[key] = copy.deepcopy(incoming_value)
            continue

        current_value = out[key]
        if is_empty_value(current_value) and not is_empty_value(incoming_value):
            out[key] = copy.deepcopy(incoming_value)
            continue

        if is_empty_value(incoming_value) or current_value == incoming_value:
            continue

        conflicts.append(
            {
                "field": key,
                "kept": current_value,
                "incoming": incoming_value,
            }
        )

    return out, conflicts


def merge_project(
    project: str,
    source_files: list[dict[str, Any]],
    threshold: float,
    exact_coord_override: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    study_pool: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"study_id_display": None, "study_records_by_source": defaultdict(list), "analyses_by_source": defaultdict(list)}
    )
    source_stats: dict[str, dict[str, Any]] = {}
    primary_source_filename = source_files[0]["filename"] if source_files else None

    assignment_backends_used: set[str] = set()

    input_analyses_total = 0
    input_studies_total = 0

    for src in source_files:
        source_stats[src["filename"]] = {
            "source_key": src["source_key"],
            "source_order": src["source_order"],
            "is_primary_source": src["filename"] == primary_source_filename,
            "studies_present": 0,
            "analyses_contributed": 0,
            "analyses_retained_unique": 0,
            "analyses_retained_unique_subsequent": 0,
            "analyses_merged_into_existing": 0,
        }

        payload = src["payload"]
        studies = payload.get("studies", [])
        input_studies_total += len(studies)
        source_study_keys_seen: set[str] = set()

        for study in studies:
            study_key, study_display = derive_study_match_key(study)
            if not study_key:
                continue

            source_study_keys_seen.add(study_key)
            bucket = study_pool[study_key]
            if bucket["study_id_display"] is None:
                bucket["study_id_display"] = study_display

            bucket["study_records_by_source"][src["filename"]].append(study)
            analyses = study.get("analyses", [])
            source_study_id = clean_text(study.get("id"))
            source_study_name = clean_text(study.get("name"))
            for analysis_idx, analysis in enumerate(analyses):
                bucket["analyses_by_source"][src["filename"]].append(
                    {
                        "analysis": analysis,
                        "source_study_id": source_study_id,
                        "source_study_name": source_study_name,
                        "source_analysis_index_in_study": analysis_idx,
                    }
                )

            source_stats[src["filename"]]["analyses_contributed"] += len(analyses)
            input_analyses_total += len(analyses)

        source_stats[src["filename"]]["studies_present"] = len(source_study_keys_seen)

    merged_studies: list[dict[str, Any]] = []
    study_diagnostics: dict[str, Any] = {}

    merged_pair_events_all: list[dict[str, Any]] = []
    metadata_conflict_count = 0
    id_reassignment_count = 0
    accepted_fuzzy_merges = 0
    exact_override_merges = 0
    unique_addition_events_all: list[dict[str, Any]] = []
    unique_addition_events_subsequent: list[dict[str, Any]] = []

    for study_key in sorted(study_pool.keys(), key=lambda x: (len(x), x)):
        bucket = study_pool[study_key]
        study_id = clean_text(bucket.get("study_id_display") or study_key)

        available_sources = [src for src in source_files if src["filename"] in bucket["study_records_by_source"]]
        first_src = available_sources[0]
        base_study = bucket["study_records_by_source"][first_src["filename"]][0]

        merged_study, metadata_conflicts = merge_study_metadata(base_study, base_study)
        for src in available_sources:
            records = bucket["study_records_by_source"][src["filename"]]
            for rec_idx, record in enumerate(records):
                if src["filename"] == first_src["filename"] and rec_idx == 0:
                    continue
                merged_study, new_conflicts = merge_study_metadata(merged_study, record)
                if new_conflicts:
                    for c in new_conflicts:
                        c["source_file"] = src["filename"]
                    metadata_conflicts.extend(new_conflicts)

        merged_study["id"] = study_id
        metadata = merged_study.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {"_original_metadata": metadata}
        original_study_ids_by_source = {
            src["source_key"]: sorted(
                {
                    clean_text(record.get("id"))
                    for record in bucket["study_records_by_source"][src["filename"]]
                    if clean_text(record.get("id"))
                }
            )
            for src in available_sources
        }
        metadata["original_study_ids_by_source"] = original_study_ids_by_source
        metadata["original_study_ids"] = sorted(
            {
                sid
                for source_ids in original_study_ids_by_source.values()
                for sid in source_ids
            }
        )
        metadata["study_match_key"] = study_key
        merged_study["metadata"] = metadata

        metadata_conflict_count += len(metadata_conflicts)

        groups: list[dict[str, Any]] = []
        global_member_order = 0
        merge_events: list[dict[str, Any]] = []
        unique_addition_events: list[dict[str, Any]] = []

        for src in available_sources:
            source_filename = src["filename"]
            source_key = src["source_key"]
            analysis_entries = bucket["analyses_by_source"].get(source_filename, [])

            source_members: list[dict[str, Any]] = []
            for idx, entry in enumerate(analysis_entries):
                analysis = entry.get("analysis", {})
                member = {
                    "id": clean_text(analysis.get("id") or analysis.get("name") or "analysis"),
                    "name": clean_text(analysis.get("name") or analysis.get("id") or f"analysis_{idx}"),
                    "coords": parse_points(analysis.get("points", [])),
                    "analysis_obj": copy.deepcopy(analysis),
                    "source_file": source_filename,
                    "source_key": source_key,
                    "source_analysis_index": idx,
                    "source_analysis_index_in_study": int(entry.get("source_analysis_index_in_study", idx)),
                    "source_study_id": clean_text(entry.get("source_study_id") or study_id),
                    "source_study_name": clean_text(entry.get("source_study_name") or ""),
                    "source_order": src["source_order"],
                    "global_order": global_member_order,
                }
                global_member_order += 1
                source_members.append(member)

            if not source_members:
                continue

            if not groups:
                for member in source_members:
                    groups.append({"members": [member]})
                    source_stats[source_filename]["analyses_retained_unique"] += 1
                    if src["source_order"] > 0:
                        source_stats[source_filename]["analyses_retained_unique_subsequent"] += 1
                    unique_event = {
                        "study_id": study_id,
                        "source_file": source_filename,
                        "source_key": source_key,
                        "source_order": src["source_order"],
                        "is_primary_source": src["source_order"] == 0,
                        "source_analysis_index": member["source_analysis_index"],
                        "source_analysis_index_in_study": member["source_analysis_index_in_study"],
                        "source_analysis_id": member["id"],
                        "source_analysis_name": member["name"],
                        "source_study_id": member["source_study_id"],
                        "source_study_name": member["source_study_name"],
                        "best_target_representative_id": None,
                        "best_target_representative_name": None,
                        "best_target_source_study_id": None,
                        "best_name_score": None,
                        "best_coord_score": None,
                        "best_combined_score": None,
                        "best_exact_coord_set": None,
                        "reason_codes": ["seed_from_first_source" if src["source_order"] == 0 else "no_existing_canonical_groups"],
                    }
                    unique_addition_events_all.append(unique_event)
                    unique_addition_events.append(unique_event)
                    if src["source_order"] > 0:
                        unique_addition_events_subsequent.append(unique_event)
                continue

            score_matrix: list[list[float]] = []
            pair_details: dict[tuple[int, int], dict[str, Any]] = {}
            rep_members = [get_representative_member(group) for group in groups]

            for i, source_member in enumerate(source_members):
                row: list[float] = []
                for j, rep in enumerate(rep_members):
                    detail = score_analysis_pair(source_member, rep)
                    pair_details[(i, j)] = detail
                    row.append(detail["combined_score"])
                score_matrix.append(row)

            pairs, backend = rectangular_assignment(score_matrix)
            assignment_backends_used.add(backend)

            matched_source_indices: set[int] = set()
            for i, j in pairs:
                source_member = source_members[i]
                detail = pair_details[(i, j)]
                merged = (
                    detail["combined_score"] >= threshold
                    or (exact_coord_override and detail["exact_coord_set"])
                )

                if merged:
                    groups[j]["members"].append(source_member)
                    matched_source_indices.add(i)
                    source_stats[source_filename]["analyses_merged_into_existing"] += 1

                    if detail["combined_score"] >= threshold:
                        accepted_fuzzy_merges += 1
                    elif detail["exact_coord_set"]:
                        exact_override_merges += 1

                    if detail["combined_score"] >= threshold and detail["exact_coord_set"]:
                        merge_reason = "threshold_and_exact"
                    elif detail["combined_score"] >= threshold:
                        merge_reason = "threshold"
                    else:
                        merge_reason = "exact_coord_override"

                    event = {
                        "study_id": study_id,
                        "source_file": source_filename,
                        "source_key": source_key,
                        "source_analysis_index": source_member["source_analysis_index"],
                        "source_analysis_index_in_study": source_member["source_analysis_index_in_study"],
                        "source_analysis_id": source_member["id"],
                        "source_analysis_name": source_member["name"],
                        "source_study_id": source_member["source_study_id"],
                        "source_study_name": source_member["source_study_name"],
                        "target_group_index": j,
                        "target_representative_id": rep_members[j]["id"],
                        "target_representative_name": rep_members[j]["name"],
                        "target_source_study_id": rep_members[j]["source_study_id"],
                        "target_source_study_name": rep_members[j]["source_study_name"],
                        "name_score": round(detail["name_score"], 6),
                        "coord_score": round(detail["coord_score"], 6),
                        "combined_score": round(detail["combined_score"], 6),
                        "coverage_penalty": round(detail["coverage_penalty"], 6),
                        "match_quality": round(detail["match_quality"], 6),
                        "exact_coord_set": bool(detail["exact_coord_set"]),
                        "merge_reason": merge_reason,
                        "reason_codes": detail["reason_codes"],
                    }
                    merge_events.append(event)
                    merged_pair_events_all.append(event)

            for i, source_member in enumerate(source_members):
                if i in matched_source_indices:
                    continue
                groups.append({"members": [source_member]})
                source_stats[source_filename]["analyses_retained_unique"] += 1
                if src["source_order"] > 0:
                    source_stats[source_filename]["analyses_retained_unique_subsequent"] += 1

                best_j = None
                best_detail = None
                if score_matrix and score_matrix[i]:
                    best_j = max(range(len(score_matrix[i])), key=lambda jj: score_matrix[i][jj])
                    best_detail = pair_details.get((i, best_j))

                unique_event = {
                    "study_id": study_id,
                    "source_file": source_filename,
                    "source_key": source_key,
                    "source_order": src["source_order"],
                    "is_primary_source": src["source_order"] == 0,
                    "source_analysis_index": source_member["source_analysis_index"],
                    "source_analysis_index_in_study": source_member["source_analysis_index_in_study"],
                    "source_analysis_id": source_member["id"],
                    "source_analysis_name": source_member["name"],
                    "source_study_id": source_member["source_study_id"],
                    "source_study_name": source_member["source_study_name"],
                    "best_target_representative_id": None if best_j is None else rep_members[best_j]["id"],
                    "best_target_representative_name": None if best_j is None else rep_members[best_j]["name"],
                    "best_target_source_study_id": None if best_j is None else rep_members[best_j]["source_study_id"],
                    "best_name_score": None if best_detail is None else round(best_detail["name_score"], 6),
                    "best_coord_score": None if best_detail is None else round(best_detail["coord_score"], 6),
                    "best_combined_score": None if best_detail is None else round(best_detail["combined_score"], 6),
                    "best_exact_coord_set": None if best_detail is None else bool(best_detail["exact_coord_set"]),
                    "reason_codes": ["retained_unique_new_canonical"] if best_detail is None else sorted(set(["retained_unique_new_canonical"] + best_detail["reason_codes"])),
                }
                unique_addition_events_all.append(unique_event)
                unique_addition_events.append(unique_event)
                if src["source_order"] > 0:
                    unique_addition_events_subsequent.append(unique_event)

        canonical_analyses: list[dict[str, Any]] = []
        group_source_memberships: list[set[str]] = []
        for group in groups:
            analysis_obj = build_analysis_object_from_group(group, study_id=study_id)
            canonical_analyses.append(analysis_obj)
            group_source_memberships.append(set(analysis_obj.get("metadata", {}).get("merged_sources", [])))

        study_token = derive_study_id_token(merged_study, study_id)
        canonical_analyses, id_ops = assign_pmid_index_ids(study_token, canonical_analyses)
        id_reassignment_count += len(id_ops)

        merged_study = copy.deepcopy(merged_study)
        merged_study["analyses"] = canonical_analyses
        merged_studies.append(merged_study)

        study_diagnostics[study_id] = {
            "source_files_present": [src["filename"] for src in available_sources],
            "original_study_ids": metadata.get("original_study_ids", []),
            "original_study_ids_by_source": metadata.get("original_study_ids_by_source", {}),
            "source_analyses_count": sum(len(bucket["analyses_by_source"].get(src["filename"], [])) for src in available_sources),
            "canonical_analyses_count": len(canonical_analyses),
            "dedup_percent": round(
                100.0
                * (1.0 - (len(canonical_analyses) / max(1, sum(len(bucket["analyses_by_source"].get(src["filename"], [])) for src in available_sources)))),
                3,
            ),
            "metadata_conflicts": metadata_conflicts,
            "merge_events": merge_events,
            "unique_additions": unique_addition_events,
            "id_reassignment_operations": id_ops,
            "canonical_analyses": [
                {
                    "id": a.get("id"),
                    "name": a.get("name"),
                    "merged_sources": sorted(list(group_source_memberships[idx])),
                    "merged_member_count": a.get("metadata", {}).get("merged_member_count", 1),
                }
                for idx, a in enumerate(canonical_analyses)
            ],
        }

    merged_studyset_id = f"{project}_merged_studyset"
    annotation_id = f"annotation_{merged_studyset_id}"

    merged_studyset = {
        "id": merged_studyset_id,
        "name": f"{project} merged studyset",
        "description": f"Merged from {len(source_files)} NiMADS source files with coordinate-first fuzzy deduplication.",
        "studies": merged_studies,
    }

    source_keys = [src["source_key"] for src in source_files]
    note_keys = {key: "boolean" for key in source_keys}

    notes: list[dict[str, Any]] = []
    all_analysis_ids = []
    for study in sorted(merged_studies, key=lambda s: (len(str(s.get("id", ""))), str(s.get("id", "")))):
        for analysis in study.get("analyses", []):
            aid = clean_text(analysis.get("id"))
            all_analysis_ids.append(aid)
            merged_sources = set(analysis.get("metadata", {}).get("merged_sources", []))
            note = {key: (key in merged_sources) for key in source_keys}
            notes.append(
                {
                    "note": note,
                    "analysis": aid,
                    "annotation": annotation_id,
                }
            )

    merged_annotation = {
        "id": annotation_id,
        "name": f"{project}_merged_annotations",
        "description": f"Source-subanalysis membership annotations for merged {project} NiMADS studyset.",
        "metadata": {
            "project": project,
            "source_files": [src["filename"] for src in source_files],
        },
        "note_keys": note_keys,
        "studyset": merged_studyset_id,
        "notes": notes,
    }

    merged_analysis_count = sum(len(study.get("analyses", [])) for study in merged_studies)
    dedup_rate = 1.0 - (merged_analysis_count / input_analyses_total) if input_analyses_total else 0.0

    merged_scores = [e["combined_score"] for e in merged_pair_events_all]
    sorted_scores = sorted(merged_scores)

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        pos = (len(values) - 1) * q
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return float(values[lower])
        frac = pos - lower
        return float(values[lower] + (values[upper] - values[lower]) * frac)

    duplicate_analysis_ids_global = len(all_analysis_ids) - len(set(all_analysis_ids))
    unique_subsequent_by_source: dict[str, int] = defaultdict(int)
    for event in unique_addition_events_subsequent:
        unique_subsequent_by_source[event["source_file"]] += 1

    diagnostics = {
        "config": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "project": project,
            "threshold": threshold,
            "exact_coord_override": exact_coord_override,
            "name_weight": NAME_WEIGHT,
            "coord_weight": COORD_WEIGHT,
            "assignment_backends_used": sorted(assignment_backends_used),
            "dedup_scope": "same_pmid_only",
            "coordinate_space_policy": "ignored_raw_xyz",
        },
        "summary": {
            "input_files": len(source_files),
            "input_studies_total": input_studies_total,
            "input_unique_studies": len(study_pool),
            "input_analyses_total": input_analyses_total,
            "merged_studies": len(merged_studies),
            "merged_analyses": merged_analysis_count,
            "dedup_rate": round(dedup_rate, 6),
            "accepted_fuzzy_merges": accepted_fuzzy_merges,
            "exact_override_merges": exact_override_merges,
            "mean_combined_score": round(sum(merged_scores) / len(merged_scores), 6) if merged_scores else 0.0,
            "median_combined_score": round(percentile(sorted_scores, 0.5), 6) if merged_scores else 0.0,
            "p25_combined_score": round(percentile(sorted_scores, 0.25), 6) if merged_scores else 0.0,
            "p75_combined_score": round(percentile(sorted_scores, 0.75), 6) if merged_scores else 0.0,
            "id_reassignment_count": id_reassignment_count,
            "id_collision_suffix_count": id_reassignment_count,
            "study_metadata_conflict_count": metadata_conflict_count,
            "duplicate_analysis_ids_global": duplicate_analysis_ids_global,
            "unique_additions_total": len(unique_addition_events_all),
            "unique_additions_subsequent_total": len(unique_addition_events_subsequent),
            "unique_additions_subsequent_by_source": dict(sorted(unique_subsequent_by_source.items())),
        },
        "source_files": source_stats,
        "studies": study_diagnostics,
        "top_low_confidence_merges": sorted(merged_pair_events_all, key=lambda e: e["combined_score"])[:50],
        "unique_additions_subsequent": sorted(
            unique_addition_events_subsequent,
            key=lambda e: (
                e["source_file"],
                len(str(e["study_id"])),
                str(e["study_id"]),
                e["source_analysis_index"],
            ),
        ),
    }

    report_html = render_html_summary(diagnostics)

    return merged_studyset, merged_annotation, diagnostics, report_html


def render_html_summary(diagnostics: dict[str, Any]) -> str:
    summary = diagnostics["summary"]
    config = diagnostics["config"]
    unique_subsequent = sorted(
        diagnostics.get("unique_additions_subsequent", []),
        key=lambda e: (
            -1.0 if e.get("best_combined_score") is None else float(e.get("best_combined_score")),
            str(e.get("source_file", "")),
            len(str(e.get("study_id", ""))),
            str(e.get("study_id", "")),
        ),
        reverse=True,
    )

    def fmt_opt(value: Any) -> str:
        if value is None:
            return ""
        return f"{float(value):.3f}"

    def pmid_link(study_id: Any) -> str:
        study_text = str(study_id)
        if study_text.isdigit():
            url = f"https://pubmed.ncbi.nlm.nih.gov/{study_text}/"
        else:
            url = f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(study_text)}"
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{escape(study_text)}</a>'

    source_rows = []
    for filename in sorted(diagnostics["source_files"].keys()):
        s = diagnostics["source_files"][filename]
        source_rows.append(
            "<tr>"
            f"<td>{escape(filename)}</td>"
            f"<td>{escape(str(s['source_key']))}</td>"
            f"<td>{'yes' if s.get('is_primary_source') else 'no'}</td>"
            f"<td>{int(s['studies_present'])}</td>"
            f"<td>{int(s['analyses_contributed'])}</td>"
            f"<td>{int(s['analyses_retained_unique'])}</td>"
            f"<td>{int(s.get('analyses_retained_unique_subsequent', 0))}</td>"
            f"<td>{int(s['analyses_merged_into_existing'])}</td>"
            "</tr>"
        )

    unique_rows = []
    near_miss_rows = []
    for e in unique_subsequent:
        row_class = ""
        if e.get("best_combined_score") is not None and float(e["best_combined_score"]) >= 0.70:
            row_class = ' class="near-miss"'

        row_html = (
            f"<tr{row_class}>"
            f"<td>{escape(e['source_file'])}</td>"
            f"<td>{escape(e['source_key'])}</td>"
            f"<td>{pmid_link(e['study_id'])}</td>"
            f"<td>{escape(str(e.get('source_study_id') or ''))}</td>"
            f"<td>{escape(str(e['source_analysis_id']))}</td>"
            f"<td>{escape(str(e['source_analysis_name']))}</td>"
            f"<td>{escape(str(e.get('best_target_source_study_id') or ''))}</td>"
            f"<td>{escape(str(e['best_target_representative_id'] or ''))}</td>"
            f"<td>{escape(str(e['best_target_representative_name'] or ''))}</td>"
            f"<td>{fmt_opt(e['best_combined_score'])}</td>"
            f"<td>{fmt_opt(e['best_name_score'])}</td>"
            f"<td>{fmt_opt(e['best_coord_score'])}</td>"
            f"<td>{'yes' if e['best_exact_coord_set'] else 'no' if e['best_exact_coord_set'] is not None else ''}</td>"
            f"<td>{escape(', '.join(e.get('reason_codes', [])))}</td>"
            "</tr>"
        )
        unique_rows.append(row_html)

        if e.get("best_combined_score") is not None and float(e["best_combined_score"]) >= 0.70:
            near_miss_rows.append(row_html)

    study_rows = []
    for study_id in sorted(diagnostics["studies"].keys(), key=lambda x: (len(x), x)):
        st = diagnostics["studies"][study_id]
        details = []
        details.append(f"<p><strong>Source files:</strong> {', '.join(st['source_files_present'])}</p>")
        if st.get("original_study_ids"):
            details.append(f"<p><strong>Original study IDs (pre-merge):</strong> {', '.join(escape(str(x)) for x in st['original_study_ids'])}</p>")

        if st["metadata_conflicts"]:
            conflict_items = "".join(
                f"<li>{escape(c['field'])}: kept={escape(str(c['kept']))} | incoming={escape(str(c['incoming']))} ({escape(c['source_file'])})</li>"
                for c in st["metadata_conflicts"]
            )
            details.append(f"<details><summary>Metadata conflicts ({len(st['metadata_conflicts'])})</summary><ul>{conflict_items}</ul></details>")

        if st["id_reassignment_operations"]:
            id_items = "".join(
                f"<li>{escape(str(op['old_id']))} -> {escape(str(op['new_id']))}</li>"
                for op in st["id_reassignment_operations"]
            )
            details.append(f"<details><summary>ID reassignment operations ({len(st['id_reassignment_operations'])})</summary><ul>{id_items}</ul></details>")

        if st["merge_events"]:
            merge_rows = []
            for e in st["merge_events"][:100]:
                merge_rows.append(
                    "<tr>"
                    f"<td>{escape(e['source_file'])}</td>"
                    f"<td>{escape(str(e.get('source_study_id') or ''))}</td>"
                    f"<td>{escape(str(e['source_analysis_id']))}</td>"
                    f"<td>{escape(str(e.get('target_source_study_id') or ''))}</td>"
                    f"<td>{escape(str(e['target_representative_id']))}</td>"
                    f"<td>{e['combined_score']:.3f}</td>"
                    f"<td>{e['name_score']:.3f}</td>"
                    f"<td>{e['coord_score']:.3f}</td>"
                    f"<td>{escape(e['merge_reason'])}</td>"
                    "</tr>"
                )
            details.append(
                "<details><summary>Merge events (showing up to 100)</summary>"
                "<div class='table-wrap'><table><thead><tr><th>Source File</th><th>Source Study ID</th><th>Source Analysis ID</th><th>Target Study ID</th><th>Target Canonical ID</th><th>Combined</th><th>Name</th><th>Coord</th><th>Reason</th></tr></thead>"
                f"<tbody>{''.join(merge_rows)}</tbody></table></div></details>"
            )

        if st.get("unique_additions"):
            unique_detail_rows = []
            for e in st["unique_additions"][:100]:
                unique_detail_rows.append(
                    "<tr>"
                    f"<td>{escape(e['source_file'])}</td>"
                    f"<td>{escape(str(e.get('source_study_id') or ''))}</td>"
                    f"<td>{escape(str(e['source_analysis_id']))}</td>"
                    f"<td>{escape(str(e.get('best_target_source_study_id') or ''))}</td>"
                    f"<td>{escape(str(e['best_target_representative_id'] or ''))}</td>"
                    f"<td>{fmt_opt(e['best_combined_score'])}</td>"
                    f"<td>{fmt_opt(e['best_name_score'])}</td>"
                    f"<td>{fmt_opt(e['best_coord_score'])}</td>"
                    f"<td>{escape(', '.join(e.get('reason_codes', [])))}</td>"
                    "</tr>"
                )
            details.append(
                "<details><summary>Unique additions (showing up to 100)</summary>"
                "<div class='table-wrap'><table><thead><tr><th>Source File</th><th>Source Study ID</th><th>Source Analysis ID</th><th>Best Existing Study ID</th><th>Best Existing Canonical ID</th><th>Best Combined</th><th>Best Name</th><th>Best Coord</th><th>Reason</th></tr></thead>"
                f"<tbody>{''.join(unique_detail_rows)}</tbody></table></div></details>"
            )

        study_rows.append(
            "<tr>"
            f"<td>{escape(study_id)}</td>"
            f"<td>{int(st['source_analyses_count'])}</td>"
            f"<td>{int(st['canonical_analyses_count'])}</td>"
            f"<td>{float(st['dedup_percent']):.2f}%</td>"
            f"<td>{sum(1 for e in st['merge_events'] if e['merge_reason'] in ('threshold', 'threshold_and_exact'))}</td>"
            f"<td>{sum(1 for e in st['merge_events'] if e['merge_reason'] == 'exact_coord_override')}</td>"
            f"<td>{sum(1 for e in st.get('unique_additions', []) if not e.get('is_primary_source'))}</td>"
            f"<td>{len(st['id_reassignment_operations'])}</td>"
            f"<td><details><summary>Details</summary>{''.join(details)}</details></td>"
            "</tr>"
        )

    low_conf_rows = []
    for e in diagnostics.get("top_low_confidence_merges", []):
        low_conf_rows.append(
            "<tr>"
            f"<td>{escape(e['study_id'])}</td>"
            f"<td>{escape(e['source_file'])}</td>"
            f"<td>{escape(str(e.get('source_study_id') or ''))}</td>"
            f"<td>{escape(str(e['source_analysis_id']))}</td>"
            f"<td>{escape(str(e.get('target_source_study_id') or ''))}</td>"
            f"<td>{escape(str(e['target_representative_id']))}</td>"
            f"<td>{e['combined_score']:.3f}</td>"
            f"<td>{e['name_score']:.3f}</td>"
            f"<td>{e['coord_score']:.3f}</td>"
            f"<td>{escape(e['merge_reason'])}</td>"
            "</tr>"
        )

    dedup_pct = summary["dedup_rate"] * 100.0
    accepted_pct = (summary["accepted_fuzzy_merges"] / max(1, summary["input_analyses_total"])) * 100.0
    exact_override_pct = (summary["exact_override_merges"] / max(1, summary["input_analyses_total"])) * 100.0
    unique_subsequent_pct = (summary["unique_additions_subsequent_total"] / max(1, summary["input_analyses_total"])) * 100.0
    near_miss_count = len(near_miss_rows)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Fuzzy Merge Summary - {escape(config['project'])}</title>
  <style>
    body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: #f7f6f2; color: #1d2730; margin: 1rem; }}
    header, section {{ background: #fff; border: 1px solid #d8dde3; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d8dde3; padding: 0.4rem; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f5; }}
    .table-wrap {{ overflow-x: auto; }}
    .near-miss td {{ background: #fff6e5; }}
  </style>
</head>
<body>
  <header>
    <h1>Fuzzy Merge Summary: {escape(config['project'])}</h1>
    <p><strong>Threshold:</strong> {config['threshold']} | <strong>Exact coordinate override:</strong> {config['exact_coord_override']} | <strong>Assignment backend(s):</strong> {', '.join(config['assignment_backends_used'])}</p>
    <p><strong>Input files:</strong> {summary['input_files']} | <strong>Input studies (total rows):</strong> {summary['input_studies_total']} | <strong>Input unique studies:</strong> {summary['input_unique_studies']}</p>
    <p><strong>Input analyses:</strong> {summary['input_analyses_total']} | <strong>Merged analyses:</strong> {summary['merged_analyses']} | <strong>Dedup rate:</strong> {dedup_pct:.2f}%</p>
    <p><strong>Accepted fuzzy merges:</strong> {summary['accepted_fuzzy_merges']} ({accepted_pct:.2f}% of input analyses) | <strong>Exact override merges:</strong> {summary['exact_override_merges']} ({exact_override_pct:.2f}%)</p>
    <p><strong>Unique additions from subsequent files:</strong> {summary['unique_additions_subsequent_total']} ({unique_subsequent_pct:.2f}% of input analyses)</p>
    <p><strong>Potential near-miss unique additions (best combined >= 0.70):</strong> {near_miss_count}</p>
    <p><strong>Score stats:</strong> mean={summary['mean_combined_score']:.3f}, median={summary['median_combined_score']:.3f}, p25={summary['p25_combined_score']:.3f}, p75={summary['p75_combined_score']:.3f}</p>
    <p><strong>ID reassignment count:</strong> {summary['id_reassignment_count']} | <strong>Study metadata conflicts:</strong> {summary['study_metadata_conflict_count']} | <strong>Global duplicate analysis IDs:</strong> {summary['duplicate_analysis_ids_global']}</p>
  </header>

  <section>
    <h2>Per-Source-File Stats</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Source File</th><th>Source Key</th><th>Primary Source</th><th>Studies Present</th><th>Analyses Contributed</th><th>Analyses Retained Unique</th><th>Unique From Subsequent</th><th>Analyses Merged Into Existing</th></tr></thead>
        <tbody>{''.join(source_rows)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Unique Contributions From Subsequent Files</h2>
    <p>These are analyses that were not merged into existing canonicals and became new canonical analyses from non-primary files. Rows highlighted in amber are near-miss duplicates (best combined >= 0.70).</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Source File</th><th>Source Key</th><th>Study (Merged)</th><th>Source Study ID (Original)</th><th>Source Analysis ID</th><th>Source Analysis Name</th><th>Best Existing Study ID</th><th>Best Existing Canonical ID</th><th>Best Existing Canonical Name</th><th>Best Combined</th><th>Best Name</th><th>Best Coord</th><th>Best Exact Coords</th><th>Reason</th></tr></thead>
        <tbody>{''.join(unique_rows) if unique_rows else "<tr><td colspan='14'>No unique contributions from subsequent files.</td></tr>"}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Near-Miss Unique Additions</h2>
    <p>Review this subset first. These were kept unique but still have relatively high similarity to an existing canonical.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Source File</th><th>Source Key</th><th>Study (Merged)</th><th>Source Study ID (Original)</th><th>Source Analysis ID</th><th>Source Analysis Name</th><th>Best Existing Study ID</th><th>Best Existing Canonical ID</th><th>Best Existing Canonical Name</th><th>Best Combined</th><th>Best Name</th><th>Best Coord</th><th>Best Exact Coords</th><th>Reason</th></tr></thead>
        <tbody>{''.join(near_miss_rows) if near_miss_rows else "<tr><td colspan='14'>No near-miss unique additions.</td></tr>"}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Per-PMID Breakdown</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Study ID</th><th>Source Analyses</th><th>Canonical Analyses</th><th>Dedup %</th><th>Merges via Threshold</th><th>Merges via Exact Override</th><th>Unique From Subsequent</th><th>ID Reassign Ops</th><th>Details</th></tr></thead>
        <tbody>{''.join(study_rows)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Top Low-Confidence Merge Events</h2>
    <p>Lowest combined-score merged events (for manual review).</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Study ID</th><th>Source File</th><th>Source Study ID</th><th>Source Analysis ID</th><th>Target Study ID</th><th>Target Canonical ID</th><th>Combined</th><th>Name</th><th>Coord</th><th>Reason</th></tr></thead>
        <tbody>{''.join(low_conf_rows)}</tbody>
      </table>
    </div>
  </section>
</body>
</html>
"""


def validate_outputs(studyset: dict[str, Any], annotation: dict[str, Any], source_keys: list[str]) -> list[str]:
    errors: list[str] = []

    analysis_ids = set()
    for study in studyset.get("studies", []):
        for analysis in study.get("analyses", []):
            analysis_ids.add(clean_text(analysis.get("id")))

    for note in annotation.get("notes", []):
        aid = clean_text(note.get("analysis"))
        if aid not in analysis_ids:
            errors.append(f"Annotation references missing analysis ID: {aid}")

    note_keys = set(annotation.get("note_keys", {}).keys())
    expected = set(source_keys)
    if note_keys != expected:
        errors.append(f"note_keys mismatch. expected={sorted(expected)} got={sorted(note_keys)}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Project folder under data/nimads (e.g., social)")
    parser.add_argument("--nimads-root", type=Path, default=Path("data/nimads"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--exact-coord-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow exact coordinate-set matches to merge even if combined score < threshold",
    )
    parser.add_argument("--report-file", default="fuzzy_merge_summary.html")
    parser.add_argument("--diagnostics-file", default="fuzzy_merge_diagnostics.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    project_dir = args.nimads_root / args.project
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    output_dir = args.output_dir or (project_dir / "merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = discover_source_files(project_dir, output_dir)
    if not source_paths:
        raise RuntimeError(f"No source JSON files found in: {project_dir}")

    source_files: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for idx, path in enumerate(source_paths):
        payload = load_source_payload(path)
        source_key = sanitize_key(path.stem, used_keys)
        source_files.append(
            {
                "path": path,
                "filename": path.name,
                "source_key": source_key,
                "source_order": idx,
                "payload": payload,
            }
        )

    merged_studyset, merged_annotation, diagnostics, report_html = merge_project(
        project=args.project,
        source_files=source_files,
        threshold=args.threshold,
        exact_coord_override=args.exact_coord_override,
    )

    validation_errors = validate_outputs(merged_studyset, merged_annotation, [src["source_key"] for src in source_files])
    if validation_errors:
        diagnostics.setdefault("validation_errors", []).extend(validation_errors)

    studyset_path = output_dir / "nimads_studyset.json"
    annotation_path = output_dir / "nimads_annotation.json"
    diagnostics_path = output_dir / args.diagnostics_file
    report_path = output_dir / args.report_file

    studyset_path.write_text(json.dumps(merged_studyset, indent=2), encoding="utf-8")
    annotation_path.write_text(json.dumps(merged_annotation, indent=2), encoding="utf-8")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    report_path.write_text(report_html, encoding="utf-8")

    summary = diagnostics["summary"]
    print(
        f"Wrote merged outputs to {output_dir} | "
        f"files={summary['input_files']} studies={summary['input_unique_studies']} "
        f"analyses_in={summary['input_analyses_total']} analyses_out={summary['merged_analyses']} "
        f"dedup_rate={summary['dedup_rate']:.3f}"
    )

    if validation_errors:
        print(f"Validation warnings: {len(validation_errors)}")
        if args.verbose:
            for err in validation_errors:
                print(f"  - {err}")


if __name__ == "__main__":
    main()
