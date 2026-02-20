#!/usr/bin/env python3
"""
Run meta-analyses for merged NiMADS datasets.

This script discovers merged NiMADS studysets in data/nimads/<project>/merged,
expects a companion nimads_annotation.json file, and runs one meta-analysis per
annotation boolean note key by slicing analyses where that key is True.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import NiMARE dependencies
try:
    from nimare.correct import FDRCorrector, FWECorrector
    from nimare.workflows import CBMAWorkflow
    from nimare.meta.cbma import MKDADensity, ALE, KDA
    from nimare.nimads import Studyset
    from nimare.reports.base import run_reports

    NIMARE_AVAILABLE = True
except ImportError:
    NIMARE_AVAILABLE = False
    print(
        "WARNING: NiMARE is not installed. "
        "Please install with: pip install nimare",
        file=sys.stderr,
    )


def create_estimator(estimator_name: str, estimator_args: Dict):
    """Create an estimator instance based on the name and arguments."""
    if not NIMARE_AVAILABLE:
        raise ImportError("NiMARE is required but not installed")

    estimator_map = {
        "ale": ALE,
        "mkdadensity": MKDADensity,
        "kda": KDA,
    }

    if estimator_name not in estimator_map:
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    estimator_class = estimator_map[estimator_name]
    return estimator_class(**estimator_args)


def create_corrector(corrector_name: str, corrector_args: Dict):
    """Create a corrector instance based on the name and arguments."""
    if not NIMARE_AVAILABLE:
        raise ImportError("NiMARE is required but not installed")

    if corrector_name == "fdr":
        return FDRCorrector(**corrector_args)
    if corrector_name == "montecarlo":
        return FWECorrector(method="montecarlo", **corrector_args)
    if corrector_name == "bonferroni":
        return FWECorrector(method="bonferroni", **corrector_args)

    raise ValueError(f"Unsupported corrector: {corrector_name}")


def find_merged_nimads_files(data_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Find merged NiMADS studyset/annotation files organized by project.

    Expected per-project paths:
      data/nimads/<project>/merged/nimads_studyset.json
      data/nimads/<project>/merged/nimads_annotation.json
    """
    nimads_dir = data_dir / "nimads"

    if not nimads_dir.exists():
        raise FileNotFoundError(f"NiMADS directory not found: {nimads_dir}")

    project_files: Dict[str, Dict[str, Path]] = {}
    for project_dir in sorted(p for p in nimads_dir.iterdir() if p.is_dir()):
        merged_dir = project_dir / "merged"
        studyset_file = merged_dir / "nimads_studyset.json"
        annotation_file = merged_dir / "nimads_annotation.json"

        if studyset_file.exists() and annotation_file.exists():
            project_files[project_dir.name] = {
                "studyset": studyset_file,
                "annotation": annotation_file,
            }
        elif studyset_file.exists() and not annotation_file.exists():
            raise FileNotFoundError(
                f"Found merged studyset but missing companion annotation file: {annotation_file}"
            )

    return project_files


def load_include_ids(include_file: Optional[Path]) -> Optional[set[str]]:
    """
    Load PMID/study IDs from a text file (one ID per line).

    The IDs are applied as a post-hoc restriction after annotation slicing.
    """
    if include_file is None:
        return None

    if not include_file.exists():
        raise FileNotFoundError(f"Include file not found: {include_file}")

    with include_file.open("r") as f:
        ids = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(ids)} PMID IDs to include from {include_file}")
    return ids


def maybe_sanitize_nimads_payloads(studyset_data: Dict, annotation_data) -> Tuple[Dict, Dict]:
    """Sanitize NiMADS payloads using autonima helpers when available."""
    if isinstance(annotation_data, list):
        annotation_data = annotation_data[0] if annotation_data else {}

    try:
        from autonima.coordinates.nimads_models import (
            sanitize_studyset_dict,
            sanitize_annotation_dict,
        )
    except Exception:
        print(
            "Warning: Could not import autonima sanitizers; using raw NiMADS payloads.",
            file=sys.stderr,
        )
        return studyset_data, annotation_data

    return sanitize_studyset_dict(studyset_data), sanitize_annotation_dict(annotation_data)


def load_project_payloads(studyset_file: Path, annotation_file: Path) -> Tuple[Dict, Dict]:
    """Load and sanitize merged studyset/annotation payloads."""
    with studyset_file.open("r") as f:
        studyset_data = json.load(f)

    with annotation_file.open("r") as f:
        annotation_data = json.load(f)

    studyset_data, annotation_data = maybe_sanitize_nimads_payloads(
        studyset_data, annotation_data
    )

    if not isinstance(annotation_data, dict):
        raise ValueError(f"Invalid annotation payload in {annotation_file}: expected object")

    note_keys = annotation_data.get("note_keys", {})
    notes = annotation_data.get("notes", [])

    if not isinstance(note_keys, dict) or not note_keys:
        raise ValueError(f"Invalid annotation payload in {annotation_file}: missing note_keys")
    if not isinstance(notes, list):
        raise ValueError(f"Invalid annotation payload in {annotation_file}: notes must be a list")

    return studyset_data, annotation_data


def get_boolean_note_keys(annotation_data: Dict) -> List[str]:
    """Return annotation note keys with boolean type."""
    note_keys = annotation_data.get("note_keys", {})
    return [key for key, key_type in note_keys.items() if key_type == "boolean"]


def get_analysis_ids_for_note(annotation_data: Dict, note_key: str) -> List[str]:
    """Return analysis IDs where annotation note_key evaluates True."""
    analysis_ids: List[str] = []

    for row in annotation_data.get("notes", []):
        analysis_id = str(row.get("analysis", "")).strip()
        note = row.get("note", {})
        if not analysis_id or not isinstance(note, dict):
            continue
        if bool(note.get(note_key, False)):
            analysis_ids.append(analysis_id)

    return analysis_ids


def run_meta_analysis_for_note_key(
    studyset: Studyset,
    annotation_data: Dict,
    note_key: str,
    output_dir: Path,
    estimator_name: str = "mkdadensity",
    estimator_args: Optional[Dict] = None,
    corrector_name: str = "fdr",
    corrector_args: Optional[Dict] = None,
    include_ids: Optional[set[str]] = None,
):
    """Run meta-analysis for one annotation boolean note key."""
    note_type = annotation_data.get("note_keys", {}).get(note_key)
    if note_type != "boolean":
        print(f"Skipping note key '{note_key}' (non-boolean type: {note_type})")
        return None

    analysis_ids = get_analysis_ids_for_note(annotation_data, note_key)
    if not analysis_ids:
        print(f"Skipping note key '{note_key}' (0 analyses in annotation slice)")
        return None

    print(f"\n{'=' * 80}")
    print(f"Running note key: {note_key}")
    print(f"Slicing to {len(analysis_ids)} analyses from annotation")
    print(f"{'=' * 80}")

    sliced_studyset = studyset.slice(analyses=analysis_ids)

    if include_ids is not None:
        filtered_analysis_ids = [
            analysis.id
            for study in sliced_studyset.studies
            if study.id in include_ids
            for analysis in study.analyses
        ]
        sliced_studyset = sliced_studyset.slice(analyses=filtered_analysis_ids)
        restricted_studies = len(sliced_studyset.studies)
        restricted_analyses = sum(len(study.analyses) for study in sliced_studyset.studies)
        print(
            "Applied post-hoc PMID filter: "
            f"{restricted_analyses} analyses across {restricted_studies} studies"
        )

    for study in sliced_studyset.studies:
        study.name = study.id
        for analysis in study.analyses:
            analysis.name = analysis.id

    dataset = sliced_studyset.to_dataset()
    if len(dataset.ids) == 0:
        print(f"Skipping note key '{note_key}' (slice produced 0 analyses in dataset)")
        return None

    print(f"Dataset contains {len(dataset.ids)} analyses")

    estimator = create_estimator(estimator_name, estimator_args or {})
    corrector = create_corrector(corrector_name, corrector_args or {})

    print(
        f"Running meta-analysis with {estimator_name} estimator "
        f"and {corrector_name} corrector..."
    )
    workflow = CBMAWorkflow(
        estimator=estimator,
        corrector=corrector,
        diagnostics="focuscounter",
        output_dir=str(output_dir),
    )

    meta_results = workflow.fit(dataset)

    print("Generating reports...")
    run_reports(meta_results, str(output_dir))

    print(f"✓ Results saved to: {output_dir}")
    return meta_results


def main():
    """Main function to run merged NiMADS annotation-sliced meta-analyses."""
    parser = argparse.ArgumentParser(
        description="Run meta-analyses on merged NiMADS datasets (one run per annotation boolean note key)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings on all merged projects
  python run_meta_analyses.py

  # Run with specific estimator and corrector
  python run_meta_analyses.py --estimator ale --corrector montecarlo

  # Process only selected merged projects
  python run_meta_analyses.py --projects social reward

  # Post-hoc restrict to a PMID list after annotation slicing
  python run_meta_analyses.py --projects social --include-ids data/pmids_social.txt

  # Use custom data and analysis directories
  python run_meta_analyses.py --data-dir /path/to/data \\
      --analysis-dir /path/to/analysis
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory with nimads/ subdirectory (default: data)",
    )

    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis"),
        help="Path to the output analysis directory (default: analysis)",
    )

    parser.add_argument(
        "--projects",
        nargs="+",
        help="Specific projects to process (default: all projects with merged studyset+annotation)",
    )

    parser.add_argument(
        "--estimator",
        choices=["ale", "mkdadensity", "kda"],
        default="mkdadensity",
        help="CBMA estimator to use (default: mkdadensity)",
    )

    parser.add_argument(
        "--estimator-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the estimator (default: {})",
    )

    parser.add_argument(
        "--corrector",
        choices=["fdr", "montecarlo", "bonferroni"],
        default="fdr",
        help="Corrector to use (default: fdr)",
    )

    parser.add_argument(
        "--corrector-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the corrector (default: {})",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip analyses that already have output directories",
    )

    parser.add_argument(
        "--include-ids",
        type=Path,
        default=None,
        help="Path to text file with PMID/study IDs to include (one per line), applied after annotation slicing.",
    )

    args = parser.parse_args()

    if not NIMARE_AVAILABLE:
        print(
            "ERROR: NiMARE is not installed. Please install it first:",
            file=sys.stderr,
        )
        print("  pip install nimare", file=sys.stderr)
        return 1

    try:
        estimator_args = json.loads(args.estimator_args)
        corrector_args = json.loads(args.corrector_args)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}", file=sys.stderr)
        return 1

    try:
        include_ids = load_include_ids(args.include_ids)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    print(f"Searching for merged NiMADS files in {args.data_dir / 'nimads'}...")
    try:
        project_files = find_merged_nimads_files(args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.projects:
        project_files = {k: v for k, v in project_files.items() if k in args.projects}
        if not project_files:
            available = ", ".join(find_merged_nimads_files(args.data_dir).keys())
            print(
                f"Error: No matching projects found. "
                f"Available merged projects: {available}",
                file=sys.stderr,
            )
            return 1

    if not project_files:
        print(
            "Error: No merged NiMADS projects found with both "
            "merged/nimads_studyset.json and merged/nimads_annotation.json",
            file=sys.stderr,
        )
        return 1

    print(f"\nFound {len(project_files)} merged project(s):")
    project_note_keys: Dict[str, List[str]] = {}
    project_payloads: Dict[str, Tuple[Dict, Dict]] = {}
    for project, paths in project_files.items():
        studyset_data, annotation_data = load_project_payloads(paths["studyset"], paths["annotation"])
        project_payloads[project] = (studyset_data, annotation_data)
        bool_keys = get_boolean_note_keys(annotation_data)
        project_note_keys[project] = bool_keys
        print(
            f"  - {project}: {len(bool_keys)} boolean note key(s) "
            f"from {paths['annotation'].relative_to(args.data_dir)}"
        )

    total_analyses = sum(len(keys) for keys in project_note_keys.values())
    completed = 0
    failed = 0
    skipped = 0

    for project, paths in project_files.items():
        studyset_file = paths["studyset"]
        annotation_file = paths["annotation"]
        studyset_data, annotation_data = project_payloads[project]

        print(f"\n{'#' * 80}")
        print(f"# Project: {project}")
        print(f"# Studyset: {studyset_file}")
        print(f"# Annotation: {annotation_file}")
        print(f"{'#' * 80}")

        print("Creating studyset...")
        studyset = Studyset(studyset_data)

        for note_key in get_boolean_note_keys(annotation_data):
            output_dir = args.analysis_dir / project / note_key

            if args.skip_existing and output_dir.exists() and list(output_dir.iterdir()):
                print(f"\n⊘ Skipping {project}/{note_key} (output already exists)")
                skipped += 1
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = run_meta_analysis_for_note_key(
                    studyset,
                    annotation_data,
                    note_key,
                    output_dir,
                    estimator_name=args.estimator,
                    estimator_args=estimator_args,
                    corrector_name=args.corrector,
                    corrector_args=corrector_args,
                    include_ids=include_ids,
                )
                if result is None:
                    skipped += 1
                else:
                    completed += 1
            except Exception as e:
                print(
                    f"\n✗ Error processing {project}/{note_key}: {e}",
                    file=sys.stderr,
                )
                import traceback

                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Total annotation-sliced analyses: {total_analyses}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
