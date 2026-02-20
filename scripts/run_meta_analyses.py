#!/usr/bin/env python3
"""
Run meta-analyses for merged NiMADS datasets via autonima.meta.

This script discovers merged NiMADS studysets in data/nimads/<project>/merged,
expects a companion nimads_annotation.json file, and delegates meta-analysis
execution to autonima.meta.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from autonima import meta as autonima_meta

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


def load_boolean_note_keys(annotation_file: Path) -> List[str]:
    """Read annotation note_keys and return boolean keys only."""
    with annotation_file.open("r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        payload = payload[0] if payload else {}

    note_keys = payload.get("note_keys", {})
    if not isinstance(note_keys, dict):
        return []

    return [key for key, key_type in note_keys.items() if key_type == "boolean"]


def main() -> int:
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

    try:
        estimator_args = json.loads(args.estimator_args)
        corrector_args = json.loads(args.corrector_args)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}", file=sys.stderr)
        return 1

    include_ids: Optional[set[str]] = None
    try:
        include_ids = autonima_meta.load_include_ids(args.include_ids)
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
    for project, paths in project_files.items():
        note_keys = load_boolean_note_keys(paths["annotation"])
        project_note_keys[project] = note_keys
        print(
            f"  - {project}: {len(note_keys)} boolean note key(s) "
            f"from {paths['annotation'].relative_to(args.data_dir)}"
        )

    total_analyses = sum(len(keys) for keys in project_note_keys.values())
    completed = 0
    failed = 0
    skipped = 0

    for project, paths in project_files.items():
        studyset_file = paths["studyset"]
        annotation_file = paths["annotation"]
        note_keys = project_note_keys[project]

        print(f"\n{'#' * 80}")
        print(f"# Project: {project}")
        print(f"# Studyset: {studyset_file}")
        print(f"# Annotation: {annotation_file}")
        print(f"{'#' * 80}")

        project_output_dir = args.analysis_dir / project

        try:
            results = autonima_meta.run_meta_analyses_from_files(
                studyset_file=studyset_file,
                annotation_file=annotation_file,
                output_dir=project_output_dir,
                estimator_name=args.estimator,
                estimator_args=estimator_args,
                corrector_name=args.corrector,
                corrector_args=corrector_args,
                include_ids=include_ids,
                skip_existing=args.skip_existing,
                columns=note_keys,
            )
            completed += len(results)
            skipped += max(0, len(note_keys) - len(results))
        except Exception as e:
            print(f"\n✗ Error processing project {project}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            failed += len(note_keys)

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
