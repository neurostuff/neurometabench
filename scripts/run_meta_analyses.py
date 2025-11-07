#!/usr/bin/env python3
"""
Run meta-analyses for all NiMADS datasets found in data/nimads/

This script discovers all NiMADS JSON files in the data/nimads directory,
runs meta-analyses on each one, and deposits results in analysis/{topic-name}/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
        file=sys.stderr
    )


def create_estimator(estimator_name: str, estimator_args: Dict):
    """Create an estimator instance based on the name and arguments."""
    if not NIMARE_AVAILABLE:
        raise ImportError("NiMARE is required but not installed")
    
    # Map estimator names to classes
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
    
    # Map corrector names to classes
    if corrector_name == "fdr":
        return FDRCorrector(**corrector_args)
    elif corrector_name == "montecarlo":
        return FWECorrector(method="montecarlo", **corrector_args)
    elif corrector_name == "bonferroni":
        return FWECorrector(method="bonferroni", **corrector_args)
    else:
        raise ValueError(f"Unsupported corrector: {corrector_name}")


def find_nimads_files(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all NiMADS JSON files organized by topic.
    
    Args:
        data_dir: Path to the data/nimads directory
        
    Returns:
        Dictionary mapping topic names to lists of JSON file paths
    """
    nimads_dir = data_dir / "nimads"
    
    if not nimads_dir.exists():
        raise FileNotFoundError(f"NiMADS directory not found: {nimads_dir}")
    
    # Find all topic directories
    topic_files = {}
    for topic_dir in nimads_dir.iterdir():
        if topic_dir.is_dir():
            # Find all JSON files in this topic directory
            json_files = list(topic_dir.glob("*.json"))
            if json_files:
                topic_files[topic_dir.name] = sorted(json_files)
    
    return topic_files


def run_meta_analysis(
    studyset_file: Path,
    output_dir: Path,
    estimator_name: str = "mkdadensity",
    estimator_args: Optional[Dict] = None,
    corrector_name: str = "fdr",
    corrector_args: Optional[Dict] = None
):
    """
    Run meta-analysis on a single NiMADS studyset file.
    
    Args:
        studyset_file: Path to the NiMADS JSON file
        output_dir: Directory to save results
        estimator_name: Name of the CBMA estimator to use
        estimator_args: Arguments for the estimator
        corrector_name: Name of the corrector to use
        corrector_args: Arguments for the corrector
    """
    if not NIMARE_AVAILABLE:
        raise ImportError("NiMARE is required but not installed")
    
    print(f"\n{'='*80}")
    print(f"Processing: {studyset_file.name}")
    print(f"{'='*80}")
    
    # Load the studyset
    print("Loading studyset JSON...")
    with open(studyset_file, 'r') as f:
        studyset_data = json.load(f)
    
    print("Creating studyset...")
    studyset = Studyset(studyset_data)
    
    # Ensure uniqueness of study and analysis names
    print("Ensuring unique IDs...")
    for study in studyset.studies:
        study.name = study.id
        for analysis in study.analyses:
            analysis.name = analysis.id
    
    # Convert to dataset
    print("Converting to NiMARE dataset...")
    dataset = studyset.to_dataset()
    
    print(f"Dataset contains {len(dataset.ids)} analyses")
    
    # Set up estimator and corrector
    estimator = create_estimator(estimator_name, estimator_args or {})
    corrector = create_corrector(corrector_name, corrector_args or {})
    
    # Run meta-analysis
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
    
    # Generate reports
    print("Generating reports...")
    run_reports(meta_results, str(output_dir))
    
    print(f"✓ Results saved to: {output_dir}")
    
    return meta_results


def main():
    """Main function to run meta-analyses on all NiMADS datasets."""
    parser = argparse.ArgumentParser(
        description="Run meta-analyses on all NiMADS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (MKDADensity + FDR)
  python run_meta_analyses.py
  
  # Run with specific estimator and corrector
  python run_meta_analyses.py --estimator ale --corrector montecarlo
  
  # Process only specific topics
  python run_meta_analyses.py --topics emotion reward
  
  # Use custom data and analysis directories
  python run_meta_analyses.py --data-dir /path/to/data \\
      --analysis-dir /path/to/analysis
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory with nimads/ subdirectory (default: data)"
    )
    
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis"),
        help="Path to the output analysis directory (default: analysis)"
    )
    
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to process (default: all topics found)"
    )
    
    parser.add_argument(
        "--estimator",
        choices=["ale", "mkdadensity", "kda"],
        default="mkdadensity",
        help="CBMA estimator to use (default: mkdadensity)"
    )
    
    parser.add_argument(
        "--estimator-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the estimator (default: {})"
    )
    
    parser.add_argument(
        "--corrector",
        choices=["fdr", "montecarlo", "bonferroni"],
        default="fdr",
        help="Corrector to use (default: fdr)"
    )
    
    parser.add_argument(
        "--corrector-args",
        type=str,
        default="{}",
        help="JSON string of arguments for the corrector (default: {})"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip analyses that already have output directories"
    )
    
    args = parser.parse_args()
    
    # Check if NiMARE is available
    if not NIMARE_AVAILABLE:
        print(
            "ERROR: NiMARE is not installed. Please install it first:",
            file=sys.stderr
        )
        print("  pip install nimare", file=sys.stderr)
        return 1
    
    # Parse estimator and corrector arguments
    try:
        estimator_args = json.loads(args.estimator_args)
        corrector_args = json.loads(args.corrector_args)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}", file=sys.stderr)
        return 1
    
    # Check if data directory exists
    if not args.data_dir.exists():
        print(
            f"Error: Data directory not found: {args.data_dir}",
            file=sys.stderr
        )
        return 1
    
    # Find all NiMADS files
    print(f"Searching for NiMADS files in {args.data_dir / 'nimads'}...")
    try:
        topic_files = find_nimads_files(args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Filter topics if specified
    if args.topics:
        topic_files = {
            k: v for k, v in topic_files.items() if k in args.topics
        }
        if not topic_files:
            available = ', '.join(find_nimads_files(args.data_dir).keys())
            print(
                f"Error: No matching topics found. "
                f"Available topics: {available}",
                file=sys.stderr
            )
            return 1
    
    print(f"\nFound {len(topic_files)} topics:")
    for topic, files in topic_files.items():
        print(f"  - {topic}: {len(files)} file(s)")
    
    # Process each topic and file
    total_analyses = sum(len(files) for files in topic_files.values())
    completed = 0
    failed = 0
    skipped = 0
    
    for topic, files in topic_files.items():
        print(f"\n{'#'*80}")
        print(f"# Topic: {topic}")
        print(f"{'#'*80}")
        
        for json_file in files:
            # Create output directory for this specific analysis
            # Use the JSON filename (without extension) as the analysis name
            analysis_name = json_file.stem
            output_dir = args.analysis_dir / topic / analysis_name
            
            # Check if we should skip existing analyses
            if (args.skip_existing and output_dir.exists() and
                    list(output_dir.iterdir())):
                print(f"\n⊘ Skipping {json_file.name} (output already exists)")
                skipped += 1
                continue
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                run_meta_analysis(
                    json_file,
                    output_dir,
                    estimator_name=args.estimator,
                    estimator_args=estimator_args,
                    corrector_name=args.corrector,
                    corrector_args=corrector_args
                )
                completed += 1
            except Exception as e:
                print(
                    f"\n✗ Error processing {json_file.name}: {e}",
                    file=sys.stderr
                )
                import traceback
                traceback.print_exc()
                failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total analyses: {total_analyses}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())