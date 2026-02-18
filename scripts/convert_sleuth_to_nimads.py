#!/usr/bin/env python
"""
Convert Sleuth format files to NIMADS format.

This script reads meta_datasets.tsv, finds corresponding Sleuth files in
raw/meta-datasets/, converts them to NIMADS format using nimare, and
saves them to data/nimads/{pmid}/.

Files with the same base name but different coordinate space suffixes
(-MNI, -Tal, -Talairach) are merged into a single file with -Merged suffix.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
from difflib import SequenceMatcher
import nimare.io

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def parse_reference_convention(value: str) -> Tuple[str, str]:
    """
    Parse study/analysis names from Sleuth convention text.

    Expected pattern:
    "Study; Analysis; OptionalTag"
    """
    text = str(value or "").strip()
    if not text:
        return "", ""

    parts = [part.strip() for part in text.split(";")]
    study_label = parts[0] if parts else text
    analysis_label = parts[1] if len(parts) > 1 else ""
    return study_label, analysis_label


def ensure_unique_analysis_ids(analyses: List[Dict[str, Any]]) -> None:
    """Ensure unique analysis IDs within one study, suffixing repeated IDs."""
    seen: Dict[str, int] = {}
    for idx, analysis in enumerate(analyses):
        base_id = str(analysis.get("id", "")).strip() or str(analysis.get("name", "")).strip() or f"analysis_{idx}"
        if base_id not in seen:
            seen[base_id] = 1
            analysis["id"] = base_id
            continue

        seen[base_id] += 1
        analysis["id"] = f"{base_id}__{seen[base_id]}"


def normalize_studies_by_reference(nimads_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Group studies by first ';' segment and move second segment to analysis id/name.

    This enforces the project convention:
    - study id/name: first segment
    - analysis id/name: second segment
    - trailing segment(s): ignored
    """
    studies = nimads_dict.get("studies", [])
    if not isinstance(studies, list):
        return nimads_dict

    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for raw_study in studies:
        if not isinstance(raw_study, dict):
            continue

        raw_id = raw_study.get("id", "")
        raw_name = raw_study.get("name", "")
        source_text = raw_id if str(raw_id).strip() else raw_name
        study_label, analysis_label = parse_reference_convention(source_text)

        study_key = study_label.strip() or str(source_text).strip()
        if not study_key:
            continue

        if study_key not in grouped:
            canonical = {
                "id": study_key,
                "name": study_key,
                "authors": raw_study.get("authors", ""),
                "publication": raw_study.get("publication", ""),
                "metadata": raw_study.get("metadata", {}) if isinstance(raw_study.get("metadata", {}), dict) else {},
                "analyses": [],
            }
            for key, value in raw_study.items():
                if key not in canonical and key not in {"analyses"}:
                    canonical[key] = value
            grouped[study_key] = canonical
            order.append(study_key)
        else:
            canonical = grouped[study_key]
            if _is_empty(canonical.get("authors")) and not _is_empty(raw_study.get("authors")):
                canonical["authors"] = raw_study.get("authors")
            if _is_empty(canonical.get("publication")) and not _is_empty(raw_study.get("publication")):
                canonical["publication"] = raw_study.get("publication")
            incoming_metadata = raw_study.get("metadata", {})
            if isinstance(incoming_metadata, dict):
                for key, value in incoming_metadata.items():
                    if key not in canonical["metadata"] or _is_empty(canonical["metadata"].get(key)):
                        canonical["metadata"][key] = value

        analyses = raw_study.get("analyses", [])
        if not isinstance(analyses, list):
            analyses = []

        for idx, analysis in enumerate(analyses):
            if not isinstance(analysis, dict):
                continue
            normalized_analysis = dict(analysis)
            label = analysis_label or str(normalized_analysis.get("name", "")).strip() or str(normalized_analysis.get("id", "")).strip() or f"analysis_{idx}"
            normalized_analysis["name"] = label
            normalized_analysis["id"] = label
            grouped[study_key]["analyses"].append(normalized_analysis)

    normalized_studies = [grouped[key] for key in order]
    for study in normalized_studies:
        ensure_unique_analysis_ids(study.get("analyses", []))

    nimads_dict["studies"] = normalized_studies
    return nimads_dict


def load_pmid_mappings(csv_path: Path) -> Tuple[Dict[str, str], List[str]]:
    """
    Load nimads_study_id -> matched_pmid mappings from a fuzzy-match CSV.

    Matches map_pmcid_to_nimads.py behavior:
    - skip no_match
    - include exact_match/manual_override always when matched_pmid present
    - include fuzzy_match/multiple_matches only when the nimads_study_id appears once
    """
    warnings: List[str] = []
    mappings: Dict[str, str] = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"nimads_study_id", "match_status", "matched_pmid"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            warnings.append(
                f"Mapping file {csv_path.name} missing required columns: {required_cols}"
            )
            return mappings, warnings

        rows = list(reader)

    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        study_id = str(row.get("nimads_study_id", "")).strip()
        if study_id:
            counts[study_id] += 1

    for row in rows:
        study_id = str(row.get("nimads_study_id", "")).strip()
        status = str(row.get("match_status", "")).strip()
        pmid = str(row.get("matched_pmid", "")).strip()
        if not study_id:
            continue

        if status == "no_match":
            continue

        if status in {"fuzzy_match", "multiple_matches"} and counts[study_id] > 1:
            continue

        if status in {"exact_match", "manual_override", "fuzzy_match", "multiple_matches"} and pmid:
            if study_id in mappings and mappings[study_id] != pmid:
                warnings.append(
                    f"Conflicting PMIDs for {study_id} in {csv_path.name}: {mappings[study_id]} vs {pmid}. Keeping first."
                )
                continue
            mappings[study_id] = pmid

    return mappings, warnings


def apply_optional_pmid_mapping(
    nimads_dict: Dict[str, Any],
    project_key: str,
    dataset_stem: str,
    fuzzy_map_dir: Optional[Path],
) -> Tuple[Dict[str, Any], Optional[Path], int, int, int, List[str]]:
    """
    Replace study IDs with mapped PMIDs when a matching fuzzy-map CSV is available.

    Returns:
        (updated_dict, mapping_csv_path_or_none, total_studies, mapped_count, not_mapped_count, warnings)
    """
    studies = nimads_dict.get("studies", [])
    if not isinstance(studies, list):
        return nimads_dict, None, 0, 0, 0, []

    total_studies = len(studies)
    if fuzzy_map_dir is None:
        return nimads_dict, None, total_studies, 0, total_studies, []

    csv_path = fuzzy_map_dir / project_key / f"matched_studies_{dataset_stem}.csv"
    if not csv_path.exists():
        return nimads_dict, None, total_studies, 0, total_studies, []

    mappings, warnings = load_pmid_mappings(csv_path)
    if not mappings:
        return nimads_dict, csv_path, total_studies, 0, total_studies, warnings

    mapped_count = 0
    for study in studies:
        study_id = str(study.get("id", "")).strip()
        if study_id in mappings:
            study["id"] = mappings[study_id]
            mapped_count += 1

    return nimads_dict, csv_path, total_studies, mapped_count, total_studies - mapped_count, warnings


def normalize_folder_name(topic_name: str) -> str:
    """
    Normalize a topic name to create a valid, human-friendly folder name.
    
    Parameters
    ----------
    topic_name : str
        The topic name from meta_datasets.tsv
        
    Returns
    -------
    str
        A normalized folder name (lowercase, spaces->underscores,
        special chars removed)
    """
    # Convert to lowercase
    normalized = topic_name.lower()
    
    # Replace spaces and hyphens with underscores
    normalized = normalized.replace(' ', '_').replace('-', '_')
    
    # Remove parentheses and their contents
    normalized = re.sub(r'\([^)]*\)', '', normalized)
    
    # Remove special characters, keep only alphanumeric and underscores
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    
    # Remove multiple consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    return normalized


def fuzzy_match_folder(
    topic_name: str,
    available_folders: List[str],
    threshold: float = 0.6
) -> Optional[str]:
    """
    Find the best matching folder name for a given topic using fuzzy matching.
    
    Parameters
    ----------
    topic_name : str
        The topic name from meta_datasets.tsv
    available_folders : list of str
        List of available folder names in raw/meta-datasets/
    threshold : float
        Minimum similarity score (0-1) to consider a match
        
    Returns
    -------
    str or None
        The best matching folder name, or None if no good match found
    """
    best_match = None
    best_score = 0.0
    
    # Normalize topic name for comparison
    topic_normalized = topic_name.lower().strip()
    
    for folder in available_folders:
        # Extract the main part of the folder name (after the number prefix)
        # e.g., "1. Social - Pintos Lobo" -> "Social - Pintos Lobo"
        folder_parts = folder.split('. ', 1)
        if len(folder_parts) > 1:
            folder_main = folder_parts[1].lower().strip()
        else:
            folder_main = folder.lower().strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(
            None, topic_normalized, folder_main
        ).ratio()
        
        # Also check if topic words are in folder name
        topic_words = set(topic_normalized.split())
        folder_words = set(folder_main.split())
        word_overlap = len(topic_words & folder_words) / max(
            len(topic_words), 1
        )
        
        # Combined score
        score = (similarity * 0.7) + (word_overlap * 0.3)
        
        if score > best_score:
            best_score = score
            best_match = folder
    
    if best_score >= threshold:
        return best_match
    
    return None


def convert_sleuth_files(
    sleuth_dir: Path,
    output_dir: Path,
    pmid: str,
    topic: str,
    project_key: str,
    fuzzy_map_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Convert all Sleuth .txt files in a directory to NIMADS format.
    
    Parameters
    ----------
    sleuth_dir : Path
        Directory containing Sleuth .txt files
    output_dir : Path
        Directory where NIMADS files will be saved
    pmid : str
        PMID identifier for the study
    topic : str
        Topic name for the studyset
        
    Returns
    -------
    dict
        Statistics about the conversion (files converted, errors, etc.)
    """
    stats = {
        'files_found': 0,
        'files_converted': 0,
        'errors': []
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files in the directory
    txt_files = list(sleuth_dir.glob('*.txt'))
    stats['files_found'] = len(txt_files)
    
    if not txt_files:
        stats['errors'].append(f"No .txt files found in {sleuth_dir}")
        return stats
    
    # Convert each file
    for txt_file in txt_files:
        try:
            # Determine the base name for output
            file_stem = txt_file.stem
            
            # Generate studyset_id and name
            studyset_id = f"nimads_{pmid}_{file_stem}"
            studyset_name = f"{topic} - {file_stem}"
            
            # Convert Sleuth to NIMADS
            # Don't specify target (use None) - this preserves the original
            # space from the file's //Reference= tag without conversion
            nimads_dict = nimare.io.convert_sleuth_to_nimads_dict(
                text_file=str(txt_file),
                studyset_id=studyset_id,
                studyset_name=studyset_name
            )
            original_study_count = len(nimads_dict.get("studies", []))
            nimads_dict = normalize_studies_by_reference(nimads_dict)
            normalized_study_count = len(nimads_dict.get("studies", []))
            nimads_dict, mapping_csv, mapped_total, mapped_count, not_mapped_count, mapping_warnings = apply_optional_pmid_mapping(
                nimads_dict=nimads_dict,
                project_key=project_key,
                dataset_stem=file_stem,
                fuzzy_map_dir=fuzzy_map_dir,
            )
            for warning in mapping_warnings:
                print(f"  ⚠ Mapping warning: {warning}")
            
            # Save as JSON
            output_file = output_dir / f"{file_stem}.json"
            with open(output_file, 'w') as f:
                json.dump(nimads_dict, f, indent=2)
            
            stats['files_converted'] += 1
            print(
                f"  ✓ Converted: {txt_file.name} -> {output_file.name} "
                f"(studies: {original_study_count} -> {normalized_study_count})"
            )
            if mapping_csv is not None:
                print(
                    f"    PMID mapping from {mapping_csv.name}: {mapped_count}/{mapped_total} "
                    f"(unmapped={not_mapped_count})"
                )
            
        except Exception as e:
            error_msg = f"Error converting {txt_file.name}: {str(e)}"
            stats['errors'].append(error_msg)
            print(f"  ✗ {error_msg}")
    
    return stats


def get_base_name_and_suffix(filename: str) -> tuple:
    """
    Extract base name and coordinate space suffix from filename.
    
    Parameters
    ----------
    filename : str
        The filename (without extension)
        
    Returns
    -------
    tuple
        (base_name, suffix) where suffix is one of 'MNI', 'Tal', 'Talairach', or None
    """
    # Check for coordinate space suffixes
    for suffix in ['-MNI', '-Tal', '-Talairach']:
        if filename.endswith(suffix):
            base_name = filename[:-len(suffix)]
            return base_name, suffix[1:]  # Remove the leading '-'
    
    return filename, None


def merge_nimads_files(
    files_to_merge: List[Path],
    output_file: Path,
    project_key: str,
    fuzzy_map_dir: Optional[Path] = None,
) -> bool:
    """
    Merge multiple NIMADS JSON files into a single file.
    
    Parameters
    ----------
    files_to_merge : list of Path
        List of NIMADS JSON file paths to merge
    output_file : Path
        Output path for the merged file
        
    Returns
    -------
    bool
        True if merge was successful, False otherwise
    """
    if len(files_to_merge) < 2:
        return False
    
    try:
        # Load all files
        all_studies = []
        merged_id = None
        merged_name = None
        
        for file_path in sorted(files_to_merge):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Set the merged id and name based on the first file
            if merged_id is None:
                # Replace the suffix with -Merged
                base_id = data['id']
                for suffix in ['-MNI', '-Tal', '-Talairach']:
                    if base_id.endswith(suffix):
                        base_id = base_id[:-len(suffix)]
                        break
                merged_id = f"{base_id}-Merged"
                
                base_name = data['name']
                for suffix in ['-MNI', '-Tal', '-Talairach']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                merged_name = f"{base_name}-Merged"
            
            # Collect all studies
            all_studies.extend(data.get('studies', []))
        
        # Create merged data structure
        merged_data = {
            'id': merged_id,
            'name': merged_name,
            'studies': all_studies
        }
        original_study_count = len(merged_data["studies"])
        merged_data = normalize_studies_by_reference(merged_data)
        normalized_study_count = len(merged_data["studies"])
        merged_data, mapping_csv, mapped_total, mapped_count, not_mapped_count, mapping_warnings = apply_optional_pmid_mapping(
            nimads_dict=merged_data,
            project_key=project_key,
            dataset_stem=output_file.stem,
            fuzzy_map_dir=fuzzy_map_dir,
        )
        for warning in mapping_warnings:
            print(f"  ⚠ Mapping warning: {warning}")
        
        # Write merged file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"  ✓ Merged {len(files_to_merge)} files into: {output_file.name}")
        print(f"    - Total studies: {original_study_count} -> {normalized_study_count}")
        if mapping_csv is not None:
            print(
                f"    - PMID mapping from {mapping_csv.name}: {mapped_count}/{mapped_total} "
                f"(unmapped={not_mapped_count})"
            )
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error merging files: {str(e)}")
        return False


def process_merges(
    output_dir: Path,
    project_key: str,
    fuzzy_map_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Process all files in output directory and merge those with matching base names.
    
    After merging, deletes the individual Tal and MNI files, keeping only the Merged copy.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing NIMADS JSON files
        
    Returns
    -------
    dict
        Statistics about merges performed
    """
    stats = {
        'files_checked': 0,
        'groups_found': 0,
        'files_merged': 0,
        'merged_files_created': 0,
        'files_deleted': 0
    }
    
    # Find all JSON files
    json_files = list(output_dir.glob('*.json'))
    stats['files_checked'] = len(json_files)
    
    # Group files by base name
    file_groups = defaultdict(list)
    for json_file in json_files:
        base_name, suffix = get_base_name_and_suffix(json_file.stem)
        if suffix:  # Only group files with coordinate space suffixes
            file_groups[base_name].append((json_file, suffix))
    
    # Process groups that have multiple files
    for base_name, files in file_groups.items():
        if len(files) >= 2:
            stats['groups_found'] += 1
            
            # Sort files by suffix for consistent ordering
            files.sort(key=lambda x: x[1])
            file_paths = [f[0] for f in files]
            
            # Create merged filename
            merged_filename = f"{base_name}-Merged.json"
            merged_path = output_dir / merged_filename
            
            print(f"\n  Merging files with base name '{base_name}':")
            for file_path, suffix in files:
                print(f"    - {file_path.name} ({suffix})")
            
            # Perform merge
            if merge_nimads_files(
                file_paths,
                merged_path,
                project_key=project_key,
                fuzzy_map_dir=fuzzy_map_dir,
            ):
                stats['files_merged'] += len(files)
                stats['merged_files_created'] += 1
                
                # Delete the individual files after successful merge
                print(f"  Removing individual files (keeping only Merged copy):")
                for file_path, suffix in files:
                    try:
                        file_path.unlink()
                        stats['files_deleted'] += 1
                        print(f"    ✓ Deleted: {file_path.name}")
                    except Exception as e:
                        print(f"    ✗ Error deleting {file_path.name}: {str(e)}")
    
    return stats


def main():
    """Main function to convert all Sleuth files to NIMADS format."""
    
    parser = argparse.ArgumentParser(
        description='Convert Sleuth format files to NIMADS format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all projects
  python convert_sleuth_to_nimads.py
  
  # Process only the social project
  python convert_sleuth_to_nimads.py --project social
        """
    )
    
    parser.add_argument(
        '--project',
        type=str,
        help='Specific project to process (e.g., "social"). Matches against normalized folder names in data/nimads/. If not specified, processes all projects.'
    )
    parser.add_argument(
        '--fuzzy-map-dir',
        type=str,
        default='raw/fuzzy_match_study_to_pmcid',
        help='Optional directory containing matched_studies_*.csv files. If available, study IDs are replaced with mapped PMIDs.'
    )
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    meta_datasets_file = base_dir / 'data' / 'meta_datasets.csv'
    sources_dir = base_dir / 'raw' / 'meta-datasets'
    output_base_dir = base_dir / 'data' / 'nimads'
    fuzzy_map_dir = base_dir / args.fuzzy_map_dir
    
    # Validate paths
    if not meta_datasets_file.exists():
        print(f"Error: meta_datasets file not found: {meta_datasets_file}")
        return 1
    
    if not sources_dir.exists():
        print(f"Error: sources directory not found: {sources_dir}")
        return 1

    if fuzzy_map_dir.exists():
        print(f"PMID mapping enabled from: {fuzzy_map_dir}")
    else:
        print(f"PMID mapping disabled (directory not found): {fuzzy_map_dir}")
    
    # Read meta_datasets.csv
    print(f"Reading meta_datasets.csv...")
    df = pd.read_csv(meta_datasets_file)
    
    # Filter by project if specified
    if args.project:
        # Match by normalized folder name
        project_normalized = normalize_folder_name(args.project)
        df = df[df['topic'].apply(normalize_folder_name) == project_normalized]
        
        if df.empty:
            print(f"Error: Project '{args.project}' not found in meta_datasets.csv")
            all_projects = pd.read_csv(meta_datasets_file)['topic'].apply(normalize_folder_name).unique()
            print(f"Available projects: {', '.join(sorted(all_projects))}")
            return 1
        
        print(f"Filtering to project: {args.project} (normalized: {project_normalized})")
    
    # Get available folders
    available_folders = [
        d.name for d in sources_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    print(f"Found {len(available_folders)} folders in raw/meta-datasets/")
    print(f"Processing {len(df)} entries from meta_datasets.tsv\n")
    
    # Overall statistics
    total_stats = {
        'processed': 0,
        'matched': 0,
        'not_matched': 0,
        'files_converted': 0,
        'errors': []
    }
    
    # Process each row
    for idx, row in df.iterrows():
        pmid = str(row['pmid'])
        topic = str(row['topic'])
        
        print(f"[{idx + 1}/{len(df)}] Processing PMID {pmid}: {topic}")
        
        # Find matching folder
        matched_folder = fuzzy_match_folder(topic, available_folders)
        
        if matched_folder:
            print(f"  Matched to folder: {matched_folder}")
            total_stats['matched'] += 1
            
            # Convert files
            sleuth_dir = sources_dir / matched_folder
            # Use normalized topic name for output directory
            normalized_topic = normalize_folder_name(topic)
            output_dir = output_base_dir / normalized_topic
            
            stats = convert_sleuth_files(
                sleuth_dir=sleuth_dir,
                output_dir=output_dir,
                pmid=pmid,
                topic=topic,
                project_key=normalized_topic,
                fuzzy_map_dir=fuzzy_map_dir if fuzzy_map_dir.exists() else None,
            )
            
            total_stats['files_converted'] += stats['files_converted']
            total_stats['errors'].extend(stats['errors'])
            
            converted = stats['files_converted']
            found = stats['files_found']
            print(f"  Converted {converted}/{found} files")
        else:
            print(f"  ✗ No matching folder found for topic: {topic}")
            total_stats['not_matched'] += 1
        
        total_stats['processed'] += 1
        print()
    
    # Print summary
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Total entries processed: {total_stats['processed']}")
    print(f"Successfully matched: {total_stats['matched']}")
    print(f"Not matched: {total_stats['not_matched']}")
    print(f"Total files converted: {total_stats['files_converted']}")
    print(f"Total errors: {len(total_stats['errors'])}")
    
    if total_stats['errors']:
        print("\nErrors encountered:")
        for error in total_stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(total_stats['errors']) > 10:
            print(f"  ... and {len(total_stats['errors']) - 10} more errors")
    
    # Process merges for each output directory
    print("\n" + "=" * 70)
    print("PROCESSING MERGES")
    print("=" * 70)
    
    merge_stats_total = {
        'files_checked': 0,
        'groups_found': 0,
        'files_merged': 0,
        'merged_files_created': 0,
        'files_deleted': 0
    }
    
    # Find all subdirectories in the output base directory
    output_dirs = [d for d in output_base_dir.iterdir() if d.is_dir()]
    
    for output_dir in output_dirs:
        print(f"\nProcessing directory: {output_dir.name}")
        merge_stats = process_merges(
            output_dir=output_dir,
            project_key=output_dir.name,
            fuzzy_map_dir=fuzzy_map_dir if fuzzy_map_dir.exists() else None,
        )
        
        # Accumulate statistics
        for key in merge_stats_total:
            merge_stats_total[key] += merge_stats[key]
        
        if merge_stats['groups_found'] > 0:
            print(f"  Found {merge_stats['groups_found']} group(s) to merge")
            print(f"  Created {merge_stats['merged_files_created']} merged file(s)")
    
    print("\n" + "=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)
    print(f"Total files checked: {merge_stats_total['files_checked']}")
    print(f"Groups merged: {merge_stats_total['groups_found']}")
    print(f"Individual files merged: {merge_stats_total['files_merged']}")
    print(f"Merged files created: {merge_stats_total['merged_files_created']}")
    print(f"Individual files deleted: {merge_stats_total['files_deleted']}")
    
    print(f"\n✓ Conversion and merge complete!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
