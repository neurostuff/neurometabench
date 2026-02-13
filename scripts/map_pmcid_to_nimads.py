#!/usr/bin/env python
"""
Map the human-verified PMCID to NIMADS ID mappings (from the fuzzy matching process) back to the NiMADS files for each meta-analysis.
"""

import json
import os
import argparse
import sys
from pathlib import Path
from collections import Counter
import csv
import pandas as pd


def validate_csv(csv_path):
    """
    Validate fuzzy matching CSV file.
    
    Checks for:
    - Duplicate nimads_study_id values
    - Duplicate matched_pmid values (excluding empty values)
    - Invalid match_status values (should be 'exact_match' or 'manual_override')
    
    Returns:
        tuple: (is_valid, errors, warnings, mappings)
    """
    errors = []
    warnings = []
    mappings = {}
    
    nimads_ids = []
    matched_pmids = []
    
    try:
        # First pass: count rows per nimads_id
        nimads_id_counts = Counter()
        all_rows = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            required_cols = {'nimads_study_id', 'match_status', 'matched_pmid'}
            if not required_cols.issubset(reader.fieldnames):
                errors.append(f"Missing required columns. Expected: {required_cols}, Found: {set(reader.fieldnames)}")
                return False, errors, warnings, mappings
            
            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                nimads_id = row['nimads_study_id']
                nimads_id_counts[nimads_id] += 1
                all_rows.append((row_num, row))
        
        # Second pass: process rows with conditional logic
        for row_num, row in all_rows:
            nimads_id = row['nimads_study_id']
            match_status = row['match_status']
            matched_pmid = row['matched_pmid']
            
            # Check match_status
            if match_status not in ['exact_match', 'manual_override', 'fuzzy_match', 'multiple_matches', 'no_match']:
                errors.append(f"Row {row_num}: Invalid match_status '{match_status}'. Expected 'exact_match' or 'manual_override'")
            
            # Always skip no_match
            if match_status == 'no_match':
                warnings.append(f"Row {row_num}: Skipping '{nimads_id}' with match_status '{match_status}'")
            # Only skip fuzzy_match and multiple_matches if there's more than 1 row for this nimads_id
            elif match_status in ['fuzzy_match', 'multiple_matches'] and nimads_id_counts[nimads_id] > 1:
                warnings.append(f"Row {row_num}: Skipping '{nimads_id}' with match_status '{match_status}' (multiple rows for this ID)")
            # Process exact_match, manual_override, and single-row fuzzy_match/multiple_matches
            elif match_status in ['exact_match', 'manual_override', 'fuzzy_match', 'multiple_matches']:
                if not matched_pmid:
                    errors.append(f"Row {row_num}: match_status is '{match_status}' but matched_pmid is empty for '{nimads_id}'")
                else:
                    nimads_ids.append(nimads_id)
                    matched_pmids.append(matched_pmid)
                    mappings[nimads_id] = matched_pmid
        
        # Check for duplicate nimads_study_id
        nimads_counts = Counter(nimads_ids)
        duplicates = [id for id, count in nimads_counts.items() if count > 1]
        if duplicates:
            errors.append(f"Duplicate nimads_study_id values found: {duplicates}")
        
        # Check for duplicate matched_pmid
        pmid_counts = Counter(matched_pmids)
        pmid_duplicates = [pmid for pmid, count in pmid_counts.items() if count > 1]
        if pmid_duplicates:
            warnings.append(f"Duplicate matched_pmid values found: {pmid_duplicates} (may indicate multiple NiMADS entries for same paper)")
    
    except Exception as e:
        errors.append(f"Error reading CSV: {str(e)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings, mappings


def validate_nimads_coverage(nimads_path, mappings):
    """
    Validate that every unique study_id in the NiMADS file has an entry in the mappings.
    
    Args:
        nimads_path: Path to NiMADS JSON file
        mappings: Dict mapping nimads_study_id to matched_pmid
    
    Returns:
        tuple: (is_valid, errors, nimads_study_ids)
    """
    errors = []
    nimads_study_ids = set()
    
    try:
        with open(nimads_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'studies' in data:
            for study in data['studies']:
                study_id = study.get('id')
                if study_id:
                    nimads_study_ids.add(study_id)
        
        # Check if all NiMADS study IDs are in the mapping
        missing_ids = nimads_study_ids - set(mappings.keys())
        
        if missing_ids:
            errors.append(f"Found {len(missing_ids)} study IDs in NiMADS file not present in fuzzy matching CSV:")
            for missing_id in sorted(missing_ids):
                errors.append(f"  - {missing_id}")
    
    except Exception as e:
        errors.append(f"Error reading NiMADS file: {str(e)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, nimads_study_ids


def load_project_pmids(project_name, meta_datasets_path, included_studies_path):
    """
    Load the expected PMIDs for a project from the metadata files.
    
    Args:
        project_name: Name of the project (e.g., 'social')
        meta_datasets_path: Path to meta_datasets.csv
        included_studies_path: Path to included_studies_wt.csv
    
    Returns:
        tuple: (expected_pmids set, meta_pmid, errors)
    """
    errors = []
    expected_pmids = set()
    meta_pmid = None
    
    try:
        # Load meta_datasets.csv to find the meta_pmid for this project
        meta_df = pd.read_csv(meta_datasets_path)
        
        # Match by topic (case-insensitive)
        project_rows = meta_df[meta_df['topic'].str.lower() == project_name.lower()]
        
        if project_rows.empty:
            errors.append(f"Project '{project_name}' not found in meta_datasets.csv")
            return expected_pmids, meta_pmid, errors
        
        if len(project_rows) > 1:
            errors.append(f"Multiple entries found for project '{project_name}' in meta_datasets.csv")
            return expected_pmids, meta_pmid, errors
        
        meta_pmid = str(project_rows.iloc[0]['pmid'])
        
        # Load included_studies_wt.csv and filter by meta_pmid
        included_df = pd.read_csv(included_studies_path)
        
        # Filter studies for this project
        project_studies = included_df[included_df['meta_pmid'].astype(str) == meta_pmid]
        
        # Extract unique study PMIDs
        expected_pmids = set(project_studies['study_pmid'].astype(str).unique())
        
    except Exception as e:
        errors.append(f"Error loading project PMIDs: {str(e)}")
    
    return expected_pmids, meta_pmid, errors


def validate_pmid_coverage(project_name, fuzzy_dir, expected_pmids):
    """
    Validate that all expected PMIDs appear in at least one fuzzy matching file for the project.
    
    Args:
        project_name: Name of the project (e.g., 'social')
        fuzzy_dir: Directory containing fuzzy match CSV files
        expected_pmids: Set of expected PMIDs from included_studies
    
    Returns:
        tuple: (is_valid, errors, all_matched_pmids)
    """
    errors = []
    all_matched_pmids = set()
    
    try:
        project_fuzzy_dir = Path(fuzzy_dir) / project_name
        
        if not project_fuzzy_dir.exists():
            errors.append(f"Fuzzy match directory not found: {project_fuzzy_dir}")
            return False, errors, all_matched_pmids
        
        # Find all CSV files in the fuzzy match directory
        csv_files = list(project_fuzzy_dir.glob('matched_studies_*.csv'))
        
        if not csv_files:
            errors.append(f"No matched_studies_*.csv files found in {project_fuzzy_dir}")
            return False, errors, all_matched_pmids
        
        # Collect all matched PMIDs across all fuzzy matching files
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        matched_pmid = row.get('matched_pmid', '').strip()
                        if matched_pmid:
                            all_matched_pmids.add(matched_pmid)
            except Exception as e:
                errors.append(f"Error reading {csv_file.name}: {str(e)}")
        
        # Check for missing PMIDs
        missing_pmids = expected_pmids - all_matched_pmids
        
        if missing_pmids:
            errors.append(f"Found {len(missing_pmids)} PMIDs from included_studies not present in any fuzzy matching file:")
            # Show first 10 missing PMIDs
            for pmid in sorted(missing_pmids)[:10]:
                errors.append(f"  - {pmid}")
            if len(missing_pmids) > 10:
                errors.append(f"  ... and {len(missing_pmids) - 10} more")
    
    except Exception as e:
        errors.append(f"Error validating PMID coverage: {str(e)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, all_matched_pmids


def map_pmids_to_nimads(nimads_path, mappings, output_path=None):
    """
    Map PMIDs to NiMADS file by replacing study IDs.
    
    Args:
        nimads_path: Path to input NiMADS JSON file
        mappings: Dict mapping nimads_study_id to matched_pmid
        output_path: Path for output file (if None, overwrites input)
    
    Returns:
        tuple: (success, num_mapped, num_not_found)
    """
    try:
        with open(nimads_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_mapped = 0
        num_not_found = 0
        
        if 'studies' in data:
            for study in data['studies']:
                original_id = study['id']
                if original_id in mappings:
                    study['id'] = mappings[original_id]
                    num_mapped += 1
                else:
                    num_not_found += 1
        
        # Write output
        if output_path is None:
            output_path = nimads_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True, num_mapped, num_not_found
    
    except Exception as e:
        print(f"Error processing {nimads_path}: {str(e)}", file=sys.stderr)
        return False, 0, 0


def process_project(project_name, fuzzy_dir, nimads_dir, dry_run=False,
                   meta_datasets_path=None, included_studies_path=None,
                   validate_pmid_coverage_flag=False):
    """
    Process all matching files for a given project.
    
    Args:
        project_name: Name of the project (e.g., 'social')
        fuzzy_dir: Directory containing fuzzy match CSV files
        nimads_dir: Directory containing NiMADS JSON files
        dry_run: If True, don't write changes
        meta_datasets_path: Path to meta_datasets.csv (optional)
        included_studies_path: Path to included_studies_wt.csv (optional)
        validate_pmid_coverage_flag: If True, validate PMID coverage
    """
    project_fuzzy_dir = Path(fuzzy_dir) / project_name
    project_nimads_dir = Path(nimads_dir) / project_name
    
    if not project_fuzzy_dir.exists():
        print(f"Warning: Fuzzy match directory not found: {project_fuzzy_dir}")
        return
    
    if not project_nimads_dir.exists():
        print(f"Warning: NiMADS directory not found: {project_nimads_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing project: {project_name}")
    print(f"{'='*60}")
    
    # Optional: Validate PMID coverage across all fuzzy matching files
    if validate_pmid_coverage_flag and meta_datasets_path and included_studies_path:
        print(f"\nValidating PMID coverage for project '{project_name}'...")
        expected_pmids, meta_pmid, load_errors = load_project_pmids(
            project_name, meta_datasets_path, included_studies_path
        )
        
        if load_errors:
            print(f"  WARNING: Could not load expected PMIDs:")
            for error in load_errors:
                print(f"    {error}")
        else:
            print(f"  Loaded {len(expected_pmids)} expected PMIDs for meta_pmid={meta_pmid}")
            
            coverage_valid, coverage_errors, all_matched_pmids = validate_pmid_coverage(
                project_name, fuzzy_dir, expected_pmids
            )
            
            if not coverage_valid:
                print(f"  WARNING: PMID coverage validation failed:")
                for error in coverage_errors:
                    print(f"    {error}")
                print(f"  Found {len(all_matched_pmids)} PMIDs in fuzzy matching files")
                print(f"  Expected {len(expected_pmids)} PMIDs from included_studies")
            else:
                print(f"  ✓ PMID Coverage Validation: PASSED")
                print(f"  All {len(expected_pmids)} expected PMIDs found in fuzzy matching files")
    
    # Find all CSV files in the fuzzy match directory
    csv_files = list(project_fuzzy_dir.glob('matched_studies_*.csv'))
    
    if not csv_files:
        print(f"No matched_studies_*.csv files found in {project_fuzzy_dir}")
        return
    
    total_processed = 0
    total_mapped = 0
    total_not_found = 0
    
    for csv_file in sorted(csv_files):
        # Extract the dataset name from filename
        # e.g., matched_studies_ALL.csv -> ALL
        dataset_name = csv_file.stem.replace('matched_studies_', '')
        
        # Find corresponding NiMADS file
        nimads_file = project_nimads_dir / f"{dataset_name}-Merged.json"
        
        if not nimads_file.exists():
            # Try without -Merged suffix
            nimads_file = project_nimads_dir / f"{dataset_name}.json"
            if not nimads_file.exists():
                print(f"\nWarning: NiMADS file not found for {csv_file.name}")
                print(f"  Expected: {dataset_name}-Merged.json or {dataset_name}.json")
                continue
        
        print(f"\n{'-'*60}")
        print(f"Processing: {csv_file.name} -> {nimads_file.name}")
        print(f"{'-'*60}")
        
        # Validate CSV
        is_valid, errors, warnings, mappings = validate_csv(csv_file)
        
        # Print warnings
        for warning in warnings:
            print(f"  WARNING: {warning}")
        
        # Print errors
        if errors:
            print(f"  ERROR: Validation failed for {csv_file.name}:")
            for error in errors:
                print(f"    - {error}")
            print(f"  Skipping {nimads_file.name}")
            continue
        
        print(f"  CSV Validation: PASSED")
        print(f"  Found {len(mappings)} valid mappings")
        
        # Validate that all NiMADS study IDs are covered in the mapping
        coverage_valid, coverage_errors, nimads_study_ids = validate_nimads_coverage(
            nimads_file, mappings
        )
        
        if not coverage_valid:
            print(f"  ERROR: Coverage validation failed for {nimads_file.name}:")
            for error in coverage_errors:
                print(f"    {error}")
            print(f"  Skipping {nimads_file.name}")
            continue
        
        print(f"  Coverage Validation: PASSED")
        print(f"  All {len(nimads_study_ids)} NiMADS study IDs found in mapping")
        
        if dry_run:
            print(f"  DRY RUN: Would update {nimads_file}")
            total_processed += 1
            total_mapped += len(mappings)
        else:
            # Apply mappings
            success, num_mapped, num_not_found = map_pmids_to_nimads(
                nimads_file, mappings
            )
            
            if success:
                print(f"  ✓ Successfully mapped {num_mapped} studies")
                if num_not_found > 0:
                    print(f"  ⚠ {num_not_found} studies not found in mapping file")
                total_processed += 1
                total_mapped += num_mapped
                total_not_found += num_not_found
            else:
                print(f"  ✗ Failed to process {nimads_file}")
    
    print(f"\n{'='*60}")
    print(f"Project {project_name} Summary:")
    print(f"  Files processed: {total_processed}")
    print(f"  Studies mapped: {total_mapped}")
    print(f"  Studies not in mapping: {total_not_found}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Map PMIDs from fuzzy matching results to NiMADS files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all projects
  python map_pmcid_to_nimads.py
  
  # Process only the social project
  python map_pmcid_to_nimads.py --project social
  
  # Dry run (validate only, don't write changes)
  python map_pmcid_to_nimads.py --dry-run
  
  # Validate PMID coverage
  python map_pmcid_to_nimads.py --project social --validate-pmid-coverage
        """
    )
    
    parser.add_argument(
        '--project',
        type=str,
        help='Specific project to process (e.g., "social"). If not specified, processes all projects.'
    )
    
    parser.add_argument(
        '--fuzzy-dir',
        type=str,
        default='raw/fuzzy_match_study_to_pmcid',
        help='Directory containing fuzzy match results (default: raw/fuzzy_match_study_to_pmcid)'
    )
    
    parser.add_argument(
        '--nimads-dir',
        type=str,
        default='data/nimads',
        help='Directory containing NiMADS files (default: data/nimads)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate only, do not write changes'
    )
    
    parser.add_argument(
        '--validate-pmid-coverage',
        action='store_true',
        help='Validate that all PMIDs from included_studies appear in fuzzy matching files'
    )
    
    parser.add_argument(
        '--included-studies',
        type=str,
        default='data/included_studies.csv',
        help='Path to included studies CSV file (default: data/included_studies.csv)'
    )
    
    parser.add_argument(
        '--meta-datasets',
        type=str,
        default='data/meta_datasets.csv',
        help='Path to meta_datasets.csv (default: data/meta_datasets.csv)'
    )
    
    args = parser.parse_args()
    
    fuzzy_dir = Path(args.fuzzy_dir)
    nimads_dir = Path(args.nimads_dir)
    
    if not fuzzy_dir.exists():
        print(f"Error: Fuzzy match directory not found: {fuzzy_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not nimads_dir.exists():
        print(f"Error: NiMADS directory not found: {nimads_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Check if validation files exist when PMID coverage validation is requested
    meta_datasets_path = None
    included_studies_path = None
    if args.validate_pmid_coverage:
        meta_datasets_path = Path(args.meta_datasets)
        included_studies_path = Path(args.included_studies)
        
        if not meta_datasets_path.exists():
            print(f"Error: meta_datasets file not found: {meta_datasets_path}", file=sys.stderr)
            print(f"Cannot perform PMID coverage validation without this file.", file=sys.stderr)
            sys.exit(1)
        
        if not included_studies_path.exists():
            print(f"Error: included_studies file not found: {included_studies_path}", file=sys.stderr)
            print(f"Cannot perform PMID coverage validation without this file.", file=sys.stderr)
            sys.exit(1)
    
    if args.dry_run:
        print("="*60)
        print("DRY RUN MODE - No changes will be written")
        print("="*60)
    
    if args.validate_pmid_coverage:
        print("="*60)
        print("PMID COVERAGE VALIDATION ENABLED")
        print("="*60)
    
    # Determine which projects to process
    if args.project:
        projects = [args.project]
    else:
        # Find all project directories in fuzzy_dir
        projects = [d.name for d in fuzzy_dir.iterdir() if d.is_dir()]
    
    if not projects:
        print("No projects found to process")
        sys.exit(1)
    
    print(f"Projects to process: {', '.join(projects)}\n")
    
    # Process each project
    for project in projects:
        process_project(
            project,
            fuzzy_dir,
            nimads_dir,
            dry_run=args.dry_run,
            meta_datasets_path=str(meta_datasets_path) if meta_datasets_path else None,
            included_studies_path=str(included_studies_path) if included_studies_path else None,
            validate_pmid_coverage_flag=args.validate_pmid_coverage
        )
    
    print("\n" + "="*60)
    print("All processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
