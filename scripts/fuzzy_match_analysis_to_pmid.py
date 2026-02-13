#!/usr/bin/env python
"""
Fuzzy Match NIMADS Analysis IDs to PMIDs

This script matches NIMADS study names (format: "Author et al., Year") to PMIDs
from an included studies CSV file using fuzzy matching on author names and exact
matching on years.

Usage:
    python scripts/fuzzy_match_analysis_to_pmid.py \
        --project social \
        --included-studies data/included_studies.csv \
        --output-dir raw/fuzzy_match_study_to_pmcid \
        --threshold 0.85 \
        --verbose
"""

import json
import csv
import re
import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import unicodedata

import pandas as pd
from rapidfuzz import fuzz


def parse_study_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract author and year from NIMADS study name.
    
    Args:
        name: Study name (e.g., "Akitsuki et al., 2009")
    
    Returns:
        tuple: (author, year) or (None, None) if parsing fails
    
    Examples:
        >>> parse_study_name("Akitsuki et al., 2009")
        ('Akitsuki', '2009')
        >>> parse_study_name("Alba-Ferrara et al., 2011")
        ('Alba-Ferrara', '2011')
    """
    if not name or not isinstance(name, str):
        return None, None
    
    # Pattern: "Author et al., Year" or "Author, Year"
    # Handle hyphenated names, unicode characters
    pattern = r'^([^\s,]+(?:-[^\s,]+)*)\s+(?:et\s+al\.,?\s*,?\s*|,\s*)(\d{4})'
    
    match = re.match(pattern, name.strip())
    if match:
        author = match.group(1).strip()
        year = match.group(2).strip()
        return author, year
    
    # Fallback: try to extract year from end and author from beginning
    year_match = re.search(r'(\d{4})\s*$', name)
    if year_match:
        year = year_match.group(1)
        # Extract everything before "et al." or the year
        author_match = re.match(r'^([^\s,]+(?:-[^\s,]+)*)', name)
        if author_match:
            author = author_match.group(1).strip()
            return author, year
    
    return None, None


def normalize_string(s: str) -> str:
    """
    Normalize string for comparison by removing accents and converting to lowercase.
    
    Args:
        s: Input string
    
    Returns:
        Normalized string
    
    Examples:
        >>> normalize_string("Bergström")
        'bergstrom'
        >>> normalize_string("Alba-Ferrara")
        'alba-ferrara'
    """
    # Decompose unicode characters and remove accents
    nfd = unicodedata.normalize('NFD', s)
    without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return without_accents.lower()


def load_meta_datasets(filepath: str) -> pd.DataFrame:
    """
    Load meta_datasets.csv file.
    
    Args:
        filepath: Path to meta_datasets.csv
    
    Returns:
        DataFrame with meta-dataset information
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Meta datasets file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Check required columns
    required_cols = ['pmid', 'topic']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def get_meta_pmid_for_project(meta_datasets_df: pd.DataFrame, project_name: str) -> Optional[int]:
    """
    Look up meta-analysis PMID for a given project name.
    
    Args:
        meta_datasets_df: DataFrame with meta-dataset information
        project_name: Name of the project (e.g., "social")
    
    Returns:
        Meta-analysis PMID or None if not found
    """
    # Normalize project name for comparison
    project_normalized = project_name.lower().replace('_', ' ').strip()
    
    # Try exact match first
    for _, row in meta_datasets_df.iterrows():
        topic = str(row['topic']).lower().strip()
        if topic == project_normalized:
            return int(row['pmid'])
    
    # Try partial match
    for _, row in meta_datasets_df.iterrows():
        topic = str(row['topic']).lower().strip()
        if project_normalized in topic or topic in project_normalized:
            return int(row['pmid'])
    
    return None


def load_included_studies(filepath: str, meta_pmid: int) -> pd.DataFrame:
    """
    Load CSV and filter by meta_pmid.
    
    Args:
        filepath: Path to included_studies CSV
        meta_pmid: Meta-analysis PMID to filter by
    
    Returns:
        DataFrame with filtered studies
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Included studies file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Check required columns
    required_cols = ['meta_pmid', 'study_pmid', 'author', 'year', 'title']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter by meta_pmid
    filtered = df[df['meta_pmid'] == meta_pmid].copy()
    
    # Ensure year is string for matching (convert to int first to remove decimals)
    filtered['year'] = filtered['year'].astype(int).astype(str)
    
    # Add normalized author column for matching
    filtered['author_normalized'] = filtered['author'].apply(normalize_string)
    
    return filtered


def load_nimads_studies(filepath: str) -> List[Dict[str, str]]:
    """
    Extract study names from NIMADS JSON.
    
    Args:
        filepath: Path to NIMADS JSON file
    
    Returns:
        List of study dicts with id and name
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON structure is invalid
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"NIMADS file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'studies' not in data:
        raise ValueError("Invalid NIMADS file: missing 'studies' field")
    
    studies = []
    for study in data['studies']:
        if 'id' in study:
            studies.append({
                'id': study['id'],
                'name': study.get('name', study['id'])
            })
    
    return studies


def find_nimads_files(project_name: str, nimads_dir: str = 'data/nimads') -> List[Path]:
    """
    Find all NIMADS JSON files for a given project.
    
    Args:
        project_name: Name of the project (e.g., "social")
        nimads_dir: Base directory for NIMADS files
    
    Returns:
        List of Path objects for NIMADS JSON files
    """
    project_dir = Path(nimads_dir) / project_name
    
    if not project_dir.exists():
        return []
    
    # Find all .json files in the project directory
    json_files = list(project_dir.glob('*.json'))
    
    return sorted(json_files)


def fuzzy_match_study(
    nimads_author: str,
    nimads_year: str,
    included_studies: pd.DataFrame,
    threshold: float = 0.85
) -> List[Dict]:
    """
    Find matching studies using a three-phase approach:
    
    Phase 1: Look for exact matches with exact year
    Phase 2: If no exact matches found, use fuzzy matching with exact year
    Phase 3: If still no matches, try ±1 year with exact or fuzzy matching
    
    Args:
        nimads_author: Author name from NIMADS
        nimads_year: Year from NIMADS
        included_studies: DataFrame of potential matches
        threshold: Minimum similarity score (0.0-1.0)
    
    Returns:
        List of match dicts with pmid, author, year, title, score
        Sorted by score descending
    """
    # Normalize the NIMADS author for comparison
    nimads_author_norm = normalize_string(nimads_author)
    
    # Helper function to find matches for a specific year
    def find_matches_for_year(year_str: str, year_penalty: float = 0.0):
        year_matches = included_studies[included_studies['year'] == year_str]
        if year_matches.empty:
            return [], []
        
        exact_matches = []
        fuzzy_matches = []
        
        for _, row in year_matches.iterrows():
            csv_author = row['author']
            csv_author_norm = row['author_normalized']
            
            # Check for exact match
            if nimads_author == csv_author:
                exact_matches.append({
                    'pmid': row['study_pmid'],
                    'author': csv_author,
                    'year': row['year'],
                    'title': row['title'],
                    'score': max(0.0, 1.0 - year_penalty)
                })
            elif nimads_author_norm == csv_author_norm:
                exact_matches.append({
                    'pmid': row['study_pmid'],
                    'author': csv_author,
                    'year': row['year'],
                    'title': row['title'],
                    'score': max(0.0, 0.95 - year_penalty)
                })
            else:
                # Calculate fuzzy matching scores
                scores = []
                
                ratio = fuzz.ratio(nimads_author_norm, csv_author_norm) / 100.0
                scores.append(ratio)
                
                partial_ratio = fuzz.partial_ratio(nimads_author_norm, csv_author_norm) / 100.0
                scores.append(partial_ratio)
                
                # Token-based matching (for hyphenated names)
                token_sort = fuzz.token_sort_ratio(nimads_author_norm, csv_author_norm) / 100.0
                scores.append(token_sort)
                
                # Take the best score and apply year penalty
                best_score = max(0.0, max(scores) - year_penalty)
                
                # Only include if above threshold
                if best_score >= threshold:
                    fuzzy_matches.append({
                        'pmid': row['study_pmid'],
                        'author': csv_author,
                        'year': row['year'],
                        'title': row['title'],
                        'score': best_score
                    })
        
        return exact_matches, fuzzy_matches
    
    # PHASE 1 & 2: Try exact year first
    exact_matches, fuzzy_matches = find_matches_for_year(nimads_year)
    
    # If we found exact matches, return only those
    if exact_matches:
        exact_matches.sort(key=lambda x: x['score'], reverse=True)
        return exact_matches
    
    # If we found fuzzy matches, return those
    if fuzzy_matches:
        fuzzy_matches.sort(key=lambda x: x['score'], reverse=True)
        return fuzzy_matches
    
    # PHASE 3: No matches with exact year, try ±1 year
    # Apply a penalty of 0.05 for year mismatch
    year_penalty = 0.05
    
    try:
        nimads_year_int = int(nimads_year)
        year_plus_one = str(nimads_year_int + 1)
        year_minus_one = str(nimads_year_int - 1)
        
        all_matches = []
        
        # Try year + 1
        exact_plus, fuzzy_plus = find_matches_for_year(year_plus_one, year_penalty)
        all_matches.extend(exact_plus)
        all_matches.extend(fuzzy_plus)
        
        # Try year - 1
        exact_minus, fuzzy_minus = find_matches_for_year(year_minus_one, year_penalty)
        all_matches.extend(exact_minus)
        all_matches.extend(fuzzy_minus)
        
        # Sort all matches by score descending
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        return all_matches
        
    except ValueError:
        # If year is not a valid integer, return empty
        return []


def determine_match_status(match_count: int, best_score: float) -> str:
    """
    Determine the match status based on count and score.
    
    Args:
        match_count: Number of matches found
        best_score: Best match score
    
    Returns:
        Match status string
    """
    if match_count == 0:
        return "no_match"
    elif match_count == 1:
        if best_score == 1.0:
            return "exact_match"
        else:
            return "fuzzy_match"
    else:
        return "multiple_matches"


def process_matches(nimads_studies: List[Dict], included_studies: pd.DataFrame, 
                    threshold: float, verbose: bool = False) -> List[Dict]:
    """
    Process all NIMADS studies and find matches.
    
    Args:
        nimads_studies: List of NIMADS study dicts
        included_studies: DataFrame of included studies
        threshold: Matching threshold
        verbose: Print progress information
    
    Returns:
        List of all match results
    """
    all_results = []
    
    total = len(nimads_studies)
    for idx, study in enumerate(nimads_studies, 1):
        study_id = study['id']
        study_name = study['name']
        
        if verbose:
            print(f"Processing {idx}/{total}: {study_id}", file=sys.stderr)
        
        # Parse the study name
        author, year = parse_study_name(study_id)
        
        if not author or not year:
            if verbose:
                print(f"  WARNING: Could not parse study name: {study_id}", file=sys.stderr)
            # Add a no-match result
            all_results.append({
                'nimads_study_id': study_id,
                'nimads_name': study_name,
                'match_status': 'no_match',
                'matched_pmid': '',
                'matched_author': '',
                'matched_year': '',
                'matched_title': '',
                'match_score': 0.0,
                'match_count': 0
            })
            continue
        
        # Find matches
        matches = fuzzy_match_study(author, year, included_studies, threshold)
        match_count = len(matches)
        
        if verbose:
            if match_count == 0:
                print(f"  No matches found", file=sys.stderr)
            elif match_count == 1:
                print(f"  Match: {matches[0]['author']} (score: {matches[0]['score']:.2f})", file=sys.stderr)
            else:
                print(f"  {match_count} matches found:", file=sys.stderr)
                for m in matches:
                    print(f"    - {m['author']} (score: {m['score']:.2f})", file=sys.stderr)
        
        # Determine match status
        best_score = matches[0]['score'] if matches else 0.0
        match_status = determine_match_status(match_count, best_score)
        
        if match_count == 0:
            # Add single no-match row
            all_results.append({
                'nimads_study_id': study_id,
                'nimads_name': study_name,
                'match_status': match_status,
                'matched_pmid': '',
                'matched_author': '',
                'matched_year': '',
                'matched_title': '',
                'match_score': 0.0,
                'match_count': 0
            })
        else:
            # Add one row per match
            for match in matches:
                all_results.append({
                    'nimads_study_id': study_id,
                    'nimads_name': study_name,
                    'match_status': match_status,
                    'matched_pmid': match['pmid'],
                    'matched_author': match['author'],
                    'matched_year': match['year'],
                    'matched_title': match['title'],
                    'match_score': match['score'],
                    'match_count': match_count
                })
    
    return all_results


def write_results(results: List[Dict], output_file: str):
    """
    Write results to CSV file.
    
    Args:
        results: List of match result dicts
        output_file: Output file path
    """
    fieldnames = [
        'nimads_study_id', 'nimads_name', 'match_status',
        'matched_pmid', 'matched_author', 'matched_year',
        'matched_title', 'match_score', 'match_count'
    ]
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def validate_coverage(nimads_studies: List[Dict], results: List[Dict],
                      nimads_file: Path, output_file: Path) -> List[str]:
    """
    Validate that every unique study_id in the NiMADS file has an entry in the results.
    
    Args:
        nimads_studies: List of NIMADS study dicts
        results: List of match result dicts
        nimads_file: Path to the NiMADS file
        output_file: Path to the output CSV file
    
    Returns:
        List of validation error messages (empty if validation passes)
    """
    errors = []
    
    # Get all unique study IDs from NiMADS file
    nimads_study_ids = set(study['id'] for study in nimads_studies)
    
    # Get all study IDs that appear in results
    results_study_ids = set(result['nimads_study_id'] for result in results)
    
    # Find missing study IDs
    missing_study_ids = nimads_study_ids - results_study_ids
    
    if missing_study_ids:
        errors.append(f"VALIDATION ERROR: {len(missing_study_ids)} study ID(s) from NiMADS file "
                     f"'{nimads_file.name}' are missing in the output file '{output_file.name}':")
        for study_id in sorted(missing_study_ids):
            errors.append(f"  - {study_id}")
    
    return errors


def process_project(project_name: str, included_studies_file: str, output_dir: str,
                   meta_datasets_file: str, threshold: float, verbose: bool):
    """
    Process all NIMADS files for a given project.
    
    Args:
        project_name: Name of the project (e.g., "social")
        included_studies_file: Path to included_studies CSV
        output_dir: Output directory for results
        meta_datasets_file: Path to meta_datasets.csv
        threshold: Matching threshold
        verbose: Print progress information
    """
    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Processing project: {project_name}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
    
    # Load meta datasets and find meta-PMID
    if verbose:
        print(f"Loading meta datasets from: {meta_datasets_file}", file=sys.stderr)
    
    meta_datasets_df = load_meta_datasets(meta_datasets_file)
    meta_pmid = get_meta_pmid_for_project(meta_datasets_df, project_name)
    
    if meta_pmid is None:
        print(f"ERROR: Could not find meta-PMID for project '{project_name}'", file=sys.stderr)
        print(f"Available topics:", file=sys.stderr)
        for _, row in meta_datasets_df.iterrows():
            print(f"  - {row['topic']} (PMID: {row['pmid']})", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        print(f"Found meta-PMID: {meta_pmid}\n", file=sys.stderr)
    
    # Load included studies
    if verbose:
        print(f"Loading included studies from: {included_studies_file}", file=sys.stderr)
    
    included_studies = load_included_studies(included_studies_file, meta_pmid)
    
    if verbose:
        print(f"Found {len(included_studies)} studies for meta-PMID {meta_pmid}\n", file=sys.stderr)
    
    # Find all NIMADS files for this project
    nimads_files = find_nimads_files(project_name)
    
    if not nimads_files:
        print(f"ERROR: No NIMADS files found for project '{project_name}'", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        print(f"Found {len(nimads_files)} NIMADS files:", file=sys.stderr)
        for nf in nimads_files:
            print(f"  - {nf.name}", file=sys.stderr)
        print(f"\nMatching with threshold: {threshold}\n", file=sys.stderr)
    
    # Process each NIMADS file
    project_output_dir = Path(output_dir) / project_name
    project_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_studies = 0
    total_exact = 0
    total_fuzzy = 0
    total_multiple = 0
    total_no_match = 0
    
    validation_errors = []
    
    for nimads_file in nimads_files:
        if verbose:
            print(f"\n{'-'*60}", file=sys.stderr)
            print(f"Processing: {nimads_file.name}", file=sys.stderr)
            print(f"{'-'*60}\n", file=sys.stderr)
        
        # Load NIMADS studies
        nimads_studies = load_nimads_studies(nimads_file)
        total_studies += len(nimads_studies)
        
        if verbose:
            print(f"Found {len(nimads_studies)} studies in NIMADS file\n", file=sys.stderr)
        
        # Process matches
        results = process_matches(nimads_studies, included_studies, threshold, verbose)
        
        # Count match types
        exact_matches = sum(1 for r in results if r['match_status'] == 'exact_match' and r['match_count'] > 0)
        fuzzy_matches = sum(1 for r in results if r['match_status'] == 'fuzzy_match')
        multiple_matches = len(set(r['nimads_study_id'] for r in results if r['match_status'] == 'multiple_matches'))
        no_matches = sum(1 for r in results if r['match_status'] == 'no_match')
        
        total_exact += exact_matches
        total_fuzzy += fuzzy_matches
        total_multiple += multiple_matches
        total_no_match += no_matches
        
        # Generate output filename
        base_name = nimads_file.stem  # e.g., "ALL-Merged" from "ALL-Merged.json"
        output_file = project_output_dir / f"matched_studies_{base_name}.csv"
        
        # Write results
        write_results(results, str(output_file))
        
        # Validate coverage
        file_validation_errors = validate_coverage(nimads_studies, results, nimads_file, output_file)
        if file_validation_errors:
            validation_errors.extend(file_validation_errors)
            if verbose:
                print(f"\n⚠️  VALIDATION FAILED ⚠️", file=sys.stderr)
                for error in file_validation_errors:
                    print(error, file=sys.stderr)
        
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"File: {nimads_file.name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"Total studies:         {len(nimads_studies)}", file=sys.stderr)
            print(f"Exact matches:         {exact_matches}", file=sys.stderr)
            print(f"Fuzzy matches:         {fuzzy_matches}", file=sys.stderr)
            print(f"Multiple matches:      {multiple_matches}", file=sys.stderr)
            print(f"No matches:            {no_matches}", file=sys.stderr)
            print(f"Output:                {output_file}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
    
    # Print overall summary
    if verbose:
        print(f"\n\n{'='*60}", file=sys.stderr)
        print(f"PROJECT SUMMARY: {project_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Files processed:       {len(nimads_files)}", file=sys.stderr)
        print(f"Total studies:         {total_studies}", file=sys.stderr)
        print(f"Exact matches:         {total_exact}", file=sys.stderr)
        print(f"Fuzzy matches:         {total_fuzzy}", file=sys.stderr)
        print(f"Multiple matches:      {total_multiple}", file=sys.stderr)
        print(f"No matches:            {total_no_match}", file=sys.stderr)
        print(f"Output directory:      {project_output_dir}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
    
    # Report validation errors if any
    if validation_errors:
        print(f"\n⚠️  VALIDATION ERRORS FOUND ⚠️", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        for error in validation_errors:
            print(error, file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        raise ValueError(f"Validation failed: {len(validation_errors)} error(s) found. "
                        "Every unique study_id in NiMADS files must have a corresponding entry in the fuzzy match output.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Fuzzy match NIMADS study IDs to PMIDs for a project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process social project
  python scripts/fuzzy_match_analysis_to_pmid.py --project social
  
  # Process with custom threshold
  python scripts/fuzzy_match_analysis_to_pmid.py --project social --threshold 0.90
  
  # Verbose output
  python scripts/fuzzy_match_analysis_to_pmid.py --project social --verbose
        """
    )
    
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Name of the project to process (e.g., "social")'
    )
    
    parser.add_argument(
        '--included-studies',
        type=str,
        default='data/included_studies_wt.csv',
        help='Path to included studies CSV file (default: data/included_studies_wt.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='raw/fuzzy_match_study_to_pmcid',
        help='Output directory for results (default: raw/fuzzy_match_study_to_pmcid)'
    )
    
    parser.add_argument(
        '--meta-datasets',
        type=str,
        default='data/meta_datasets.csv',
        help='Path to meta_datasets.csv (default: data/meta_datasets.csv)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Minimum similarity score for matching (0.0-1.0, default: 0.85)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed matching information to stderr'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")
    
    try:
        process_project(
            project_name=args.project,
            included_studies_file=args.included_studies,
            output_dir=args.output_dir,
            meta_datasets_file=args.meta_datasets,
            threshold=args.threshold,
            verbose=args.verbose
        )
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
