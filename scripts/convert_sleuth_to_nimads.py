#!/usr/bin/env python
"""
Convert Sleuth format files to NIMADS format.

This script reads meta_datasets.tsv, finds corresponding Sleuth files in
raw/meta-datasets/, converts them to NIMADS format using nimare, and
saves them to data/nimads/{pmid}/.

Files with the same base name but different coordinate space suffixes 
(-MNI, -Tal, -Talairach) are merged into a single file with -Merged suffix.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
from difflib import SequenceMatcher
import nimare.io

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    topic: str
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
            
            # Save as JSON
            output_file = output_dir / f"{file_stem}.json"
            with open(output_file, 'w') as f:
                json.dump(nimads_dict, f, indent=2)
            
            stats['files_converted'] += 1
            print(f"  ✓ Converted: {txt_file.name} -> {output_file.name}")
            
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


def merge_nimads_files(files_to_merge: List[Path], output_file: Path) -> bool:
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
        
        # Write merged file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"  ✓ Merged {len(files_to_merge)} files into: {output_file.name}")
        print(f"    - Total studies: {len(all_studies)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error merging files: {str(e)}")
        return False


def process_merges(output_dir: Path) -> Dict[str, int]:
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
            if merge_nimads_files(file_paths, merged_path):
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
    
    # Paths
    base_dir = Path(__file__).parent.parent
    meta_datasets_file = base_dir / 'data' / 'meta_datasets.tsv'
    sources_dir = base_dir / 'raw' / 'meta-datasets'
    output_base_dir = base_dir / 'data' / 'nimads'
    
    # Read meta_datasets.tsv
    print("Reading meta_datasets.tsv...")
    df = pd.read_csv(meta_datasets_file, sep='\t')
    
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
                topic=topic
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
        merge_stats = process_merges(output_dir)
        
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
    
    print("\n✓ Conversion and merge complete!")


if __name__ == '__main__':
    main()