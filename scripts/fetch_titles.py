#!/usr/bin/env python3
"""
Script to fetch titles, publication years, and first authors for
study_pmid values and add them as new columns to the CSV file.
"""

import pandas as pd
import requests
import time
from typing import Optional
import xml.etree.ElementTree as ET


def fetch_data_from_pmid(pmid: str) -> (
        tuple[Optional[str], Optional[str], Optional[str]]):
    """
    Fetch the title, publication year, and first author of a study given
    its PMID using the NCBI EFetch API.
    
    Parameters
    ----------
    pmid : str
        PubMed ID of the study
        
    Returns
    -------
    tuple[str or None, str or None, str or None]
        Tuple of (title, publication_year, first_author) of the study if found,
        otherwise (None, None, None)
    """
    if not pmid or pd.isna(pmid):
        return None, None, None
        
    try:
        # NCBI EFetch API URL
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        # Make the request with a timeout and retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 429:  # Too Many Requests
                    print(f"Rate limited for PMID {pmid}. "
                          f"Waiting before retry "
                          f"{attempt + 1}/{max_retries}...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                print(f"Request failed for PMID {pmid}. "
                      f"Retrying {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Parse the XML response
        root = ET.fromstring(response.text)
        
        # Find the title in the XML
        title_element = root.find(".//ArticleTitle")
        title = None
        if title_element is not None:
            title = title_element.text
            
        # Try alternative path
        if title is None:
            title_element = root.find(".//Title")
            if title_element is not None:
                title = title_element.text
                
        # Find the publication year in the XML
        year_element = root.find(".//PubDate/Year")
        year = None
        if year_element is not None:
            year = year_element.text
            
        # Try alternative path for year
        if year is None:
            year_element = root.find(
                ".//PubMedPubDate[@PubStatus='pubmed']/Year")
            if year_element is not None:
                year = year_element.text
                
        # Find the first author in the XML
        first_author = None
        author_element = root.find(".//AuthorList/Author[1]/LastName")
        if author_element is not None:
            first_author = author_element.text
        else:
            # Try alternative path for author
            author_element = root.find(".//Author/LastName")
            if author_element is not None:
                first_author = author_element.text
                
        return title, year, first_author
    except Exception as e:
        print(f"Error fetching data for PMID {pmid}: {e}")
        return None, None, None


def add_titles_years_and_authors_to_csv(input_file: str, output_file: str,
                                        sample_size: Optional[int] = None):
    """
    Add titles, publication years, and first authors to the CSV file based on
    study_pmid values.
    
    Parameters
    ----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to the output CSV file
    sample_size : int or None
        Number of rows to process (for testing), None for all rows
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # If sample_size is specified, use only that many rows
    if sample_size is not None:
        df = df.head(sample_size)
    
    # Check if 'title' column already exists
    if 'title' in df.columns:
        print("Title column already exists. Only filling missing titles.")
        # Only process rows where title is missing but study_pmid exists
        mask = df['title'].isna() & df['study_pmid'].notna()
        rows_to_process = df[mask]
    else:
        print("Title column does not exist. Creating and filling titles.")
        # Initialize the title column
        df['title'] = None
        # Process all rows with valid study_pmid
        mask = df['study_pmid'].notna()
        rows_to_process = df[mask]
    
    # Process each row that needs a title
    total_rows = len(rows_to_process)
    processed_count = 0
    
    for index, row in rows_to_process.iterrows():
        pmid = row['study_pmid']
        processed_count += 1
        
        # Skip if pmid is missing (shouldn't happen due to mask,
        # but just in case)
        if pd.isna(pmid):
            print(f"Skipping row {index + 1}: Missing PMID")
            continue
            
        # Fetch the title, publication year, and first author
        print(f"Fetching data for PMID {pmid} "
              f"({processed_count}/{total_rows})...")
        title, year, first_author = fetch_data_from_pmid(str(pmid))
        
        # Add the title to the dataframe
        df.at[index, 'title'] = title
        
        # Add the publication year to the dataframe if year column
        # exists or create it
        if 'year' not in df.columns:
            df['year'] = None
        df.at[index, 'year'] = year
        
        # Add the first author to the dataframe if author column
        # exists or create it
        if 'author' not in df.columns:
            df['author'] = None
        df.at[index, 'author'] = first_author
        
        # Add a delay to respect API rate limits
        time.sleep(0.5)
    
    # Save the updated dataframe to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated CSV saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add titles, publication years, and first authors ' +
                    'to CSV file based on study_pmid values.')
    parser.add_argument('input_file', type=str,
                        help='Input CSV file with study_pmid column')
    parser.add_argument('output_file', type=str,
                        help='Output CSV file with added titles')
    parser.add_argument('--sample', type=int,
                        help='Number of rows to process (for testing)')
    
    args = parser.parse_args()
    
    add_titles_years_and_authors_to_csv(
        args.input_file, args.output_file, args.sample)