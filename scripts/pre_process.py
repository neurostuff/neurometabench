# Read TSV, from column PubMedLink and extract the PubMed ID (PMID) from the URL. Add to pmid column.
# Also remove spaces from column names

import pandas as pd
import requests

def pmid_to_pmcid(pmid: str) -> str | None:
    """
    Convert a PMID to a PMCID using the NCBI ID Converter API.

    Parameters
    ----------
    pmid : str
        PubMed ID of the study

    Returns
    -------
    str | None
        PMCID if found, otherwise None
    """
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": pmid,
        "format": "json",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    records = data.get("records", [])
    if records and "pmcid" in records[0]:
        return records[0]["pmcid"].replace('PMC', '')  # Remove 'PMC' prefix for consistency
    return None


def extract_pmid_from_url(url):
    """Extract PubMed ID from a PubMed URL."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    parts = url.split('/')
    for part in parts:
        if part.isdigit():
            return part
    return None


def preprocess(input_file, output_file):
    """Read TSV file, extract PMIDs from PubMedLink column, and save to new TSV."""
    df = pd.read_csv(input_file, sep='\t')
    
    # Check if 'PubMedLink' column exists
    if 'PubMedLink' not in df.columns:
        raise ValueError("Input file must contain a 'PubMedLink' column.")
    
    # Extract PMIDs
    df['pmid'] = df['PubMedLink'].apply(extract_pmid_from_url)
    # Set as first column
    df.insert(0, 'pmid', df.pop('pmid'))

    # Remove PubMedLink column
    df.drop(columns=['PubMedLink'], inplace=True)

    # Convert PMIDs to PMCIDs
    df['pmcid'] = df['pmid'].apply(lambda x: pmid_to_pmcid(x) if pd.notna(x) else None)

    # Remove spaces from column names
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace('-', '_', regex=True)
    
    # Save to new TSV file
    df.to_csv(output_file, sep='\t', index=False)

    # Save out PMCIDs to text file
    pmcids = df['pmcid'].dropna().unique()
    f_name = output_file.replace('_pmid.tsv', '_pmcids.txt')
    with open(f_name, 'w') as f:
        for pmcid in pmcids:
            f.write(f'{pmcid}\n')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract PMIDs from PubMedLink column in a TSV file.')
    parser.add_argument('input_file', type=str, help='Input TSV file with PubMedLink column.')
    parser.add_argument('output_file', type=str, help='Output TSV file with extracted PMIDs.')

    args = parser.parse_args()
    
    preprocess(args.input_file, args.output_file)