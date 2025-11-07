import pandas as pd

# === Input paths ===
normalized_path = "raw/Meta-Analysis_normalized.xlsx"
annotations_path = "data/all_studies_titles.csv"
output_path = "data/all_studies_annotated_2.csv"

# === Load both datasets ===
normalized = pd.read_excel(normalized_path)
annotations = pd.read_csv(annotations_path)

# === Standardize title columns for merging ===
normalized["Name_clean"] = normalized["Name"].str.strip().str.lower()
annotations["title_clean"] = annotations["title"].str.strip().str.lower()

# Remove period at end of titles for better matching
annotations["title_clean"] = annotations["title_clean"].str.rstrip(".")


# === Merge ===
merged = pd.merge(
    annotations,
    normalized,
    how="left",
    left_on="title_clean",
    right_on="Name_clean"
)

# === Clean up ===
merged.drop(columns=["Name_clean", "title_clean"], inplace=True)

# === Optional: reorder columns for readability ===
cols = [
    "meta_pmid", "study_pmid", "status", "final_status", "reason", "title",
    "Author", "Year", "Status", "Reason", "SourceSheet"
]
merged = merged[[c for c in cols if c in merged.columns]]

# === Save output ===
merged.to_csv(output_path, index=False)

print(f"✅ Merged spreadsheet saved to: {output_path}")
