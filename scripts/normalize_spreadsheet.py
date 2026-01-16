import pandas as pd
import numpy as np

# === Input / Output paths ===
file_path = "raw/Meta-Analysis II - including_exclusion.xlsx"
output_path = "raw/Meta-Analysis_normalized.xlsx"

# === Load Excel file ===
excel_file = pd.ExcelFile(file_path)


# === Load Excel ===
xls = pd.ExcelFile(file_path)
sheets = [s for s in xls.sheet_names if s != "2019 articles (to be reviewed)"]

def load_and_normalize(df, sheet_name):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in ["Name", "Author", "Year", "Status"]:
        if c not in df.columns:
            df[c] = np.nan

    # Handle Unnamed: 5
    print(df.columns)
    reason_col = None
    for c in ['Reason', 'Unnamed: 5', 'Unnamed: 4']:
        if c in df.columns:
            reason_col = c
            break
    if reason_col is not None:
        reason = df[reason_col].astype(str).str.strip()
        reason = reason.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    else:
        reason = pd.Series(np.nan, index=df.index)

    # NEW logic:
    # - YES sheet: always use Unnamed:5 content (even if blank)
    # - Other sheets: use Unnamed:5 if present, else sheet name as reason
    if sheet_name == "YES":
        df["Reason_raw"] = reason
        df["Reason"] = reason  # keep blank if Unnamed:5 empty
    else:
        df["Reason_raw"] = reason
        df["Reason"] = reason.fillna(sheet_name)

    # Source sheet info
    df["SourceSheet"] = sheet_name

    # Normalize Status text
    df["Status"] = df["Status"].astype(str).str.strip().str.upper()
    df["Status"] = df["Status"].replace({"NAN": np.nan, "NONE": np.nan})
    df["Status"] = df["Status"].fillna("YES")

    # Clean keys
    df["Name_clean"] = df["Name"].astype(str).str.strip().str.lower()
    df["Author_clean"] = df["Author"].astype(str).str.strip().str.lower()
    df["Year_clean"] = df["Year"].astype(str).str.strip().str.lower()

    return df[[
        "Name", "Author", "Year", "Status", "Reason", "Reason_raw",
        "SourceSheet", "Name_clean", "Author_clean", "Year_clean"
    ]]


# Load all sheets into list of DataFrames
dfs = []
for s in sheets:
    df = xls.parse(s)
    dfs.append(load_and_normalize(df, s))

combined = pd.concat(dfs, ignore_index=True)

# Group key: prefer exact match on Name+Author+Year; fallback to Name only if needed.
combined["group_key"] = (
    combined["Name_clean"].fillna("") + "||" +
    combined["Author_clean"].fillna("") + "||" +
    combined["Year_clean"].fillna("")
)

# Helper to select one row per group based on priority rules
def pick_best_row(group):
    # Make a copy to avoid SettingWithCopy warnings
    g = group.copy()

    # Create helper boolean columns
    g["has_reason_text"] = g["Reason_raw"].notna()  # true if Unnamed:5 had text
    g["is_no"] = g["Status"] == "NO"
    g["is_yes"] = g["Status"] == "YES"

    # 1) Any row with is_no AND has_reason_text -> pick the first (or last) of these
    candidates = g[(g["is_no"]) & (g["has_reason_text"])]
    if not candidates.empty:
        return candidates.iloc[-1]  # pick last to prefer later sheets if duplicated

    # 2) Any row with is_no -> pick one (use last)
    candidates = g[g["is_no"]]
    if not candidates.empty:
        return candidates.iloc[-1]

    # 3) Any YES with reason text (from YES sheet Unnamed:5) -> pick last
    candidates = g[(g["is_yes"]) & (g["has_reason_text"])]
    if not candidates.empty:
        return candidates.iloc[-1]

    # 4) fallback: pick last row
    return g.iloc[-1]

# Apply grouping
selected_rows = combined.groupby("group_key", dropna=False).apply(lambda grp: pick_best_row(grp))

# Fix index if MultiIndex
if isinstance(selected_rows.index, pd.MultiIndex):
    selected_rows.index = selected_rows.index.droplevel(0)

# Clean up final DataFrame
final = selected_rows.reset_index(drop=True)
# If Reason is still NaN for YES rows, set to "Included (YES)"
mask_yes_no_reason = (final["Status"] == "YES") & (final["Reason"].isna())
final.loc[mask_yes_no_reason, "Reason"] = "Included (YES)"

# For completeness, if Reason is NaN but Status == NO, fill with SourceSheet as reason
mask_no_no_reason = (final["Status"] == "NO") & (final["Reason"].isna())
final.loc[mask_no_no_reason, "Reason"] = final.loc[mask_no_no_reason, "SourceSheet"]

# Drop helper columns before saving
final = final.drop(columns=[
    "Reason_raw", "Name_clean", "Author_clean", "Year_clean", "group_key"
], errors='ignore')

# Reorder columns
cols_order = ["Name", "Author", "Year", "Status", "Reason", "SourceSheet"]
cols_order = [c for c in cols_order if c in final.columns]
final = final[cols_order]

# Save
final.to_excel(output_path, index=False)
print(f"✅ Saved normalized file to: {output_path}")