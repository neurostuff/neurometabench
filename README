# neurometabench

**neurometabench** is a structured dataset of neuroimaging meta-analyses designed to benchmark methods for automatically reconstructing meta-analyses using AI.

The initial set of meta-analyses was sourced primarily from Dr. Angie Laird’s laboratory, with metadata manually curated for each study. Our long-term goal is to expand this resource to include a broader range of meta-analytic datasets and to develop methods that leverage large language models (LLMs) for automatic extraction of meta-analytic information from full-text articles.

---

## Dataset contents

For each study, the dataset includes:

* **Publication metadata** (e.g., PMID)
* **Identification criteria** (e.g., search parameters, methods used to find studies, total number of records identified)
* **Study objectives and inclusion/exclusion criteria**
* **Final list and count of included studies**

For studies sourced from Dr. Laird’s lab, we additionally provide the original analyses and corresponding coordinate data, enabling direct replication of published results.
*(TODO: Convert these analyses to NiMADS format for replication within NiMARE.)*

---

## Repository structure

All structured data files are located under the **`data/`** directory.

### Metadata tables

* **`included_studies.tsv`** – Links meta-analyses to their included studies.
* **`meta_datasets_pmid.txt`** – Reference list of PMIDs/PMCIDs for meta-analyses.
* **`meta_datasets.tsv`** – Curated annotations for each meta-analysis (study characteristics, inclusion/exclusion criteria, search details, etc.).

### Full-text study data

In addition to metadata tables, neurometabench provides the **fetched text and extracted data** for both meta-analyses and their included studies:

* **Meta-analysis study data** → `data/meta-studies/`
* **Individual included studies** → `data/studies/`

Each contains two subfolders:

* **`pmc-oa/`** – Studies available in PubMed Central Open Access (included in this repository).
* **`ace/`** – Non–open-access studies (not distributed on GitHub; must be obtained manually).

---

## Directory overview

```
data/
├── included_studies.tsv
├── meta_datasets_annotated_pmid_pmcids.txt
├── meta_datasets_annotated.tsv 
│
├── meta-studies/
│   ├── pmc-oa/   # Open-access meta-analysis full texts and extracted tables
│   └── ace/      # Non–open-access meta-analyses (not included in repo)
│
└── studies/
    ├── pmc-oa/   # Open-access included study full texts and extracted tables
    └── ace/      # Non–open-access included studies (not included in repo)
```

---

## Availability

* A subset of meta-analyses are **openly available in PubMed Central (PMC)**. For these, full text and tables have been fetched and processed.
* For studies **not available in PMC**, the full text must be downloaded manually and information extracted by hand.
