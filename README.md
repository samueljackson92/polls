# UK Polling Data Scraper

This project downloads national polling results for the next United Kingdom general election from Wikipedia and saves them as a parquet file.

## Installation

Install the package and its dependencies:

```bash
pip install -e .
```

## Usage

Run the script to download the latest polling data:

```bash
python download_uk_polls.py
```

This will:
1. Fetch polling data from the Wikipedia page
2. Parse and clean the polling tables
3. Save the results to `uk_polling_data.parquet`

## Output

The script creates a parquet file containing:

**Metadata columns:**
- `poll_date` - Date or date range when poll was conducted
- `pollster` - Polling organization name (e.g., YouGov, Ipsos, Opinium)
- `client` - Client/commissioner of the poll
- `area` - Geographic area (GB or UK)
- `sample_size` - Number of respondents
- `year` - Year the poll was conducted (extracted from Wikipedia table headings)

**Party support columns (percentages):**
- `con_pct` - Conservative Party
- `lab_pct` - Labour Party
- `lib_dem_pct` - Liberal Democrats
- `reform_pct` - Reform UK
- `green_pct` - Green Party
- `snp_pct` - Scottish National Party
- `plaid_pct` - Plaid Cymru
- `others_pct` - Other parties

**Additional columns:**
- `lead` - Lead margin between top parties
- `extracted_at` - Timestamp when data was downloaded

All column names use snake_case for consistency in data science workflows.

## Data Source

Data is scraped from: https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_United_Kingdom_general_election