"""
Script to extract UK polling data from HTML and save as parquet file.
Extracts tables from the National poll results section and combines them into one dataframe.
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import requests
from pathlib import Path


def parse_poll_date(date_str, year):
    """
    Parse polling date string and extract the end date.

    Args:
        date_str: Date string like "29-30 Apr", "27 Apr", "27 Mar–7 Apr2026", "31 Aug – 24 Sep 2025"
        year: Year as integer (fallback if year not in date string)

    Returns:
        Date string in YYYY-MM-DD format
    """
    if not date_str:
        return None

    # Remove nowrap tags and extra whitespace
    date_str = date_str.replace("\u00a0", " ").strip()

    # Check if year is embedded in the date string (e.g., "Apr2026" or "2025" at the end)
    year_match = re.search(r"(\d{4})", date_str)
    if year_match:
        year = int(year_match.group(1))
        # Remove the year from the string to simplify parsing
        date_str = re.sub(r"\d{4}", "", date_str).strip()

    if not year:
        return None

    # Map month abbreviations to numbers
    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    # Pattern for dates like "29-30 Apr" or "24-27 Apr" or "27 Apr" (single month)
    # We want to extract the last day and month
    single_month_match = re.search(r"(\d+)\s*([A-Za-z]+)$", date_str)

    # Pattern for cross-month ranges like "27 Mar–7 Apr" or "31 Aug – 24 Sep"
    # Format: day month [dash] day month
    cross_month_match = re.search(
        r"\d+\s*([A-Za-z]+)\s*[–-]\s*(\d+)\s*([A-Za-z]+)", date_str
    )

    # Pattern for month-only dates like "Dec" (year already extracted)
    month_only_match = re.search(r"^([A-Za-z]+)$", date_str)

    if cross_month_match:
        # Cross-month range - use the end date (last day and last month)
        day = int(cross_month_match.group(2))
        month_str = cross_month_match.group(3)
        month = month_map.get(month_str, None)

        if month:
            try:
                date_obj = datetime(year, month, day)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                return None

    elif single_month_match:
        # Single month range or single date
        day = int(single_month_match.group(1))
        month_str = single_month_match.group(2)
        month = month_map.get(month_str, None)

        if month:
            try:
                date_obj = datetime(year, month, day)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                return None

    elif month_only_match:
        # Month only (like "Dec") - use last day of the month
        month_str = month_only_match.group(1)
        month = month_map.get(month_str, None)

        if month:
            # Get the last day of the month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            last_day = next_month - timedelta(days=1)
            return last_day.strftime("%Y-%m-%d")

    return None


def extract_polling_tables(html_file):
    """
    Extract polling data from HTML file.

    Args:
        html_file: Path to the HTML file

    Returns:
        DataFrame with combined polling data
    """
    # Read the HTML file
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Find the National poll results section
    national_section = soup.find("h2", id="National_poll_results")

    if not national_section:
        raise ValueError("Could not find National poll results section")

    # Find all year headings (h3 tags with numeric IDs like "2026", "2025", "2024")
    # and the tables that follow them
    tables = []
    table_years = []

    # Get all h3 elements that could be year headings
    all_headings = soup.find_all("h3")
    year_headings = []

    for heading in all_headings:
        heading_id = heading.get("id", "")
        if heading_id.isdigit() and len(heading_id) == 4:
            year_headings.append((int(heading_id), heading))
            print(f"Found year heading: {heading_id}")

    print(f"Total year headings found: {len(year_headings)}")
    print(f"Year headings: {[y for y, h in year_headings]}")

    # Find the Seat projections section to know where to stop
    seat_projections_section = soup.find("h2", id="Seat_projections")
    sub_national_section = soup.find("h2", id="Sub-national_poll_results")

    # Find all wiki tables in the document
    all_tables = soup.find_all("table", class_="wikitable")
    print(f"Found {len(all_tables)} total wikitable tables in document")

    # Get document HTML once for efficiency
    all_html = str(soup)

    # Get positions of year headings in the document
    year_positions = []
    for year, heading in year_headings:
        heading_text = f'<h3 id="{year}"'
        pos = all_html.find(heading_text)
        if pos >= 0:
            year_positions.append((pos, year))
    year_positions.sort()  # Sort by position

    # Get position of Seat projections section
    seat_proj_pos = len(all_html)  # Default to end
    if seat_projections_section:
        seat_proj_text = '<h2 id="Seat_projections"'
        pos = all_html.find(seat_proj_text)
        if pos >= 0:
            seat_proj_pos = pos

    print(f"Year positions in document: {[(y, p) for p, y in year_positions]}")
    print(f"Seat projections at position: {seat_proj_pos}")

    # Assign each table to a year based on its position in the document
    table_info = []
    for idx, table in enumerate(all_tables):
        # Get table position in document using the full table HTML to ensure uniqueness
        table_html = str(table)
        table_pos_in_doc = all_html.find(table_html)

        if table_pos_in_doc < 0:
            print(f"  Table {idx + 1} - could not find position in document, skipping")
            continue  # Table not found in HTML (shouldn't happen)

        # Skip tables after Seat projections section
        if table_pos_in_doc > seat_proj_pos:
            print(
                f"  Table {idx + 1} at position {table_pos_in_doc} - AFTER Seat projections, skipping"
            )
            continue

        # Find the closest preceding year heading
        assigned_year = None
        for pos, year in reversed(year_positions):
            if pos < table_pos_in_doc:
                assigned_year = year
                print(
                    f"  Table {idx + 1} at position {table_pos_in_doc} - assigned to year {year}"
                )
                break

        if assigned_year:
            tables.append(table)
            table_years.append(assigned_year)
            table_info.append((table_pos_in_doc, assigned_year))

    print(
        f"\nFound {len(tables)} tables in National poll results section (before Seat projections)"
    )
    print(
        f"Tables per year: {dict((y, table_years.count(y)) for y in set(table_years))}"
    )

    # Extract data from each table
    all_data = []

    for table_idx, table in enumerate(tables):
        print(f"\nProcessing table {table_idx + 1} (year {table_years[table_idx]})...")
        table_year = table_years[table_idx]

        # Find all rows
        rows = table.find_all("tr")

        if len(rows) < 2:
            print(f"  Skipping - only {len(rows)} rows")
            continue

        # Extract headers from first two rows (they have a 2-row header structure)
        headers = []
        header_rows = rows[:2]

        # Process the first header row to get party names
        first_header = header_rows[0]
        for th in first_header.find_all("th"):
            # Get the text content
            text = th.get_text(strip=True)

            # Handle rowspan for multi-row headers
            rowspan = int(th.get("rowspan", 1))

            if rowspan == 2:
                headers.append(text)
            else:
                # These will be party abbreviations
                # Extract the link text which contains the party abbreviation
                link = th.find("a")
                if link:
                    headers.append(link.get_text(strip=True))
                else:
                    headers.append(text)

        # Process data rows
        rows_added_from_table = 0
        for row in rows[2:]:
            cells = row.find_all("td")

            if len(cells) == 0:
                continue

            row_data = {}

            # Track the current column position accounting for colspan
            col_position = 0

            for cell in cells:
                # Get colspan value (default is 1)
                colspan = int(cell.get("colspan", 1))

                # Assign the cell value to the appropriate header(s)
                # For merged cells (colspan > 1), we only use the first column's header
                if col_position < len(headers):
                    header = headers[col_position]

                    # Extract text and clean it
                    text = cell.get_text(strip=True)

                    # Handle percentages - remove % sign
                    if "%" in text:
                        text = text.replace("%", "")

                    # Handle em-dash (—) for missing values
                    if text == "—" or text == "":
                        text = None

                    # Handle hidden content (collapsed sections)
                    # Just extract the first visible number for "Others"
                    if header == "Others" and text:
                        # Extract first number from text
                        match = re.search(r"(\d+)%", cell.get_text())
                        if match:
                            text = match.group(1)

                    row_data[header] = text

                # Move to next column position, accounting for colspan
                col_position += colspan

            # Add year to the row
            if row_data and table_year:
                row_data["Year"] = table_year

            if row_data:
                all_data.append(row_data)
                rows_added_from_table += 1

        print(f"  Added {rows_added_from_table} data rows from this table")

    # Create DataFrame
    df = pd.DataFrame(all_data)

    print(f"Extracted {len(df)} polling records")

    # Clean up column names - standardize party columns
    party_columns_map = {
        "Lab": "Labour",
        "Con": "Conservative",
        "Ref": "Reform",
        "LD": "Liberal_Democrats",
        "Grn": "Green",
        "SNP": "SNP",
        "PC": "Plaid_Cymru",
    }

    # Rename date column to "Dates"
    date_col = None
    for col in df.columns:
        if "date" in col.lower() and "conduct" in col.lower():
            date_col = col
            break

    if date_col:
        party_columns_map[date_col] = "Dates"

    # Rename columns
    df.rename(columns=party_columns_map, inplace=True)

    # Parse dates and create a new "Date" column with the end date in YYYY-MM-DD format
    if "Dates" in df.columns and "Year" in df.columns:
        df["Date"] = df.apply(
            lambda row: parse_poll_date(row["Dates"], row["Year"]), axis=1
        )

    # Clean citations from Pollster column (e.g., "YouGov[2]" -> "YouGov", "Pollster[a][b]" -> "Pollster")
    if "Pollster" in df.columns:
        df["Pollster"] = (
            df["Pollster"].astype(str).str.replace(r"\[[a-zA-Z0-9]+\]", "", regex=True)
        )

    # Convert percentage columns to numeric
    party_cols = [
        "Labour",
        "Conservative",
        "Reform",
        "Liberal_Democrats",
        "Green",
        "SNP",
        "Plaid_Cymru",
        "Others",
        "Lead",
    ]

    for col in party_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert Sample size to numeric
    if "Samplesize" in df.columns:
        df["Samplesize"] = pd.to_numeric(
            df["Samplesize"].astype(str).str.replace(",", ""), errors="coerce"
        )
    elif "Sample size" in df.columns:
        df.rename(columns={"Sample size": "Sample_size"}, inplace=True)
        df["Sample_size"] = pd.to_numeric(
            df["Sample_size"].astype(str).str.replace(",", ""), errors="coerce"
        )

    df = df.rename(columns={"Samplesize": "Sample_size"}, inplace=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.sort_values(by=["date"], ascending=True).reset_index(drop=True)
    return df


def download_wikipedia_page(url, output_file):
    """
    Download Wikipedia page and save to file.

    Args:
        url: Wikipedia URL to download
        output_file: Path to save HTML file
    """
    print(f"Downloading page from {url}...")

    # Make request with user agent to avoid blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise error if request failed

    # Create directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save HTML to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Successfully saved HTML to {output_file}")


def main():
    """Main execution function."""
    wikipedia_url = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_United_Kingdom_general_election"
    html_file = "data/raw/poll.html"
    output_file = "data/processed/uk_polling_data.parquet"

    # Download latest Wikipedia page
    download_wikipedia_page(wikipedia_url, html_file)

    print(f"\nReading polling data from {html_file}...")
    df = extract_polling_tables(html_file)

    # Display info about the dataframe
    print("\nDataFrame Info:")
    print(df.info())

    print("\nFirst few rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    # Create directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False)

    print(f"Successfully saved {len(df)} polling records to {output_file}")


if __name__ == "__main__":
    main()
