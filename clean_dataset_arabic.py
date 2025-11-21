import pdfplumber
import pandas as pd
import re


## The exported PDF text of arabic language has just "?" hence the code is not working 


# Output DataFrame with English column names
df = pd.DataFrame(columns=['EA_Requirement', 'Status', 'RFP_Coverage', 'Gap_analysis'])

# Helper function to check for empty values
def is_not_empty(val):
    return val is not None and str(val).strip() != ""

# Arabic keyword for partial status (update if needed)
PARTIAL_KEYWORD = "جزئي"  # 'partially' in Arabic

# Characters to remove: newline, RTL mark, Arabic tatweel
def clean_text(text):
    if text is None:
        return ""
    return re.sub(r'[\n\u200f\u0640]', ' ', str(text)).strip()

with pdfplumber.open("Dataset/rfp-details-arabic.pdf") as pdf:
    for page in pdf.pages[1:]:  # Skip first page if needed
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                if len(row) < 6:  # Ensure row has enough columns
                    continue

                # Extract elements from row
                element2 = clean_text(row[1])
                element3 = clean_text(row[2])
                element4 = clean_text(row[3])
                element5 = clean_text(row[4])

                # Skip if any critical element is empty
                if not (is_not_empty(element2) and is_not_empty(element3) and
                        is_not_empty(element4) and is_not_empty(element5)):
                    continue

                # Append to dataframe
                df.loc[len(df)] = [element2, element3, element4, element5]

# Convert Status column to integer labels
# 1 if partial (جزئي) else 0
df['Status'] = df['Status'].apply(lambda x: 1 if PARTIAL_KEYWORD in str(x) else 0)

# Optionally, replace any leftover newlines in all columns
for col in df.columns:
    df[col] = df[col].apply(clean_text)

# Check unique Status values
print("Unique Status labels:", df['Status'].unique())

# Save to CSV
df.to_csv("Dataset/extracted_file_arabic.csv", index=False)
print("Arabic table extraction complete. CSV saved.")
