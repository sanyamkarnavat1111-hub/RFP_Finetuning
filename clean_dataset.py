import pdfplumber
import pandas as pd

df = pd.DataFrame(columns=['EA_Requirement' , 'Status' , 'RFP_Coverage' , 'Gap_analysis'])

def is_not_empty(val):
    return val is not None and str(val).strip() != ""

with pdfplumber.open("rfp-details.pdf") as pdf:
    for page in pdf.pages[1:]:
        tables = page.extract_tables()
        for row in tables:
            for element in row:
                
                if len(element) < 6:
                    continue

                element1 = element[0]
                element2 = element[1]
                element3 = element[2]
                element4 = element[3]
                element5 = element[4]
                element6 = element[5]

                # ----- Check count of 'met' in element2 -----
                if isinstance(element2, str):
                    met_count = element2.lower().count("met")
                else:
                    met_count = 0

                if met_count > 2:
                    continue   # skip this row

                # ----- NEW LOGIC: skip if ANY of element2–element5 is empty -----
                if not (
                    is_not_empty(element2) and
                    is_not_empty(element3) and
                    is_not_empty(element4) and
                    is_not_empty(element5)
                ):
                    continue

                # ----- Append to dataframe -----
                df.loc[len(df)] = [element2, element3, element4, element5]


for column in df.columns:
    
    if column == "Status":
        df[column] = df[column].replace(r'\n', '', regex=True)
    else:

        # Replace newline characters with space in the entire DataFrame
        df[column] = df[column].replace(r'\n', ' ', regex=True)

# Now Convert status to labelled data (integer)
df['Status'] = df['Status'].apply(lambda x : 1 if "partially" in str(x).lower().strip() else 0)

# Check unique columns in Status 
print(df['Status'].unique())

df.to_csv("extracted_file.csv" , index=False)