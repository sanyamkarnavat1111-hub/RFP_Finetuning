import pandas as pd

df = pd.read_csv("Dataset/extracted_file.csv")

# Convert 1/0 to text labels
df["Status"] = df["Status"].map({
    1: "partially met",
    0: "not met"
})

df.to_csv("Dataset/extracted_file_converted.csv", index=False)

print(df.head())
