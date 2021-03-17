import pandas as pd

# Load datasets (from excel file).
d1 = pd.read_excel("data\\dataset-a.xlsx", sheet_name="ECG1A03_new", header=1)
d2 = pd.read_excel("data\\dataset-b.xlsx", sheet_name="ECG1A04_new", header=1)

print(d1.columns.tolist())
print(d2.columns.tolist())

