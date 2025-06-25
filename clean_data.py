import pandas as pd

df = pd.read_csv("heart_disease.csv")
df = df.drop_duplicates()
df = df.dropna()

# finding numeric columns.
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Apply min-max normalization
df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())