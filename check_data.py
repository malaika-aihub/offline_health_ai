import pandas as pd

df = pd.read_csv("data/raw/HAM10000_metadata.csv")

print(df['image_id'].head(10))
