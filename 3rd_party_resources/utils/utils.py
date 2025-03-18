import pandas as pd

df = pd.read_parquet('train-00000-of-00001-aaf72b9960b78228.parquet')
df.to_csv('data.csv', index=False)

