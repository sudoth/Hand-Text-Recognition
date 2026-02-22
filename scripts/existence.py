import pandas as pd, os

df = pd.read_csv("data/processed/train.csv")
print("missing:", sum(not os.path.exists(p) for p in df["image_path"].head(500)))
