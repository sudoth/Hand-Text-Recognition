import pandas as pd

tr = pd.read_csv("data/processed/train.csv")
va = pd.read_csv("data/processed/val.csv")
te = pd.read_csv("data/processed/test.csv")
g = "writer_id" if "writer_id" in tr.columns else "form_id"
A = set(tr[g])
B = set(va[g])
C = set(te[g])
print("group:", g, " overlaps:", len(A & B), len(A & C), len(B & C))
