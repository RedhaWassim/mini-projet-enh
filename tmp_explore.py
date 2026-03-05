import pandas as pd

df = pd.read_csv('ton_iot_dataset/train_test_network.csv', low_memory=False)

with open('tmp_output.txt', 'w') as f:
    f.write("=== COLUMNS ===\n")
    for i, c in enumerate(df.columns):
        f.write(f"  {i:2d}: {c} (dtype={df[c].dtype})\n")

    f.write(f"\nShape: {df.shape}\n")

    f.write(f"\n=== TYPE (attack category) ===\n")
    f.write(df['type'].value_counts().to_string() + "\n")

    f.write(f"\n=== LABEL (binary) ===\n")
    f.write(df['label'].value_counts().to_string() + "\n")

    f.write(f"\n=== SAMPLE VALUES (first 2 rows) ===\n")
    for c in df.columns:
        f.write(f"  {c}: {df[c].iloc[0]} | {df[c].iloc[1]}\n")

    nans = df.isnull().sum()
    f.write(f"\n=== NULLS ===\n")
    f.write((nans[nans > 0].to_string() if nans.sum() > 0 else "No nulls") + "\n")

    dash_counts = (df == '-').sum()
    f.write(f"\n=== DASH PLACEHOLDERS ===\n")
    f.write(dash_counts[dash_counts > 0].to_string() + "\n")

print("Done. Output saved to tmp_output.txt")
