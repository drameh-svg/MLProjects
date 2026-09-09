'''
Zeinab Drameh
ACAD 222,
Spring 2026
Final Project Part 1
'''

import os
import pandas as pd

# Step 1: Build simple dataset
data = []

for split in ["train", "test", "val"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = f"chest_xray/{split}/{label}"

        for img in os.listdir(folder):
            data.append({
                "image_path": os.path.join(folder, img),
                "label": label
            })

df = pd.DataFrame(data)

# Step 2: Convert label to numeric
df["label_num"] = df["label"].map({"NORMAL": 0, "PNEUMONIA": 1})

# Step 3: Info
print("\nINFO:")
df.info()

# Step 4: Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Step 5: Describe
print("\nDESCRIBE:")
print(df.describe())

# Step 6: Correlation
print("\nCORRELATION:")
print(df.corr(numeric_only=True))