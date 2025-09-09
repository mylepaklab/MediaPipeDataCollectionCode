import pandas as pd
import os
import string  # for A-Z letters

dfs = []

# Letters A-Z
gesture_labels = list(string.ascii_uppercase)

# Add custom gestures
gesture_labels += ['ya', 'stop']

# Add numbers 1 through 10
gesture_labels += [str(i) for i in range(1, 11)]

# Add custom numbers
gesture_labels += ['11', '21', '50', '70', '100']

for label in gesture_labels:
    # Use 'letter' prefix for letters and words, 'number' for digits
    if label.isalpha() or label.lower() in ['ya', 'stop']:
        filename = f"letter_{label}.csv"
    else:
        filename = f"number_{label}.csv"

    if not os.path.exists(filename):
        print(f"⚠️ File {filename} not found, skipping.")
        continue

    # Read CSV without headers
    df = pd.read_csv(filename, header=None)

    # Generate landmark column names
    num_cols = df.shape[1]
    landmark_cols = []
    for i in range(num_cols - 1):
        coord_type = ['x', 'y', 'z'][i % 3]
        landmark_index = i // 3
        landmark_cols.append(f"{coord_type}{landmark_index}")
    landmark_cols.append('label')

    # Assign column names
    df.columns = landmark_cols

    # Assign correct label as uppercase string (even for numbers and words)
    df['label'] = str(label).upper()

    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Force label column to string
combined_df['label'] = combined_df['label'].astype(str)

# Save to CSV
combined_df.to_csv("combined_clean_gesture_data.csv", index=False)
print("✅ Combined and cleaned CSV saved as 'combined_clean_gesture_data.csv'")

