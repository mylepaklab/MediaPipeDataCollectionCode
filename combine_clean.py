import pandas as pd
import os

# Map your filenames to gesture labels
gesture_files = {
    'letter_A.csv': 'A',
    'letter_B.csv': 'B',
    'letter_C.csv': 'C',
    # Add more files as needed
}

dfs = []

for filename, label in gesture_files.items():
    if not os.path.exists(filename):
        print(f"File {filename} not found, skipping.")
        continue
    
    # Read CSV, no matter its header situation
    # We'll fix column names below
    df = pd.read_csv(filename, header=None)  # read without headers
    
    # Fix column names — assume MediaPipe landmarks plus one label column
    # Count columns: usually 63 (21 landmarks * 3 coords) or 64 with label column
    # Let's assign generic names for landmarks, then add 'label'
    num_cols = df.shape[1]
    
    # Create column names for landmarks
    landmark_cols = []
    for i in range(num_cols - 1):
        # Naming: x0, y0, z0, x1, y1, z1, ...
        coord_type = ['x', 'y', 'z'][i % 3]
        landmark_index = i // 3
        landmark_cols.append(f"{coord_type}{landmark_index}")
        
    # Last column is label (gesture)
    landmark_cols.append('label')
    
    # Assign column names
    df.columns = landmark_cols
    
    # Now overwrite label column with the correct label (gesture)
    df['label'] = label
    
    dfs.append(df)

# Concatenate all cleaned dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save combined cleaned CSV
combined_df.to_csv("combined_clean_gesture_data.csv", index=False)

print("✅ Combined and cleaned CSV saved as 'combined_clean_gesture_data.csv'")
