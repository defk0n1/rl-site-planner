import numpy as np
import pandas as pd
import os
# print("Files in current directory:")
# for file in os.listdir('.'):
#     print(f"'{file}'")



def load_clutter_map(path):
    # --- Step 1: Read clutter index and clutter data ---
    print("[1] Reading clutter index and data...")
    clutter_index = pd.read_csv(path, sep=r"\s+", names=["filename", "min_x", "max_x", "min_y", "max_y", "res"])
    print(f"Clutter index : {clutter_index}")

    clutter_row = clutter_index.iloc[1]

    res = clutter_row["res"]


    width = int((clutter_row["max_x"] - clutter_row["min_x"]) / res)
    height = int((clutter_row["max_y"] - clutter_row["min_y"]) / res)
    print(f"[1.1] Loaded clutter tile: {clutter_row['filename']} ({width}x{height})")


    # Partially load subarray from clutter data
    whole_clutter_data = np.memmap(clutter_row["filename"], dtype=np.uint16).reshape((height, width))

    clutter_data = whole_clutter_data[5000:5500, 5000:5500]

    print(f"Loaded clutter data: {clutter_data}")
    print(f'Clutter unique values : {np.unique(clutter_data)}')

    # Parse clutter mapping (mapping original values to clutter classes)
    original_values = np.array([
        256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
        2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 5120, 61912
    ])
    new_classes = np.where(original_values == 61912, 0, original_values // 256)
    mapping = dict(zip(original_values, new_classes))
    raster_mapped = np.vectorize(mapping.get)(clutter_data)
    return raster_mapped

