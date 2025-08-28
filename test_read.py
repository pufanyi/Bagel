import pandas as pd
import os

file_path = '/mnt/raid10/pufanyi/Bagel/bagel_hf_dataset/t2i/data-00000-of-00003.arrow'

try:
    if os.path.exists(file_path):
        df = pd.read_feather(file_path)
        print("File read successfully!")
        print(df.info())
    else:
        print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")