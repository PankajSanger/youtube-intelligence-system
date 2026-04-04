import pandas as pd
import os

DATA_PATH = "data/youtube_data_for_project.xlsx"


def save_new_data(new_data):
    """
    Save new fetched YouTube data into Excel dataset
    Handles:
    - append
    - deduplication
    """

    df_new = pd.DataFrame(new_data)

    # Load existing data if present
    if os.path.exists(DATA_PATH):
        df_old = pd.read_excel(DATA_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # Remove duplicates based on video_id
    df.drop_duplicates(subset=["video_id"], inplace=True)

    # Save updated dataset
    df.to_excel(DATA_PATH, index=False)

    return df