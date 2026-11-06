from pathlib import Path
from typing import Union

import pandas as pd


def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)


def load_text_data(path: Union[str, Path]) -> list[str]:
    """
    Load text data from a CSV file.

    The CSV file must contain a column named 'text'.

    Args:
        path: Path to the CSV file.

    Returns:
        A list containing non-empty text values.

    Raises:
        KeyError: If the 'text' column does not exist.
    """
    data = load_dataframe(path)

    if "text" not in data.columns:
        raise KeyError(
            "The dataset must contain a 'text' column."
        )

    texts = (
        data["text"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return texts[texts != ""].tolist()
