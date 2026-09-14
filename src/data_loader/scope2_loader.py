import numpy as np
import pandas as pd
from pathlib import Path


def load_scope2_proteins(filepath):
    """
    Load SCoPE2 Proteins-processed.csv.
    Format: proteins as rows, cells as columns.
    Returns: DataFrame (proteins x cells)
    """
    df = pd.read_csv(filepath, index_col=0)
    return df


def prepare_for_benchmark(protein_df, labels=None, n_classes=2):
    """
    Convert (proteins x cells) -> (cells x proteins) with labels.
    """
    X = protein_df.T.values
    cell_names = protein_df.columns.tolist()
    protein_names = protein_df.index.tolist()

    if labels is not None:
        y = np.asarray(labels)
    else:
        # Fallback: split cells evenly into n_classes (for demo only)
        n = len(cell_names)
        y = np.array([f"Type_{i % n_classes}" for i in range(n)])

    return X, y, cell_names, protein_names
