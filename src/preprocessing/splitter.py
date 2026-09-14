from sklearn.model_selection import train_test_split, StratifiedKFold


def split_cell_aware(X, y, test_size=0.2, random_state=42):
    """Cell-aware stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def get_cv_splitter(n_splits=5, random_state=42):
    """StratifiedKFold cross-validator."""
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
