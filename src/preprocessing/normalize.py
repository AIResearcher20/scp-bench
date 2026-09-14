import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def clr_transform(X):
    """Centered Log-Ratio transformation."""
    X = np.asarray(X, dtype=float)
    X = np.clip(X, 1e-6, None)
    log_x = np.log(X)
    return log_x - log_x.mean(axis=1, keepdims=True)


def impute_knn(X_train, X_test, n_neighbors=5):
    """Leakage-safe KNN imputation: fit on train only."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test


def preprocess_pipeline(X_train, X_test, method="clr",
                        impute=True, scale=True, n_neighbors=5):
    """Full leakage-safe pipeline: impute -> CLR -> scale."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if impute:
        X_train, X_test = impute_knn(X_train, X_test, n_neighbors=n_neighbors)

    if method == "clr":
        X_train = clr_transform(X_train)
        X_test = clr_transform(X_test)
    else:
        raise ValueError(f"Unknown method: {method}")

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test
