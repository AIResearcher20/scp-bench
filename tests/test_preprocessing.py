import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.preprocessing.normalize import clr_transform, preprocess_pipeline
from src.evaluation.metrics import compute_metrics


def test_clr_shape():
    X = np.random.rand(10, 5) + 1
    assert clr_transform(X).shape == X.shape


def test_clr_centered():
    X = np.random.rand(10, 5) + 1
    assert np.allclose(clr_transform(X).mean(axis=1), 0, atol=1e-6)


def test_pipeline_shapes():
    X_train = np.random.rand(80, 5) + 1
    X_test = np.random.rand(20, 5) + 1
    Xtr, Xte = preprocess_pipeline(X_train, X_test)
    assert Xtr.shape == X_train.shape
    assert Xte.shape == X_test.shape


def test_imputation_handles_nan():
    X_train = np.random.rand(50, 5) + 1
    X_test = np.random.rand(10, 5) + 1
    X_train[0, 0] = np.nan
    X_test[0, 0] = np.nan
    Xtr, Xte = preprocess_pipeline(X_train, X_test, impute=True)
    assert not np.isnan(Xtr).any()
    assert not np.isnan(Xte).any()


def test_metrics_binary():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = compute_metrics(y_true, y_pred)
    assert 0 <= m["accuracy"] <= 1
    assert "f1_macro" in m
    assert "confusion_matrix" in m
