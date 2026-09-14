"""scProteomicsBench: main benchmarking pipeline."""
import os
import sys
import json
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocessing.normalize import preprocess_pipeline
from src.preprocessing.splitter import split_cell_aware, get_cv_splitter
from src.evaluation.metrics import compute_metrics


def get_models():
    """Return dict of models."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000,
                                                   random_state=42,
                                                   class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100,
                                                random_state=42,
                                                class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric="logloss",
                                  verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42,
                                    verbose=-1),
    }


def run_benchmark(data_path, config=None):
    """Run full benchmarking pipeline."""
    if config is None:
        config = {"test_size": 0.2, "cv_folds": 3}

    print(f"📥 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")

    label_col = df.columns[-1]
    y_raw = df[label_col].values
    X = df.drop(columns=[label_col]).values

    # Encode labels to integers (0, 1, ...)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"   X: {X.shape}, y: {len(y)} samples, "
          f"{len(le.classes_)} classes")
    print(f"   Classes: {le.classes_.tolist()}")

    print("\n✂️  Leakage-safe split...")
    X_train, X_test, y_train, y_test = split_cell_aware(
        X, y, test_size=config["test_size"]
    )

    print("🔧 Preprocessing (KNN impute -> CLR -> scale)...")
    X_train, X_test = preprocess_pipeline(X_train, X_test,
                                            impute=True, scale=True)

    cv = get_cv_splitter(n_splits=config["cv_folds"])
    results = {}

    for name, model in get_models().items():
        print(f"\n🔬 {name}")
        try:
            # CV
            cv_scores = []
            for tr, va in cv.split(X_train, y_train):
                m = type(model)(**model.get_params())
                m.fit(X_train[tr], y_train[tr])
                pred = m.predict(X_train[va])
                cv_scores.append(
                    compute_metrics(y_train[va], pred)["f1_macro"]
                )

            # Train final
            tracemalloc.start()
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Test
            y_pred = model.predict(X_test)
            y_proba = (model.predict_proba(X_test)
                       if hasattr(model, "predict_proba") else None)
            test_metrics = compute_metrics(y_test, y_pred, y_proba)

            results[name] = {
                "cv_f1_macro_mean": float(np.mean(cv_scores)),
                "cv_f1_macro_std": float(np.std(cv_scores)),
                "test_metrics": test_metrics,
                "train_time_sec": float(train_time),
                "peak_memory_mb": float(peak / 1024 / 1024),
            }
            print(f"   CV F1: {np.mean(cv_scores):.4f} ± "
                  f"{np.std(cv_scores):.4f}")
            print(f"   Test accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Time: {train_time:.2f}s | "
                  f"Memory: {peak/1024/1024:.2f} MB")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    data_path = (sys.argv[1] if len(sys.argv) > 1
                 else "data/processed/dataset.csv")
    results = run_benchmark(data_path)

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to results/benchmark_results.json")
