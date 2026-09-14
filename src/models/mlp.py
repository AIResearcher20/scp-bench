"""PyTorch MLP classifier with sklearn-compatible API."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch MLP with sklearn-compatible fit/predict API."""

    def __init__(self, hidden_dims=(256, 128), epochs=100, lr=1e-3,
                 batch_size=64, random_state=42, verbose=False):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, n_features, n_classes):
        layers = []
        prev = n_features
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.model_ = self._build_model(n_features, n_classes)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        self.model_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                out = self.model_(X_t[idx])
                loss = criterion(out, y_t[idx])
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
