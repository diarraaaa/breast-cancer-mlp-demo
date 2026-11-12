"""Entraîne le MLP du TP1 (PyTorch) et exporte poids + scaler dans docs/model.json.

Reprend exactement le pipeline du notebook Tp1_Deep_Learning.ipynb :
split stratifié 60/20/20 (random_state=42), StandardScaler ajusté sur train,
CancerMLP 30 -> 16 -> 8 -> 1, SGD lr=0.01, batch 32, 75 époques, seed 42.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / "docs" / "model.json"

# 1. Données : sklearn code 0 = maligne, on ré-encode comme le TP (B = 0, M = 1)
data = load_breast_cancer()
X = data.data.astype(np.float64)
y = (1 - data.target).astype(np.int64)
feature_names = list(data.feature_names)

# 2. Split 60/20/20 stratifié
indices = np.arange(len(y))
idx_train, idx_temp = train_test_split(indices, test_size=0.4, stratify=y, random_state=42)
idx_dev, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=42)

# 3. Standardisation (fit sur train uniquement)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X[idx_train])
X_dev_s = scaler.transform(X[idx_dev])
X_test_s = scaler.transform(X[idx_test])


# 4. Modèle
class CancerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)


def to_t(a):
    return torch.tensor(a, dtype=torch.float32)


torch.manual_seed(42)
net = CancerMLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

X_train_t, y_train_t = to_t(X_train_s), to_t(y[idx_train]).unsqueeze(1)
X_dev_t, y_dev_t = to_t(X_dev_s), to_t(y[idx_dev]).unsqueeze(1)
loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# 5. Entraînement
for epoch in range(75):
    net.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(net(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) in (1, 15, 30, 45, 60, 75):
        net.eval()
        with torch.no_grad():
            tr = criterion(net(X_train_t), y_train_t).item()
            dv = criterion(net(X_dev_t), y_dev_t).item()
        print(f"Époque {epoch + 1:02d} | train {tr:.4f} | dev {dv:.4f}")

# 6. Évaluation test
net.eval()
with torch.no_grad():
    prob = torch.sigmoid(net(to_t(X_test_s))).numpy().ravel()
pred = (prob >= 0.5).astype(int)
y_test = y[idx_test]
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "auc": roc_auc_score(y_test, prob),
}
print({k: round(v, 4) for k, v in metrics.items()})


# 7. Export JSON (poids au format PyTorch : [out, in])
def p(t):
    return t.detach().numpy().tolist()


export = {
    "feature_names": feature_names,
    "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
    "layers": [
        {"W": p(net.fc1.weight), "b": p(net.fc1.bias), "activation": "relu"},
        {"W": p(net.fc2.weight), "b": p(net.fc2.bias), "activation": "relu"},
        {"W": p(net.fc3.weight), "b": p(net.fc3.bias), "activation": "sigmoid"},
    ],
    "test_metrics": metrics,
    "test_set": [
        {"index": int(i), "label": int(y[i]), "x": X[i].tolist(), "py_prob": float(pr)}
        for i, pr in zip(idx_test, prob)
    ],
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(export), encoding="utf-8")
print(f"Exporté -> {OUT}")
