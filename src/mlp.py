import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import settings as sett

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def compute_metrics(model, dataloader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def make_model():
    return MLP(sett.input_size, sett.hidden_size, sett.output_size)

def train_one_fold(model, train_loader, val_loader, epochs, lr, patience, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    bad_epochs = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            pred = model(Xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    return best_val

def cross_validate(X, y, model_fn, n_splits=5, batch_size=32, epochs=50, lr=1e-3, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_scores = []

    with mlflow.start_run():

        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

            model = model_fn()

            best_val = train_one_fold(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=lr,
                patience=patience,
                device=device
            )

            fold_scores.append(best_val)

            metrics = compute_metrics(model, val_loader, device)

            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric(f"fold_{fold}_f1", metrics["f1"])
            mlflow.log_metric(f"fold_{fold}_val_loss", best_val)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        mlflow.log_metric("cv_mean_val_loss", mean_score)
        mlflow.log_metric("cv_std_val_loss", std_score)
        torch.save(model.state_dict(), sett.model_name)

# X is a torch tensor. Must be preproc.
def predict(model, X):
#    model = MLP(sett.input_size, sett.hidden_size, sett.output_size)
#    model.load_state_dict(torch.load(sett.model_name))
    model.eval()

    with torch.no_grad():
        output = model(X)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0
    return pred, prob



if __name__ == "__main__":
    input_size = 10
    hidden_size = 20
    output_size = 1
    learning_rate = 0.01

    model = MLP(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    X = torch.randn(100, input_size)
    y = torch.randn(100, output_size)

    epochs = 100
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# checkpoint = torch.load("checkpoint.pth")
# model.load_state_dict(checkpoint["model_state"])
# optimizer.load_state_dict(checkpoint["optimizer_state"])
# start_epoch = checkpoint["epoch"]