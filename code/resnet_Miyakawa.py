import json, torch, argparse, time, csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader




class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out
    
class ResNetBF(nn.Module):
    def __init__(self, input_dim, num_classes=10, width=512, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True)
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(width))
        self.blocks = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(width, num_classes)
    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.blocks(out)
        out = self.output_layer(out)
        return out
    
def train_one_fold(model, device, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    start_time = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
            _, pred = out.max(1)
            total += yb.size(0)
            correct += pred.eq(yb).sum().item()
        
        train_acc = correct / total
        train_loss = running_loss / total
        
        print(f"[INFO] Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.4f}")
        
    end_time = time.perf_counter()
    train_time_sec = end_time - start_time
        
    model.eval()
    val_targets = []
    val_preds = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            _, pred = out.max(1)
            val_targets.append(yb.numpy())
            val_preds.append(pred.cpu().numpy())
    
    val_targets = np.concatenate(val_targets, axis=0)
    val_preds = np.concatenate(val_preds, axis=0)
    
    val_acc = accuracy_score(val_targets, val_preds)
    val_preci = precision_score(val_targets, val_preds, average="macro")
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    
    print(f"[INFO] acc={val_acc:.4f}, preci={val_preci:.4f}, f1={val_f1:.4f}")
    print(f"[INFO] train time: {train_time_sec:.2f}sec")
        
    return val_acc, val_preci, val_f1, train_time_sec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--out_csv", type=str, default="resnet_10cv_results.csv")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    data = np.load(args.npz, allow_pickle=True)
    X = data["X_bits"]
    y = data["y"].astype(np.int64)
    print(f"[INFO] Loaded: {args.npz}")
    print(f"[INFO] X shape: {X.shape} | y shape: {y.shape}")
    
    input_dim = X.shape[1]
    print(f"[INFO] Input dimention: {input_dim}")
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print("\n==========")
        print(f" Fold: {fold} / 10")
        print("==========") 
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
        
        model = ResNetBF(
            input_dim = input_dim,
            num_classes=10,
            width=args.width,
            num_blocks=args.blocks
        ).to(device)
        
        val_acc, val_preci, val_f1, train_time_sec = train_one_fold(
            model, device, train_loader, val_loader,
            epochs=args.epochs
        )
        
        result_row = {
            "fold": fold,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "epochs": args.epochs,
            "width": args.width,
            "blocks": args.blocks,
            "val_accuracy": float(val_acc),
            "val_precision": float(val_preci),
            "val_f1": float(val_f1),
            "train_time_sec": float(train_time_sec),
        }
        
        fold_results.append(result_row)
    
    fieldnames = [
        "fold",
        "train_size",
        "val_size",
        "epochs",
        "width",
        "blocks",
        "val_accuracy",
        "val_precision",
        "val_f1",
        "train_time_sec",
    ]
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_results:
            writer.writerow(row)
        mean_acc = np.mean([r["val_accuracy"] for r in fold_results])
        mean_preci = np.mean([r["val_precision"] for r in fold_results])
        mean_f1 = np.mean([r["val_f1"] for r in fold_results])
        mean_time = np.mean([r["train_time_sec"] for r in fold_results])
        
        writer.writerow({
            "fold": "mean",
            "train_size": "",
            "val_size": "",
            "epochs": args.epochs,
            "width": args.width,
            "blocks": args.blocks,
            "val_accuracy": mean_acc,
            "val_precision": mean_preci,
            "val_f1": mean_f1,
            "train_time_sec": mean_time,
        })
    
    print(f"[SAVED] {args.out_csv} is saved.")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
