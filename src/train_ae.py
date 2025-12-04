import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

def train_ae(
    df_train, 
    df_test, 
    target_col='Class',
    latent_dim=8,
    batch_size=64,
    n_epochs=50,
    lr=1e-3,
    save_model_path='checkpoint/ae_model.pt',
    save_error_path='checkpoint/recon_error_test.npy'
):
    # ============================
    # 1️⃣ Chỉ dùng NORMAL (class 0) để train
    # ============================
    df_train_normal = df_train[df_train[target_col] == 0]

    X_train = df_train_normal.drop(columns=[target_col]).values.astype(np.float32)
    X_test  = df_test.drop(columns=[target_col]).values.astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train)),
        batch_size=batch_size,
        shuffle=True
    )
    
    # ============================
    # 2️⃣ Khởi tạo model
    # ============================
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ============================
    # 3️⃣ Training loop
    # ============================
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(X_train):.6f}")
    
    torch.save(model.state_dict(), save_model_path)
    print(f"AE model saved to {save_model_path}")

    # ============================
    # 4️⃣ Reconstruction error cho test set
    # ============================
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test)
        X_test_hat = model(X_test_tensor)
        recon_error = torch.mean((X_test_tensor - X_test_hat)**2, dim=1).numpy()
    
    np.save(save_error_path, recon_error)
    print(f"Reconstruction error saved to {save_error_path}")

    return model, recon_error
