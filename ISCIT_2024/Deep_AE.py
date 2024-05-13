import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from preprocess import read_data
import seaborn as sns
import matplotlib.pyplot as plt

class SparseAE(nn.Module):
    def __init__(self, sparsity_param=0.05, sparsity_penalty=1e-3):
        super(SparseAE, self).__init__()
        self.sparsity_param = sparsity_param
        self.sparsity_penalty = sparsity_penalty
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(106, 128),  
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 106),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        self.last_hidden = x  # Save the activation of the last hidden layer for sparsity penalty
        x = self.decoder(x)
        return x

def kl_divergence(rho, rho_hat):
    rho_hat = torch.clamp(rho_hat, min=1e-10, max=1-1e-10)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

def sparse_loss(model, x, target, criterion):
    output = model(x)
    loss = criterion(output, target)
    
    rho_hat = torch.mean(model.last_hidden, dim=0)  # Average activation per neuron in batch
    sparsity_loss = kl_divergence(model.sparsity_param, rho_hat).sum()
    
    total_loss = loss + model.sparsity_penalty * sparsity_loss
    return total_loss


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(106, 128),  
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.LeakyReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 106),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_samples = 0
        for data, in dataloader:
            optimizer.zero_grad()
            recon_batch = model(data)
            total_samples += len(data)
            loss = mse(recon_batch,data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        average_loss = train_loss / total_samples

        print(f'Epoch {epoch+1}, Loss: {average_loss:.5f}')
    return model

def test(model, dataloader):
    model.eval()
    test_loss = 0
    all_recon_errors = []
    with torch.no_grad():
        for data, in dataloader:
            recon = model(data)
            recon_error = torch.mean((recon - data)**2, dim=1)
            all_recon_errors.append(recon_error)
        all_recon_errors = torch.cat(all_recon_errors)
        all_recon_errors = all_recon_errors.numpy()
    return all_recon_errors

if __name__ == '__main__':
    nor_path = './Dataset/Normal_mixed.csv'
    abnor_path = './Dataset/Abnormal.csv'
    Train_nor, Train_abnor, Test_nor, Test_abnor = read_data(nor_path,abnor_path)
    print(Train_nor['data'].shape[1])

    nor_train_tensor = torch.tensor(Train_nor['data'], dtype=torch.float32)
    abnor_train_tensor = torch.tensor(Train_abnor['data'], dtype=torch.float32)
    nor_test_tensor = torch.tensor(Test_nor, dtype=torch.float32)
    abnor_test_tensor = torch.tensor(Test_abnor, dtype=torch.float32)

    nor_train_dataset = TensorDataset(nor_train_tensor)
    abnor_train_dataset = TensorDataset(abnor_train_tensor)
    nor_test_dataset = TensorDataset(nor_test_tensor)
    abnor_test_dataset = TensorDataset(abnor_test_tensor)

    nor_train_loader = DataLoader(nor_train_dataset, batch_size=512, shuffle=True)
    abnor_train_loader = DataLoader(abnor_train_dataset, batch_size=512, shuffle=True)
    nor_test_loader = DataLoader(nor_test_dataset, batch_size=512, shuffle=True)
    abnor_testloader = DataLoader(abnor_test_dataset, batch_size=512, shuffle=True)

    sae = SparseAE()
    vae = Autoencoder()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)
    mse = nn.MSELoss()

    #model = train(vae,nor_train_loader,optimizer,num_epochs=100)
    for epoch in range(100):
        for data, in nor_train_loader:
            optimizer.zero_grad()
            loss = sparse_loss(sae, data, data, mse)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')

    
    nor_res = test(sae,nor_test_loader)
    abnor_res = test(sae,abnor_testloader)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=nor_res, label='Normal', kde=False, ax=ax, color='#330C2F')
    sns.histplot(data=abnor_res, label='Abnormal', kde=False, ax=ax, color='#CBF3D2')
    #ax.axvline(0.01, ls='-.', label='Threshold')
    ax.legend(loc='best', fontsize=20)
    ax.set_xlim([0, 0.2])
    fig.tight_layout()
    plt.title('Histogramm')
    plt.show()