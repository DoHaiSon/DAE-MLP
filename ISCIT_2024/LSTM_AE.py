import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from preprocess import read_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'ENCODER input dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        # print(f'ENCODER reshaped dim: {x.shape}')
        x, (_, _) = self.rnn1(x)
        # print(f'ENCODER output rnn1 dim: {x.shape}')
        x, (hidden_n, _) = self.rnn2(x)
        # print(f'ENCODER output rnn2 dim: {x.shape}')
        # print(f'ENCODER hidden_n rnn2 dim: {hidden_n.shape}')
        # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, self.embedding_dim)}')
        return hidden_n.reshape((batch_size, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'DECODER input dim: {x.shape}')
        x = x.repeat(1, self.seq_len).reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER repeat dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER reshaped dim: {x.shape}')
        x, (hidden_n, cell_n) = self.rnn1(x)
        # print(f'DECODER output rnn1 dim:/ {x.shape}')
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, batch_size=32):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def detect_anomalies(model, dataloader):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for seqs, in dataloader:
            outputs = model(seqs)
            loss = criterion(outputs, seqs)
            if loss.item() > threshold:  # Đặt ngưỡng phù hợp
                anomalies.append(seqs)
    return anomalies


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
        all_recon_errors = all_recon_errors.numpy().flatten()
    return all_recon_errors

if __name__ == '__main__':
    nor_path = './Dataset/Normal_mixed.csv'
    abnor_path = './Dataset/Abnormal.csv'
    Train_nor, Train_abnor, Test_nor, Test_abnor = read_data(nor_path,abnor_path)
    print(Train_nor.shape[2])

    nor_train_tensor = torch.tensor(Train_nor, dtype=torch.float32)
    abnor_train_tensor = torch.tensor(Train_abnor, dtype=torch.float32)
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


    # vae = Autoencoder()
    vae = RecurrentAutoencoder(seq_len=20,n_features=106,embedding_dim=32,batch_size=512)

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.002)
    mse = nn.MSELoss()

    model = train(vae,nor_train_loader,optimizer,num_epochs=100)
    nor_res = test(model,nor_test_loader)
    abnor_res = test(model,abnor_testloader)
    # print(np.array(nor_res))
    # print(np.array(abnor_res))
   
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=np.array(nor_res)[:5000], label='Normal', kde=False, ax=ax, color='#330C2F')
    sns.histplot(data=np.array(abnor_res)[:5000], label='Abnormal', kde=False, ax=ax, color='#CBF3D2')
    #ax.axvline(0.01, ls='-.', label='Threshold')
    ax.legend(loc='best', fontsize=20)
    ax.set_xlim([0, 0.2])
    fig.tight_layout()
    plt.title('Histogramm')
    plt.show()