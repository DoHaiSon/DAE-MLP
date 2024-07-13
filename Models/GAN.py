import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from Utils.preprocess_main import read_data
from sklearn.preprocessing import LabelEncoder

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, dataloader, num_epochs=100, lr_g=0.0002, lr_d=0.00005, latent_dim=100):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d)
    
    for epoch in range(num_epochs):
        for real_samples in dataloader:
            real_samples = real_samples[0]  # Lấy dữ liệu từ TensorDataset
            batch_size = real_samples.size(0)
            
            # Labels for real and fake data
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            # Train discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(real_samples)
            loss_d_real = criterion(outputs, real_labels)
            z = torch.randn(batch_size, latent_dim)
            fake_samples = generator(z)
            outputs = discriminator(fake_samples.detach())
            loss_d_fake = criterion(outputs, fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_samples)
            loss_g = criterion(outputs, real_labels)
            loss_g.backward()
            optimizer_g.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")


def detect_anomalies(generator, X_test, y_test, latent_dim=100, threshold=0.1):
    generator.eval()
    
    # Generate reconstructions
    z = torch.randn(X_test.size(0), latent_dim)
    with torch.no_grad():
        generated_samples = generator(z)
    
    # Calculate reconstruction error
    reconstruction_error = torch.mean((X_test - generated_samples) ** 2, dim=1)
    
    # Classify as anomaly if error exceeds threshold
    anomalies = (reconstruction_error > threshold).cpu().numpy()
    print(anomalies)
    
    # Convert numpy boolean array to integer array (0 or 1)
    anomalies = anomalies.astype(int)
    
    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(y_test, anomalies)
    precision = precision_score(y_test, anomalies)
    recall = recall_score(y_test, anomalies)
    
    return accuracy, precision, recall

if __name__ == '__main__':
    nor_path = './Dataset/Normal_mixed.csv'
    abnor_path = './Dataset/Abnormal.csv'
    Train_nor, Train_abnor, Test_nor, Test_abnor, Test = read_data(nor_path, abnor_path)
    print(Train_nor['data'].shape[1])
    label = pd.concat([pd.Series(Train_nor['label']), pd.Series(Train_abnor['label'])], axis=0, ignore_index=True)
    X_train = Train_nor['data']
    y_train = np.array(label)
    X_test = Test['data']
    y_test = np.array(Test['label'])
    #X_train, y_train = shuffle(X_train, y_train, random_state=42)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    batch_size = 64
    train_data = TensorDataset(X_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    latent_dim = 100
    generator = Generator(latent_dim, X_train.size(1))
    discriminator = Discriminator(X_train.size(1))

    train_gan(generator, discriminator, train_loader, num_epochs=100, latent_dim=latent_dim)

    accuracy, precision, recall = detect_anomalies(generator, X_test, y_test, latent_dim=latent_dim, threshold=0.5)

    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")