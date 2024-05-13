import torch
from torch import nn
from torch.nn import functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, num_classes):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)  # Mean of z
        self.fc22 = nn.Linear(hidden_dim, z_dim)  # Log variance of z

        # Decoder layers
        self.fc3 = nn.Linear(z_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        cond = torch.cat([x, y], dim=1)
        h1 = F.relu(self.fc1(cond))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        cond = torch.cat([z, y], dim=1)
        h3 = F.relu(self.fc3(cond))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, y), mu, log_var

    def generate(self, z, y):
        return self.decode(z, y)

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
