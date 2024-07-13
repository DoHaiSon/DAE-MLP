import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from Utils.preprocess_main import read_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(42)  
np.random.seed(42)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder definition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Decoder definition
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
        
        # Classifier definition
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        code = self.encoder(x)
        reconstruction = self.decoder(code)
        classification = self.classifier(x)
        return {'decoder_output': reconstruction, 'classifier_output': classification}

    def get_reconstruction_error(self, x):
        r = self(x)
        return F.mse_loss(x, r['decoder_output'], reduction='none').mean(1)
    
    def get_classifier_prob(self, x):
        pr = self(x)
        return pr['classifier_output'].squeeze()

    def predict_class(self, x, threshold, w1, w2):
        reconstruction_error = self.get_reconstruction_error(x)
        prob = self.get_classifier_prob(x)
        anomaly_score = w1 * reconstruction_error + w2 * prob
        return anomaly_score
        #return torch.where(anomaly_score <= threshold, 'Normal', 'Abnormal')


def combined_loss(outputs, labels, x, alpha=0.5):
    reconstruction_loss = F.mse_loss(outputs['decoder_output'], x)
    classification_loss = F.binary_cross_entropy(outputs['classifier_output'].squeeze(), labels)
    return alpha * reconstruction_loss + (1 - alpha) * classification_loss, reconstruction_loss, classification_loss
    
def train(model, dataloader, optimizer, num_epochs=10, alpha=0.5):
    model.train() 
    for epoch in range(num_epochs):
        total_loss = 0
        total_res = 0
        total_clf = 0
        for x, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss, res, clf = combined_loss(outputs, labels, x, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_res += res.item()
            total_clf += clf.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}, Res_Loss: {total_res/len(dataloader)}, Clf_Loss: {total_clf/len(dataloader)}')

def test(model, dataloader, threshold, w1, w2, acc=True):
    abnor_list = []
    acc_total = 0
    recall_total = 0
    precision_total = 0
    model.eval()  
    if acc:
        with torch.no_grad(): 
            for x_batch,y_batch in dataloader:
                anomaly_scores = model.predict_class(x_batch, threshold, w1, w2)
                abnor_list.append(anomaly_scores)
                predictions = torch.where(anomaly_scores <= threshold, 0, 1)
                acc = accuracy_score(y_batch.numpy(), predictions.numpy())
                recall = recall_score(y_batch.numpy(), predictions.numpy(),average='macro')
                precision = precision_score(y_batch.numpy(), predictions.numpy(), average='macro')
                acc_total += acc
                recall_total += recall
                precision_total += precision
            anomaly_scores_tensor = torch.cat(abnor_list, dim=0)
            return acc_total/len(dataloader), recall_total/len(dataloader), precision_total/len(dataloader), anomaly_scores_tensor
    else:
        with torch.no_grad(): 
            for x_batch, in dataloader:
                anomaly_scores = model.predict_class(x_batch, threshold, w1, w2)
                abnor_list.append(anomaly_scores)
            anomaly_scores_tensor = torch.cat(abnor_list, dim=0)
        return anomaly_scores_tensor

def threshold_loop(threshold, best_threshold, model, data):
    step = 0.001                     # Initial step
    decay = 0.5                     # Decay rate
    num_decay = 30                   # Number of decay times
    pre = 0                         # Previous accuracy
    cur = 1e-9                      # Current accuracy
    best_acc = 1                    # Initial best accuracy
    occ = 10                        # Occurence of the previous accuracy better than the current one
    count = 0                       # Counter

    for d_i in range (num_decay):
        for i in range (1000):
            pre = cur
            acc,_,_,_ = test(model,data,threshold,0.5,0.5,acc=True)
            #acc  = accuracy_score(label, pred)
            threshold = threshold + step
            cur = acc
            print("Accuracy:", acc, "\nThreshold:", threshold)

            # If the previous accuracy is better than the current one. Plus the counter and store the best threshold and accuracy
            if (pre > cur):
                count = count + 1
                cur = pre
                best_threshold = threshold - count*step
                best_acc = pre
            else:
                count = 0
                best_threshold = threshold
                continue

            if count == occ + 1:
                step = step * decay
                threshold = best_threshold
                count = 0
                cur = best_acc
                print("--------------------------------------------------")
                print("Best accuracy:", best_acc, "\nFinal threshold:", best_threshold)
                break  
    return best_threshold  


if __name__ == '__main__':
    nor_path = './Dataset/Normal_mixed.csv'
    abnor_path = './Dataset/Abnormal.csv'
    BATCHSIZE = 512
    learning_rate = 0.001
    epochs = 100
    alpha_ = 0.5

    Train_nor, Train_abnor, Test_nor, Test_abnor, Test = read_data(nor_path,abnor_path)
    label = pd.concat([Train_nor['label'],Train_abnor['label']], axis=0, ignore_index=True)

    nor_train_tensor = torch.tensor(Train_nor['data'], dtype=torch.float32)
    abnor_train_tensor = torch.tensor(Train_abnor['data'], dtype=torch.float32)
    #label_tensor = torch.tensor(pd.Categorical(label).codes.astype('float32'), dtype=torch.float32)
    label_tensor = torch.tensor(label.replace({'Normal': 0, 'Abnormal': 1}).values, dtype=torch.float32)
    train_tensor = torch.concat([nor_train_tensor,abnor_train_tensor],dim=0)

    train_dataset = TensorDataset(train_tensor, label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)

    model = Autoencoder(input_dim=Train_nor['data'].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, optimizer, num_epochs=epochs, alpha=alpha_)
    torch.save(model.state_dict(),'./model/test.pth')
