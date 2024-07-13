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
from Models.DAE_MLP import Autoencoder, threshold_loop, test

def testAcc(model, dataloader, threshold, w1, w2, acc=True):
    abnor_list = []
    acc_total = 0
    recall_total = 0
    precision_total = 0
    subclass_metrics_total = {}
    model.eval()
    
    with torch.no_grad(): 
        for x_batch, y_batch in dataloader:
            # Predict the anomaly scores
            anomaly_scores = model.predict_class(x_batch, threshold, w1, w2)
            abnor_list.append(anomaly_scores)
            
            # Binary classification
            binary_predictions = torch.where(anomaly_scores <= threshold, 0, 1)
            
            # Transform class to binary
            binary_labels = torch.where(y_batch > 0, 1, 0)
            
            if acc:
                acc = accuracy_score(binary_labels.numpy(), binary_predictions.numpy())
                recall = recall_score(binary_labels.numpy(), binary_predictions.numpy(), average='macro')
                precision = precision_score(binary_labels.numpy(), binary_predictions.numpy(), average='macro')
                acc_total += acc
                recall_total += recall
                precision_total += precision
            
            # Metrics for each subclass
            subclasses = y_batch.unique()
            for subclass in subclasses:
                if subclass.item() > 0: 
                    #print(torch.logical_or(y_batch == subclass, y_batch == 0))
                    subclass_indices = torch.logical_or(y_batch == subclass, y_batch == 0).nonzero(as_tuple=True)[0] 

                    subclass_true = binary_labels[subclass_indices]
                    subclass_pred = binary_predictions[subclass_indices]
                    
                    # Classification report for subclases
                    report = classification_report(subclass_true.numpy(), subclass_pred.numpy(), output_dict=True, digits=5)
                    print(classification_report(subclass_true.numpy(), subclass_pred.numpy(), digits=5))
                    
                    if subclass.item() not in subclass_metrics_total:
                        subclass_metrics_total[subclass.item()] = {
                            'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'batch_count': 0
                        }
                    
                    subclass_metrics_total[subclass.item()]['macro avg']['precision'] += report['macro avg']['precision']
                    subclass_metrics_total[subclass.item()]['macro avg']['recall'] += report['macro avg']['recall']
                    subclass_metrics_total[subclass.item()]['macro avg']['f1-score'] += report['macro avg']['f1-score']
                    subclass_metrics_total[subclass.item()]['macro avg']['support'] += report['macro avg']['support']
                    
                    subclass_metrics_total[subclass.item()]['weighted avg']['precision'] += report['weighted avg']['precision']
                    subclass_metrics_total[subclass.item()]['weighted avg']['recall'] += report['weighted avg']['recall']
                    subclass_metrics_total[subclass.item()]['weighted avg']['f1-score'] += report['weighted avg']['f1-score']
                    subclass_metrics_total[subclass.item()]['weighted avg']['support'] += report['weighted avg']['support']
                    
                    subclass_metrics_total[subclass.item()]['batch_count'] += 1

    for subclass in subclass_metrics_total:
        count = subclass_metrics_total[subclass]['batch_count']
        subclass_metrics_total[subclass]['macro avg']['precision'] /= count
        subclass_metrics_total[subclass]['macro avg']['recall'] /= count
        subclass_metrics_total[subclass]['macro avg']['f1-score'] /= count
        
        subclass_metrics_total[subclass]['weighted avg']['precision'] /= count
        subclass_metrics_total[subclass]['weighted avg']['recall'] /= count
        subclass_metrics_total[subclass]['weighted avg']['f1-score'] /= count

    if acc:
        num_batches = len(dataloader)
        return acc_total / num_batches, recall_total / num_batches, precision_total / num_batches, subclass_metrics_total
    else:
        return subclass_metrics_total
    
def plot_distribution(nor, abnor):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=nor.numpy(), label='Normal', kde=False, ax=ax, color='#330C2F')
    sns.histplot(data=abnor.numpy(), label='Abnormal', kde=False, ax=ax, color='#CBF3D2')
    #ax.axvline(0.01, ls='-.', label='Threshold')
    ax.legend(loc='best', fontsize=20)
    ax.set_xlim([0, 0.8])
    fig.tight_layout()
    plt.title('Histogramm')
    plt.show()

if __name__ == '__main__':
    nor_path = './Dataset/Normal_mixed.csv'
    abnor_path = './Dataset/Abnormal.csv'
    BATCHSIZE = 512
    learning_rate = 0.001

    """
    The code below can be varied since the current version is used to evaluate our model in the paper
    """
    # ------------------------------------- Start of data preprocessing -------------------------------------

    Train_nor, Train_abnor, Test_nor, Test_abnor, Test = read_data(nor_path,abnor_path)
    label = pd.concat([Train_nor['label'],Train_abnor['label']], axis=0, ignore_index=True)

    nor_train_tensor = torch.tensor(Train_nor['data'], dtype=torch.float32)
    abnor_train_tensor = torch.tensor(Train_abnor['data'], dtype=torch.float32)
    #label_tensor = torch.tensor(pd.Categorical(label).codes.astype('float32'), dtype=torch.float32)
    label_tensor = torch.tensor(label.replace({'Normal': 0, 'Abnormal': 1}).values, dtype=torch.float32)
    train_tensor = torch.concat([nor_train_tensor,abnor_train_tensor],dim=0)

    nor_test_tensor = torch.tensor(Test_nor, dtype=torch.float32)
    abnor_test_tensor = torch.tensor(Test_abnor, dtype=torch.float32)
    test_tensor = torch.tensor(Test['data'], dtype=torch.float32)
    test_label_tensor = torch.tensor(Test['label'].replace({'Normal': 0, 'Abnormal': 1}).values, dtype=torch.float32)
    # print(Test['label'].values)
    # Test['label'] = Test['label'].replace({'Normal': 0, 'BP': 1, 'DoS': 2, 'DoS_Gas': 3, 'FoT': 4, 'OaU': 5})
    # Test['label'] = Test['label'].astype(int)
    # test_label = torch.tensor(Test['label'].values, dtype=torch.float32)
    #test_class_label = torch.tensor(Test_class['label'].replace({'Normal': 0, 'BP': 1, 'DoS': 2, 'DoS_Gas': 3, 'FoT': 4}).values, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor, label_tensor)
    nor_test_dataset = TensorDataset(nor_test_tensor)
    abnor_test_dataset = TensorDataset(abnor_test_tensor)
    #test_dataset = TensorDataset(test_tensor,test_label)
    test_class_dataset = TensorDataset(test_tensor, test_label_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    nor_test_loader = DataLoader(nor_test_dataset, batch_size=BATCHSIZE, shuffle=True)
    abnor_testloader = DataLoader(abnor_test_dataset, batch_size=BATCHSIZE, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    class_loader = DataLoader(test_class_dataset, batch_size=BATCHSIZE, shuffle=True)
    # --------------------------------------------------------------------------- -------------------------------------


    model = Autoencoder(input_dim=Train_nor['data'].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load('./model/test.pth'))
    nor = test(model,nor_test_loader,threshold=1, w1=0.5, w2=0.5,acc=False)
    abnor = test(model,abnor_testloader,threshold=1, w1=0.5, w2=0.5, acc = False)

    alpha = 0.5
    sigma = 5
    mean_threshold = torch.concat([nor,abnor]).mean() * alpha
    sigma_threshold = np.percentile(nor.numpy(), 100 - sigma)

    loop_thresh = threshold_loop(threshold=sigma_threshold, best_threshold=sigma_threshold, model=model, data=class_loader)
    acc, recall, precision,_ = test(model,class_loader,threshold=0.45,w1=0.5, w2=0.5, acc = True)
    #acc, recall, precision, per_class_metrics = testAcc(model,test_loader,threshold=sigma_threshold,w1=0.5, w2=0.5,acc=True)

    print('Accuracy full class:',acc)
    print('Recall full class:',recall)
    print('Precision full class:',precision)
    #print(per_class_metrics)
    #plot_distribution(nor,abnor)