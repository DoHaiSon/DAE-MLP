import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC, OneClassSVM
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from Utils.preprocess_main import read_data
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall

if __name__ == '__main__':
    nor_path = './Dataset/Normal_mixed.csv'
    abnor_path = './Dataset/Abnormal.csv'
    Train_nor, Train_abnor, Test_nor, Test_abnor, Test = read_data(nor_path,abnor_path)
    print(Train_nor['data'].shape[1])
    label = pd.concat([Train_nor['label'],Train_abnor['label']], axis=0, ignore_index=True)
    X_train = np.concatenate([Train_nor['data'],Train_abnor['data']], axis=0)
    y_train = np.array(label)
    X_test = Test['data']
    y_test = np.array(Test['label'])
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)


    models = {
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "One-Class SVM": OneClassSVM(gamma='auto'),
        "KMeans": KMeans(n_clusters=2),
        "Isolation Forest": IsolationForest(),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "PCA": PCA(n_components=2),
        "Gaussian Mixture": GaussianMixture(n_components=2),
        "Local Outlier Factor": LocalOutlierFactor(novelty=True)
    }

    results = {}
    for name, model in models.items():
        if name == "One-Class SVM":
            model.fit(X_train[y_train == 1])
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 1, 0)
        elif name == "KMeans":
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 0, 0, 1)
        elif name == "Isolation Forest":
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
        elif name == "DBSCAN":
            y_pred = model.fit_predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)
        elif name == "PCA":
            model.fit(X_train)
            X_test_pca = model.transform(X_test)
            # Giả định rằng PCA là để giảm chiều và không thể hiện tốt trong phân loại nhị phân
            # Cần có một bước phân loại bổ sung ở đây sau khi giảm chiều bằng PCA
            y_pred = KMeans(n_clusters=2).fit_predict(X_test_pca)
            y_pred = np.where(y_pred == 0, 0, 1)
        elif name == "Gaussian Mixture":
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 0, 0, 1)
        elif name == "Local Outlier Factor":
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred, zero_division=1)
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall
        }

    # In kết quả
    for name, metrics in results.items():
        print(f"Model: {name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        print("-" * 30)
    
