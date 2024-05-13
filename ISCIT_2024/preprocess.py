import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder, StandardScaler

# scaler = MinMaxScaler(feature_range=(0,1))
scaler = StandardScaler()

def one_hot_encode(data):
    unique_labels = [
    ["tcp", "udp", "icmp"],
    ["other", "private", "ecr_i", "urp_i", "urh_i", "red_i", "eco_i", "tim_i", "oth_i", "domain_u", "tftp_u", "ntp_u", "IRC", 
                "X11", "Z39_50", "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "echo", "efs", "exec", 
                "finger", "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784", "http_443", "http_8001", "icmp", "imap4",
                "iso_tsap", "klogin", "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat",
                "nnsp", "nntp", "pm_dump", "pop_2", "pop_3", "printer", "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", 
                "supdup", "systat", "telnet", "time", "uucp", "uucp_path", "vmnet", "whois"],
    ["SF", "S0", "S1", "S2", "S3", "REJ", "RSTOS0", "RSTO", "RSTR", "SH", "RSTRH", "SHR", "OTH"],
    ["Normal","OaU","DoS","DoS_Gas","FoT","BP"]
    ]
    encoded_data = []
    for row in data:
        encoding = []
        for i, column_value in enumerate(row):
            unique_column_values = unique_labels[i]
            encoding.extend([1 if column_value == unique else 0 for unique in unique_column_values])
        encoded_data.append(encoding)
    return np.array(encoded_data)

def preprocess(df, is_fit=True):

    label = df['label'].map(lambda x: 'Abnormal' if x != 'Normal' else x)

    df = df.drop(["land", "wrong_fragment",  "urgent", "rerror_rate",  "srv_rerror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"], axis=1)

    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         df[col] = LabelEncoder().fit_transform(df[col])

    numerical_data = df.select_dtypes(exclude='object').values
    categorical_data = df.select_dtypes(include='object').values

    categorical_data = one_hot_encode(categorical_data)

    data = np.concatenate([numerical_data, categorical_data], axis=1)
    if is_fit:
        scaler.fit(data)

    data = scaler.transform(data)

    return dict(data=data, label=label)

def read_data(nor_path, abnor_path):
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    Nor_df = pd.read_csv(nor_path, header=None,names= col_names, nrows= 100000)
    Abnor_df = pd.read_csv(abnor_path, header=None,names= col_names, nrows= 150000)
    print(Abnor_df['label'].value_counts())
    print(f'Number of abnormal: {len(Abnor_df)}')


    Train_nor, Test_nor = train_test_split(Nor_df, test_size=0.2, random_state=42)
    Train_abnor, Test_abnor = train_test_split(Abnor_df, test_size=0.2, random_state=42) 
    print('Class in test abnor:')
    print(Test_abnor['label'].value_counts())
    Train_abnor = Train_abnor[(Train_abnor['label'] == 'DoS') | (Train_abnor['label'] == 'BP') | (Train_abnor['label'] == 'DoS_Gas') | (Train_abnor['label'] == 'OaU')]
    print('Class in train abnor:')
    print(Train_abnor['label'].value_counts())

    # Train = pd.concat([Train_nor, Train_abnor], ignore_index=True)
    Test = pd.concat([Test_nor, Test_abnor], ignore_index=True)
    Train_nor = preprocess(Train_nor, True)
    Train_abnor = preprocess(Train_abnor, False)
    test = preprocess(Test, False)
    Test_nor = test['data'][test['label'] == 'Normal']
    Test_abnor = test['data'][test['label'] == 'Abnormal']

    return Train_nor, Train_abnor, Test_nor, Test_abnor, test
