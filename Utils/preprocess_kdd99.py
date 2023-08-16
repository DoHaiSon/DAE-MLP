import pandas as pd
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
import pandas as pd
from sklearn.model_selection import train_test_split

# filename = './Dataset/kddcup.data_10_percent_corrected.csv'

# col_names =["duration","protocol_type","service","flag","src_bytes",
#     "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
#     "logged_in","num_compromised","root_shell","su_attempted","num_root",
#     "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
#     "is_host_login","is_guest_login","count","srv_count","serror_rate",
#     "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
#     "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
#     "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
#     "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
#     "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# dataset = pd.read_csv(filename, header = None, names = col_names)

extracted_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# extracted = dataset[extracted_features]

# extracted = extracted.replace('normal.', 'Normal')

# extracted.to_csv('./Dataset/kdd99_extracted.csv', index=False)

filename = './Dataset/kdd99_extracted.csv'

dataset = pd.read_csv(filename, header = None, names = extracted_features)

train, test = train_test_split(dataset, test_size=0.3, random_state=1)

train.to_csv('./Dataset/kdd99_train.csv', index=False)
test.to_csv('./Dataset/kdd99_test.csv', index=False)