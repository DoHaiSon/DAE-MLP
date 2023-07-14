from sklearn.utils import shuffle
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

# Define variables
attack_ip   = '10.10.117.253'
host_ip     = '10.10.117.252'
ddos_range  = '192.168.4' 
ddos_port   = '8545'
local_range = '10.10.117'
attack_sv   = 'http'

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "src_ip", "src_port", "dst_ip", 
                "dst_port", "time"]

nor = pd.read_csv("/home/avitech7/haison98/RIVF/Dataset/Normal.csv", header=None, names = col_names)
brute_pass = pd.read_csv("/home/avitech7/haison98/RIVF/Dataset/Attack/BP.csv", header=None, names = col_names)
dos = pd.read_csv("/home/avitech7/haison98/RIVF/Dataset/Attack/DoS.csv", header=None, names = col_names)
dos_tran = pd.read_csv("/home/avitech7/haison98/RIVF/Dataset/Attack/FoT.csv", header=None, names = col_names)
DoS_gas = pd.read_csv("/home/avitech7/haison98/RIVF/Dataset/Attack/DoS_GasLimit.csv", header=None, names = col_names)
OaU = pd.read_csv("/home/avitech7/haison98/RIVF/Dataset/Attack/OaU.csv", header=None, names = col_names)

# Pre-processing normal state dataset
# Normal state:
# src                             = nor['src_ip'].str.contains(local_range)
# dst                             = nor['dst_ip'].str.contains(local_range)
# local                           = src & dst
# local_i                         = np.where(local == False)
# local_nor                       = nor.drop(nor.index[local_i[0]])
local_nor = nor

# Brute password state:
# src                             = brute_pass['src_ip'].str.contains(local_range)
# dst                             = brute_pass['dst_ip'].str.contains(local_range)
# local                           = src & dst
# local_i                         = np.where(local == False)
# local_brute                     = brute_pass.drop(brute_pass.index[local_i[0]])

src                             = brute_pass['src_ip'].str.contains(attack_ip)
dst                             = brute_pass['dst_ip'].str.contains(attack_ip)
service                         = brute_pass['service'].str.contains(attack_sv)
local                           = (src | dst) & service

local_i                         = np.where(local == True)
local_brute_nor                 = brute_pass.drop(brute_pass.index[local_i[0]])

local_i                         = np.where(local == False)
local_brute                     = brute_pass.drop(brute_pass.index[local_i[0]])

local_brute['label']            = 'BP'
local_brute_label               = local_brute.drop(columns = ["src_ip", "src_port", "dst_ip", "dst_port", "time"])
local_brute_label['label']      = 'BP'
# local_brute.to_csv("datasets/pre/brute_filter.csv", index = False, header=None)
# local_brute_label.to_csv("datasets/pre/brute_label.csv", index = False, header=None)

# DoS state:
src                             = np.invert( dos['src_ip'].str.contains(local_range) ) | (dos['src_ip'].str.contains(host_ip) & dos['src_port'].astype(str).str.contains(ddos_port))
dst                             = (dos['dst_ip'].str.contains(host_ip) & dos['dst_port'].astype(str).str.contains(ddos_port)) | np.invert( dos['dst_ip'].str.contains(local_range) )
local                           = src & dst
# local_i                         = np.where(local == False)
# local_dos                       = dos.drop(dos.index[local_i[0]])

# src                             = local_dos['src_ip'].str.contains(ddos_range)
# dst                             = local_dos['dst_ip'].str.contains(ddos_range)
# local                           = src | dst

local_i                         = np.where(local == True)
local_dos_nor                   = dos.drop(dos.index[local_i[0]])

local_i                         = np.where(local == False)
local_dos                       = dos.drop(dos.index[local_i[0]])

local_dos['label']              = 'DoS'
local_dos_label                 = local_dos.drop(columns = ["src_ip", "src_port", "dst_ip", "dst_port", "time"])
local_dos_label['label']        = 'DoS'
# local_dos_nor.to_csv("dos_normal.csv", index = False, header=None)
# local_dos_label.to_csv("dos_label.csv", index = False, header=None)

# DoS by flood transactions state:
# src                             = dos_tran['src_ip'].str.contains(local_range)
# dst                             = dos_tran['dst_ip'].str.contains(local_range)
# local                           = src & dst
# local_i                         = np.where(local == False)
# local_dos_tran                  = dos_tran.drop(dos_tran.index[local_i[0]])

src                             = dos_tran['src_ip'].str.contains(attack_ip)
dst                             = dos_tran['dst_ip'].str.contains(attack_ip)
service                         = dos_tran['service'].str.contains(attack_sv)
local                           = (src | dst) & service

local_i                         = np.where(local == True)
local_dos_tran_nor              = dos_tran.drop(dos_tran.index[local_i[0]])

local_i                         = np.where(local == False)
local_dos_tran                  = dos_tran.drop(dos_tran.index[local_i[0]])

local_dos_tran['label']         = 'FoT'
local_dos_tran_label            = local_dos_tran.drop(columns = ["src_ip", "src_port", "dst_ip", "dst_port", "time"])
local_dos_tran_label['label']   = 'FoT'
# local_dos_tran.to_csv("datasets/pre/dos_tran_filter.csv", index = False, header=None)
# local_dos_tran_label.to_csv("datasets/pre/dos_tran_label.csv", index = False, header=None)

# DoS_gas attack state
src                             = DoS_gas['src_ip'].str.contains(attack_ip)
dst                             = DoS_gas['dst_ip'].str.contains(attack_ip)
service                         = DoS_gas['service'].str.contains(attack_sv)
local                           = (src | dst) & service

local_i                         = np.where(local == True)
local_dos_gas_nor               = DoS_gas.drop(DoS_gas.index[local_i[0]])

local_i                         = np.where(local == False)
local_dos_gas                   = DoS_gas.drop(DoS_gas.index[local_i[0]])

local_dos_gas['label']          = 'DoS_Gas'
local_dos_gas_label             = local_dos_gas.drop(columns = ["src_ip", "src_port", "dst_ip", "dst_port", "time"])
local_dos_gas_label['label']    = 'DoS_Gas'

# DoS_gas attack state
src                             = OaU['src_ip'].str.contains(attack_ip)
dst                             = OaU['dst_ip'].str.contains(attack_ip)
service                         = OaU['service'].str.contains(attack_sv)
local                           = (src | dst) & service

local_i                         = np.where(local == True)
local_OaU_nor                   = OaU.drop(OaU.index[local_i[0]])

local_i                         = np.where(local == False)
local_OaU                       = OaU.drop(OaU.index[local_i[0]])

local_OaU['label']              = 'OaU'
local_OaU_label                 = local_OaU.drop(columns = ["src_ip", "src_port", "dst_ip", "dst_port", "time"])
local_OaU_label['label']        = 'OaU'

# local_MitM_label.to_csv("datasets/032423/MitM_w3_pre.csv", index = False, header=None)
print(len(local_nor.index), len(local_brute_nor.index), len(local_dos_nor.index), len(local_dos_tran_nor.index), 
      len(local_dos_gas_nor.index), len(local_OaU_nor.index))
# Merge normal packets from attack state to normal state
local_nor                       = local_nor.sample(n = 100000)
local_brute_nor                 = local_brute_nor.sample(n = 100000)
local_dos_nor                   = local_dos_nor.sample(n = 100000)
local_dos_tran_nor              = local_dos_tran_nor.sample(n = 100000)
local_dos_gas_nor               = local_dos_gas_nor.sample(n = 100000)
local_OaU_nor                   = local_OaU_nor.sample(n = 100000)
local_nor                       = [local_nor, local_brute_nor, local_dos_nor, local_dos_tran_nor, local_dos_gas_nor, local_OaU_nor]
local_nor                       = pd.concat(local_nor)
local_nor['label']              = 'Normal'
local_nor_label                 = local_nor.drop(columns = ["src_ip", "src_port", "dst_ip", "dst_port", "time"])
local_nor_label['label']        = 'Normal'
local_nor_label                 = shuffle(local_nor_label)
# local_nor.to_csv("datasets/pre/normal_filter.csv", index = False, header=None)
local_nor_label.to_csv("/home/avitech7/haison98/RIVF/Dataset/Normal_mixed.csv", index = False, header=None)

print("Normal state: ", len(local_nor_label.index), "samples.")
print("Brute password state: ", len(local_brute_label.index), "samples.")
print("DoS state: ", len(local_dos_label.index), "samples.")
print("DoS Transactions state: ", len(local_dos_tran_label.index), "samples.")
print("local_dos_gas_label state: ", len(local_dos_gas_label.index), "samples.")
print("local_dos_gas_label state: ", len(local_dos_gas_label.index), "samples.")

# Nomial the dataset and save into a file: dataset_nomial.csv
# nomial2visualize(dataset)

# # # Random sample from dataset
seed              = np.random.randint(1e6)
# normal_label      = local_nor_label.sample(n=100000, random_state=seed)
brute_pass_label  = local_brute_label.sample(n=25293, random_state=seed)
dos_label         = local_dos_label.sample(n=100000, random_state=seed)
dos_tran_label    = local_dos_tran_label.sample(n=100000, random_state=seed)
local_dos_gas_label = local_dos_gas_label.sample(n=91128, random_state=seed)
local_OaU_label   = local_OaU_label.sample(n=50999, random_state=seed)

# Merge into a dataset
dataset = [brute_pass_label, dos_label, dos_tran_label, local_dos_gas_label, local_OaU_label]
dataset = pd.concat(dataset)
dataset = shuffle(dataset)
dataset.to_csv("/home/avitech7/haison98/RIVF/Dataset/Abnormal.csv", index = False, header=None)

# normal            = local_nor.sample(n=30000, random_state=seed)
# brute_pass        = local_brute.sample(n=3000, random_state=seed)
# dos               = local_dos.sample(n=3000, random_state=seed)
# dos_tran          = local_dos_tran.sample(n=3000, random_state=seed)

# # Save all into separated files
# normal_label.to_csv("datasets/sample/normal_label_sample.csv", index = False, header=None)
# brute_pass_label.to_csv("C:/Users/SON/Desktop/CIB/datasets/dataset/W1_BP.csv", index = False, header=None)
# dos_label.to_csv("C:/Users/SON/Desktop/CIB/datasets/dataset/W1_DoS.csv", index = False, header=None)
# dos_tran_label.to_csv("C:/Users/SON/Desktop/CIB/datasets/dataset/W1_FoT.csv", index = False, header=None)
# MitM_label.to_csv("C:/Users/SON/Desktop/CIB/datasets/dataset/W1_MitM.csv", index = False, header=None)
# normal.to_csv("datasets/sample/normal_sample.csv", index = False, header=None)
# brute_pass.to_csv("datasets/sample/brute_pass_sample.csv", index = False, header=None)
# dos.to_csv("datasets/sample/dos_sample.csv", index = False, header=None)
# dos_tran.to_csv("datasets/sample/dos_tran_sample.csv", index = False, header=None)

# # Train/test split for DL
# dataset_label = [normal_label, brute_pass_label, dos_label, dos_tran_label]
# dataset_label = pd.concat(dataset_label)
# dataset_label = shuffle(dataset_label, random_state=seed)
# train, test   = train_test_split(dataset_label, test_size=0.3, random_state=seed)
# train.to_csv("datasets/final/train.csv", index = False, header=None)
# test.to_csv("datasets/final/test.csv", index = False, header=None)

# dataset       = [normal, brute_pass, dos, dos_tran]
# dataset       = pd.concat(dataset)
# dataset       = shuffle(dataset, random_state=seed)
# train, test   = train_test_split(dataset, test_size=0.3, random_state=seed)
# train.to_csv("datasets/final/train_raw.csv", index = False, header=None)
# test.to_csv("datasets/final/test_raw.csv", index = False, header=None)

# # Train/test split for FL 2 workers
# w1_label, w2_label = train_test_split(dataset_label, test_size=0.5, random_state=seed)
# w1_label.to_csv("datasets/train_2_w1.csv", index = False, header=None)
# w2_label.to_csv("datasets/train_2_w2.csv", index = False, header=None)

# w1, w2 = train_test_split(dataset, test_size=0.5, random_state=seed)
# w1.to_csv("datasets/train_2_w1_raw.csv", index = False, header=None)
# w2.to_csv("datasets/train_2_w2_raw.csv", index = False, header=None)

# # Train/test split for FL 3 workers
# w12_label, w3_label = train_test_split(dataset_label, train_size=0.667, random_state=seed)
# w1_label, w2_label  = train_test_split(w12_label,     train_size=0.5,   random_state=seed)
# w1_label.to_csv("datasets/train_3_w1.csv", index = False, header=None)
# w2_label.to_csv("datasets/train_3_w2.csv", index = False, header=None)
# w3_label.to_csv("datasets/train_3_w3.csv", index = False, header=None)

# w12, w3 = train_test_split(dataset, train_size=0.667, random_state=seed)
# w1, w2  = train_test_split(w12,     train_size=0.5,   random_state=seed)
# w1.to_csv("datasets/train_3_w1_raw.csv", index = False, header=None)
# w2.to_csv("datasets/train_3_w2_raw.csv", index = False, header=None)
# w3.to_csv("datasets/train_3_w3_raw.csv", index = False, header=None)

# # Train/test split for FL 4 workers
# w12_label, w34_label = train_test_split(dataset_label, train_size=0.5, random_state=seed)
# w1_label, w2_label   = train_test_split(w12_label,     train_size=0.5, random_state=seed)
# w3_label, w4_label   = train_test_split(w34_label,     train_size=0.5, random_state=seed)
# w1_label.to_csv("datasets/train_4_w1.csv", index = False, header=None)
# w2_label.to_csv("datasets/train_4_w2.csv", index = False, header=None)
# w3_label.to_csv("datasets/train_4_w3.csv", index = False, header=None)
# w4_label.to_csv("datasets/train_4_w4.csv", index = False, header=None)

# w12, w34 = train_test_split(dataset, train_size=0.5, random_state=seed)
# w1, w2   = train_test_split(w12,     train_size=0.5, random_state=seed)
# w3, w4   = train_test_split(w34,     train_size=0.5, random_state=seed)
# w1.to_csv("datasets/train_4_w1_raw.csv", index = False, header=None)
# w2.to_csv("datasets/train_4_w2_raw.csv", index = False, header=None)
# w3.to_csv("datasets/train_4_w3_raw.csv", index = False, header=None)
# w4.to_csv("datasets/train_4_w4_raw.csv", index = False, header=None)

# # Train/test split for FL 5 workers
# w1234_label, w5_label  = train_test_split(dataset_label, train_size=0.8, random_state=seed)
# w12_label,   w34_label = train_test_split(w1234_label,   train_size=0.5, random_state=seed)
# w1_label, w2_label     = train_test_split(w12_label,     train_size=0.5, random_state=seed)
# w3_label, w4_label     = train_test_split(w34_label,     train_size=0.5, random_state=seed)
# w1_label.to_csv("datasets/train_5_w1.csv", index = False, header=None)
# w2_label.to_csv("datasets/train_5_w2.csv", index = False, header=None)
# w3_label.to_csv("datasets/train_5_w3.csv", index = False, header=None)
# w4_label.to_csv("datasets/train_5_w4.csv", index = False, header=None)
# w5_label.to_csv("datasets/train_5_w5.csv", index = False, header=None)

# w1234, w5  = train_test_split(dataset, train_size=0.8, random_state=seed)
# w12,   w34 = train_test_split(w1234,   train_size=0.5, random_state=seed)
# w1, w2     = train_test_split(w12,     train_size=0.5, random_state=seed)
# w3, w4     = train_test_split(w34,     train_size=0.5, random_state=seed)
# w1.to_csv("datasets/train_5_w1_raw.csv", index = False, header=None)
# w2.to_csv("datasets/train_5_w2_raw.csv", index = False, header=None)
# w3.to_csv("datasets/train_5_w3_raw.csv", index = False, header=None)
# w4.to_csv("datasets/train_5_w4_raw.csv", index = False, header=None)
# w5.to_csv("datasets/train_5_w5_raw.csv", index = False, header=None)