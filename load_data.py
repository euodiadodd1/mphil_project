import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from glob import glob

class CustomDataset(Dataset):
  def __init__(self, data_filenames, label_filenames):
    self.data_filenames = data_filenames
    self.label_filenames = label_filenames
    self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data_filenames)

  def __getitem__(self, idx):
    data = pd.read_csv(self.data_filenames[idx], header=None, 
                 index_col=False).to_numpy().astype(float)
    labels = self.label_filenames[idx].astype(float)

    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(data)
    
    if idx == self.__len__():  
            raise IndexError  
    #print(d.shape,l.shape)
    return self.transform(scaled_data), labels

class FeatureFusionDataset(Dataset):
  def __init__(self, data_filenames, label_filenames):
    self.data_filenames = data_filenames
    self.label_filenames = label_filenames
    self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data_filenames)

  def __getitem__(self, idx):
    gaze_data = pd.read_csv(self.data_filenames[idx][0], header=None, 
                 index_col=False)
    au_data = pd.read_csv(self.data_filenames[idx][1], header=None, 
                 index_col=False)
    pose_data = pd.read_csv(self.data_filenames[idx][2], header=None, 
                 index_col=False)
    labels = self.label_filenames[idx].astype(float)

    data = pd.concat([pose_data,gaze_data, au_data], axis = 0).to_numpy().astype(float)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(data)
    
    #print(scaled_data)
    if idx == self.__len__():  
            raise IndexError  
    #print(d.shape,l.shape)
    return self.transform(scaled_data), labels

class DecisionFusionDataset(Dataset):
  def __init__(self, data_filenames, label_filenames):
    self.data_filenames = data_filenames
    self.label_filenames = label_filenames
    self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data_filenames)

  def __getitem__(self, idx):
    gaze_data = pd.read_csv(self.data_filenames[idx][0], header=None, 
                 index_col=False)
    au_data = pd.read_csv(self.data_filenames[idx][1], header=None, 
                 index_col=False)
    pose_data = pd.read_csv(self.data_filenames[idx][2], header=None, 
                 index_col=False)
    labels = self.label_filenames[idx].astype(float)

    data = [pose_data,gaze_data, au_data]
    data = [self.transform(x.to_numpy().astype(float)) for x in data]
    
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(data)
    if idx == self.__len__():  
            raise IndexError  
    #print(d.shape,l.shape)
    return scaled_data, labels

def get_labels(part, dir):
  if part == "train":
    df = pd.read_csv("/content/drive/MyDrive/metadata_train/parts_train.csv")
    df2 = pd.read_csv("/content/drive/MyDrive/metadata_train/sessions_train.csv")
  else:
    df = pd.read_csv("/content/drive/MyDrive/metadata_val/parts_val_unmasked.csv")
    df2 = pd.read_csv("/content/drive/MyDrive/metadata_val/sessions_val.csv")

  label_idx = ["OPENMINDEDNESS_Z", "CONSCIENTIOUSNESS_Z", "EXTRAVERSION_Z", "AGREEABLENESS_Z", "NEGATIVEEMOTIONALITY_Z"]
  participants = df2.loc[df2["ID"] == dir]
  p1 = participants["PART.1"].values[0]
  p2 = participants["PART.2"].values[0]
  l1 = np.asarray(df.loc[df["ID"].isin([p1,p2])][label_idx].values)

  return l1

def modality_data(modality, batch_size, section):
    ## download and load training dataset
    
    
    train_dir_list = [x for x in glob("/content/drive/MyDrive/"+ section + "_spectral/"+ modality + "/*")]
    test_dir_list = [x for x in glob("/content/drive/MyDrive/" + section + "_spectral_val/"+ modality + "/*")]
    #print(os.path.split(train_dir_list[0])[1])
    train_dirs = train_dir_list
    test_dirs = test_dir_list

    train_file_list = []
    train_labels = np.zeros((1,5))
    test_file_list = []
    test_labels =  np.zeros((1,5))

    for i in train_dirs:
        dir = int(os.path.split(i)[1])
        #print(dir)
        files = [x for x in glob(i+"/*.csv")]
        labels = get_labels("train",dir)
        #train_file_list.append(files)
        train_file_list = np.concatenate([train_file_list, files])
        train_labels = np.vstack([train_labels, labels[0], labels[1]])

    for i in test_dirs:
        dir = int(os.path.split(i)[1])
        files = [x for x in glob(i+"/*.csv")]
        labels = get_labels("test", dir)
        #train_file_list.append(files)
        test_file_list = np.concatenate([test_file_list, files])
        test_labels = np.vstack([train_labels, labels[0], labels[1]])


    train_labels = np.delete(train_labels, 0, axis=0)
    test_labels = np.delete(test_labels, 0,  axis=0)


    train_dataset = CustomDataset(data_filenames = train_file_list, label_filenames= train_labels)
    test_dataset = CustomDataset(data_filenames = test_file_list,label_filenames= test_labels)

    # #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    return trainloader, testloader

def fused_data(fusion_strategy, batch_size, section):
      
    ## download and load training dataset

    pose_dir_list = np.sort([x for x in glob("/content/drive/MyDrive/"+ section +"_spectral/pose/*")])
    gaze_dir_list = np.sort([x for x in glob("/content/drive/MyDrive/"+ section +"_spectral/gaze/*")])
    au_dir_list = np.sort([x for x in glob("/content/drive/MyDrive/"+ section +"_spectral/au/*")])

    train_dir_list = np.column_stack([pose_dir_list,gaze_dir_list,au_dir_list])

    ts_pose_dir_list = np.sort([x for x in glob("/content/drive/MyDrive/"+ section +"_spectral_val/pose/*")])
    ts_gaze_dir_list = np.sort([x for x in glob("/content/drive/MyDrive/"+ section +"_spectral_val/gaze/*")])
    ts_au_dir_list = np.sort([x for x in glob("/content/drive/MyDrive/"+ section +"_spectral_val/au/*")])

    test_dir_list = np.column_stack([ts_pose_dir_list,ts_gaze_dir_list,ts_au_dir_list])

    # train_size = int(0.8 * len(dir_list))
    # test_size = len(dir_list) - train_size

    train_dirs = train_dir_list
    test_dirs = test_dir_list

    train_file_list = np.zeros((1,3))
    train_labels = np.zeros((1,5))
    test_file_list = np.zeros((1,3))
    test_labels =  np.zeros((1,5))



    for i in train_dirs:
        dir = int(int(os.path.split(i[0])[1]))
        gaze = np.sort([x for x in glob(i[1]+"/*.csv")])
        pose = np.sort([x for x in glob(i[0]+"/*.csv")])
        au = np.sort([x for x in glob(i[2]+"/*.csv")])
        if gaze.shape == (4,):
            print(gaze.shape, pose.shape)
            print(i)
        ls = np.column_stack([pose,gaze,au])
        labels = get_labels("train",dir)
        train_file_list = np.concatenate([train_file_list, ls])
        train_labels = np.vstack([train_labels, labels[0], labels[1]])

    for i in test_dirs:
        dir = int(int(os.path.split(i[0])[1]))
        gaze = np.sort([x for x in glob(i[1]+"/*.csv")])
        pose = np.sort([x for x in glob(i[0]+"/*.csv")])
        au = np.sort([x for x in glob(i[2]+"/*.csv")])
        ls = np.column_stack([pose,gaze,au])
        labels = get_labels("test", dir)
        #train_file_list.append(files)
        test_file_list = np.concatenate([test_file_list, ls])
        test_labels = np.vstack([train_labels, labels[0], labels[1]])

        train_file_list = np.delete(train_file_list, 0, axis=0)
        test_file_list = np.delete(test_file_list, 0, axis=0)
        train_labels = np.delete(train_labels, 0, axis=0)
        test_labels = np.delete(test_labels, 0,  axis=0)


    #file_list = [f for f in glob("F:/transformed_data/Participant_Receiver/Gaze/*/*.csv") if "P0301" not in f]
    #label_list = [f for f in glob("F:/training/Training_dataset/Label/competence/*/*.csv")]

    dims = [pd.read_csv(glob(i+"/*.csv")[0], header=None, 
                 index_col=False).shape for i in train_dirs[0]]

    train_file_list.sort()
    test_file_list.sort()
    train_labels.sort()
    test_labels.sort()

    if fusion_strategy in ["avg_decision","decision","attention"]:

        train_dataset = DecisionFusionDataset(data_filenames = train_file_list,label_filenames= train_labels)
        test_dataset = DecisionFusionDataset(data_filenames = test_file_list, label_filenames= test_labels)

    else: 
        train_dataset = FeatureFusionDataset(data_filenames = train_file_list,label_filenames= train_labels)
        test_dataset = FeatureFusionDataset(data_filenames = test_file_list, label_filenames= test_labels)


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)

    return trainloader, testloader, dims


