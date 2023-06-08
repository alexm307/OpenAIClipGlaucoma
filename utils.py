
#%%
import colorsys
from os import replace
import random

#%%
import numpy as np
import pandas as pd
import torch
import tqdm
import cv2

# Helper functions for getting labels

def _fix_df(df):
    """
    Prepare the Data Frame to be readable
    """
    df_new = df.drop(['ID'], axis=0)
    df_new.columns = df_new.iloc[0, :]
    df_new.drop([np.nan], axis=0, inplace=True)
    df_new.columns.name = 'ID'
    return df_new


def read_clinical_data(abs_path='../'):
    """
    Return excel data as pandas Data Frame
    """
    df_od = pd.read_excel('../../Datasets/PapilaDB/ClinicalData/patient_data_od.xlsx', index_col=[0])
    df_os = pd.read_excel('../../Datasets/PapilaDB/ClinicalData/patient_data_os.xlsx', index_col=[0])
    return _fix_df(df=df_od), _fix_df(df=df_os)


def get_diagnosis(abs_path='../'):
    """
    Return three arrays of shape 488 with the diagnosis tag, eye ID (od, os)
    and patient ID
    """
    df_od, df_os = read_clinical_data(abs_path=abs_path)

    index_od = np.ones(df_od.iloc[:, 2].values.shape, dtype=np.int8)
    index_os = np.zeros(df_os.iloc[:, 2].values.shape, dtype=np.int8)

    eyeID = np.array(list(zip(index_od, index_os))).reshape(-1)
    tag = np.array(list(zip(df_od.iloc[:, 2].values, df_os.iloc[:, 2].values))).reshape(-1)
    patID = np.array([[int(i.replace('#', ''))] * 2 for i in df_od.index]).reshape(-1)

    return tag, eyeID, patID


def get_all_data(abs_path='../'):
    """
    Return multiple arrays of shape 488 with the all the medical data available for the patient.
    """
    df_od, df_os = read_clinical_data(abs_path=abs_path)

    index_od = np.ones(df_od.iloc[:, 2].values.shape, dtype=np.int8)
    index_os = np.zeros(df_os.iloc[:, 2].values.shape, dtype=np.int8)

    eyeID = np.array(list(zip(index_od, index_os))).reshape(-1)
    tag = np.array(list(zip(df_od.iloc[:, 2].values, df_os.iloc[:, 2].values))).reshape(-1)
    patID = np.array([[int(i.replace('#', ''))] * 2 for i in df_od.index]).reshape(-1)

    dioptre1 = np.array(list(zip(df_od.iloc[:, 3].values, df_os.iloc[:, 3].values))).reshape(-1)
    dioptre2 = np.array(list(zip(df_od.iloc[:, 4].values, df_os.iloc[:, 4].values))).reshape(-1)
    astigmatism = np.array(list(zip(df_od.iloc[:, 5].values, df_os.iloc[:, 5].values))).reshape(-1)

    age = np.array(list(zip(df_od.iloc[:, 0].values, df_os.iloc[:, 0].values))).reshape(-1)
    pachymetry = np.array(list(zip(df_od.iloc[:, 9].values, df_os.iloc[:, 9].values))).reshape(-1)
    axial_length = np.array(list(zip(df_od.iloc[:, 10].values, df_os.iloc[:, 10].values))).reshape(-1)
    phakicOrPseudoP = np.array(list(zip(df_od.iloc[:, 6].values, df_os.iloc[:, 6].values))).reshape(-1)

    return tag, eyeID, patID, dioptre1, dioptre2, astigmatism, age, pachymetry, axial_length, phakicOrPseudoP


def get_mean_std(loader):
    """
    Calculate mean and standard deviation of the images in the loader.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std



def saveModel(epoch, model_state_dict, optimizer_state_dict, loss, PATH):
    """
    Save a trained model for later use.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss
    }, PATH)

def loadModel(PATH):
    """
    Load a previously saved model.
    """
    checkpoint = torch.load(PATH)
    #odel.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
    return checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], checkpoint['epoch'], checkpoint['loss']




class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]







