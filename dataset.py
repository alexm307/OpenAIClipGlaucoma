import albumentations as A
import config as CFG

# Create custom dataset, adapted to the available data from PAPILA
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
from PIL import Image

from torchvision.transforms import transforms
# using regex to get the number from filename
import re
import cv2

def image_preprocessing(image):

    image = cv2.imread(image)
    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + 30

    return final_img

'''

The dataset class, which needs to return both images and texts.
DistilBERT is used for the text encoding.
So, sentences (captions) are tokenized with DistilBERT tokenizer and then we feed the token ids (input_ids) and
the attention masks to DistilBERT.
Normally, the captions are just normal text describing the image. For this task, the text will also contain all 
the medical information available for the patient.

'''
class CustomDataSet(Dataset):
    def __init__(self, root_dir, diagnostic, eyeID, patID, classes, tokenizer, transform=None):

        self.data = []
        self.transform = transform
        self.dir = root_dir

#to do: Make the caption as a text containing all nunmerical data as well as the class???
#Give bigger weight to class or what??

        # need to find according diagnosis for the eye found
        # find indexes of the patiend id in the file name, then find eye index based on OD or OS (OD = 0, OS = 1)
        for img in os.listdir(root_dir):
            num = re.findall(r'\d+', img)
            for idx in patID:
                if (idx == int(num[0])):
                    xLoc = np.where(patID == idx)[0][0]
                    # print(xLoc)
                    if (img.find("OD") != -1) and (eyeID[xLoc] == 1):
                        self.data.append([img, classes[diagnostic[xLoc]]])
                    else:
                        if (img.find("OS") != -1) and (eyeID[xLoc + 1] == 0):
                            self.data.append([img, classes[diagnostic[xLoc + 1]]])

        captions = []

        for x, y in self.data:
            captions.append(y)

        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )

        self.classes = classes
        print(f"Total Classes:{len(self.classes)}")  # should be 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        img_path, class_name = self.data[idx]
        img_path = os.path.join(self.dir, img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            openImg =self.transform(image=image)['image']

        item['image'] = torch.tensor(openImg).permute(2, 0, 1).float()
        item['caption'] = class_name

        return item





class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and captions must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


# Data Transformer

def create_transforms(normalize=False, mean=[0, 0, 0], std=[1, 1, 1]):
    if normalize:
        transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    else:
        transformer = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    return transformer