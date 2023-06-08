import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from transformers import DistilBertTokenizer
from torch.utils.data import Subset

from test import test_model
from utils import get_diagnosis

import config as CFG
from dataset import CustomDataSet, create_transforms, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr


##########NOT NEEDED

def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


########################


def prepare_data():
    y, eyeID, patID = get_diagnosis(CFG.dataDir)

    return y, eyeID, patID


def build_ds(tokenizer, y, eyeID, patID, classes):
    transformer = get_transforms("train")
    dataset = CustomDataSet(
        CFG.dataDir,
        diagnostic=y,
        eyeID=eyeID,
        patID=patID,
        classes=classes,
        tokenizer=tokenizer,
        transform=transformer
    )
    return dataset


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    """
    Training function, which is loading the batches, feeding them to the model and stepping the optimizer and the
    lr_scheduler
    """
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    classes = ['healthy', 'glaucoma', 'suspicious']

    y, eyeID, patID = prepare_data()

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    ds = build_ds(tokenizer, y, eyeID, patID, classes)

    n = len(ds)
    n_test = int(0.2 * n)

    train_ds = Subset(ds, range(n_test))
    valid_ds = Subset(ds, range(n_test, n))

    trainLoader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True
    )

    validLoader = dataloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )

    '''
    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, trainLoader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, validLoader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
'''
    test_model(validLoader, "best.pt")



if __name__ == "__main__":
    main()
