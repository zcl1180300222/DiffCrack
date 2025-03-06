import os
import csv
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from Ano_dataset import MVTecTrainDataset
from torch.utils.data import DataLoader
from SegNet import SegmentationSubNetwork
from unet import UNetModel
from reconstruction import Reconstruction

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


class FocalL1Loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4):
        super().__init__()
        self.focal = nn.ModuleDict({
            'focal': BinaryFocalLoss(alpha, gamma),
            'l1': nn.SmoothL1Loss()
        })

    def forward(self, pred, target):
        return 5 * self.focal['focal'](pred, target) + self.focal['l1'](pred, target)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', default='config.yaml', help='config file')
    return OmegaConf.load(parser.parse_args().config)


def build_models(config):
    unet = UNetModel(**config.unet).to(config.device)
    unet.load_state_dict(torch.load(config.model_path))
    return nn.DataParallel(unet).eval(), nn.DataParallel(SegmentationSubNetwork(6, 1)).to(config.device)


def train_epoch(model, loader, loss_fn, optimizer, scheduler, recon, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        img, mask = batch['augmented_image'].to(device), batch['anomaly_mask'].to(device)
        with torch.no_grad():
            pred_x0 = recon(img, img, 0.5)[-1]
        pred = model(torch.cat((img, pred_x0), 1))
        loss = loss_fn(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(loader)


def main():
    config = load_config()
    unet, seg_model = build_models(config)
    train_set = DataLoader(MVTecTrainDataset(**config.dataset), batch_size=32, shuffle=True)

    optimizer = optim.Adam(seg_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = FocalL1Loss().to(config.device)

    os.makedirs(config.save_dir, exist_ok=True)
    with open(f"{config.save_dir}/loss_log.csv", "w") as f:
        writer = csv.writer(f)
        for epoch in range(100):
            avg_loss = train_epoch(seg_model, train_set, loss_fn, optimizer, scheduler, Reconstruction(unet, config),
                                   config.device)
            writer.writerow([epoch + 1, avg_loss])
            if (epoch + 1) % 25 == 0:
                torch.save(seg_model.module.state_dict(), f"{config.save_dir}/seg_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()