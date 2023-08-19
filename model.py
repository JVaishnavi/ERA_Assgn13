#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 22:35:38 2023

@author: vaishnavijanakiraman
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np
import matplotlib.pyplot as plt

import config
from loss import YoloLoss
from dataset import YOLODataset



""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
layers_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


from pytorch_lightning.callbacks import LearningRateFinder


class get_lr(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, 
                              out_channels = out_channels, 
                              bias = not bn_act,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU()
        self.use_bn = bn_act

    def forward(self, x):
        if self.use_bn:
            out = self.leaky(self.bn(self.conv(x)))
        else:
            out = self.conv(x)
        return out
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, use_residual = True, num_repeat = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeat):
            self.layers += [
                nn.Sequential(
                    ConvBlock(in_channels = in_channels, 
                              out_channels = in_channels//2,
                              kernel_size=1,
                              ),
                    ConvBlock(in_channels = in_channels//2, 
                              out_channels = in_channels,
                              kernel_size = 3,
                              padding = 1)
                    )
                ]
        self.use_residual = use_residual 
        self.num_repeat = num_repeat
        
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x
                
                
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            ConvBlock(
                in_channels = in_channels, 
                out_channels = in_channels*2,
                kernel_size = 3,
                padding = 1
                ),
            ConvBlock(
                in_channels = in_channels*2, 
                out_channels = (num_classes + 5)*3,
                bn_act = False,
                kernel_size = 1
                )
            )
        self.num_classes = num_classes
        
    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
            )


class LT_model(LightningModule):
    def __init__(self, learning_rate = 0.001, hidden_size = 16, in_channels = 3, num_classes = 80):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        self.learning_rate = learning_rate
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            )
        
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in layers_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    ConvBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResBlock(in_channels, num_repeat=num_repeats))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResBlock(in_channels, use_residual=False, num_repeat=1),
                        ConvBlock(
                            in_channels = in_channels, 
                            out_channels = in_channels // 2, 
                            kernel_size=1
                            ),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers

        
    def forward(self, x):
        outputs = []
        route_connection = []
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction): #3 images are sent - small, medium, large
                outputs.append(layer(x))
                continue
            
            x = layer(x)
            if isinstance(layer, ResBlock) and layer.num_repeat==8:
                route_connection.append(x)
                
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connection[-1]], dim=1)
                route_connection.pop()
                
        return outputs
    

    
    def criterion(self, out, y):
        loss_fn = YoloLoss()
        y0, y1, y2 = (
                y[0].to(config.DEVICE),
                y[1].to(config.DEVICE),
                y[2].to(config.DEVICE),
            )
        loss = (
            loss_fn(out[0].to(config.DEVICE), y0, self.scaled_anchors[0].to(config.DEVICE))
            + loss_fn(out[1].to(config.DEVICE), y1, self.scaled_anchors[1].to(config.DEVICE))
            + loss_fn(out[2].to(config.DEVICE), y2, self.scaled_anchors[2].to(config.DEVICE))
            )
        return loss
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        #self.scaler.scale(loss).backward()
        #self.scaler.step(self.optimizer)
        #self.scaler.update()
        #self.scheduler.step()
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
         x, y = batch
         out = self(x)
         loss = self.criterion(out, y)
         self.log("val_loss", loss, prog_bar=True)
         return loss
    
    def test_step(self, batch, batch_idx):
         x, y = batch
         out = self(x)
         loss = self.criterion(out, y)
         self.log("test_loss", loss, prog_bar=True)
         return loss

     

    def configure_optimizers(self):
        print("Configuring Optimisers and scheduler")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        max_lr = 1e-3
        train_loader = self.train_dataloader()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             max_lr, 
                                                             steps_per_epoch=len(train_loader),
                                                             epochs = config.NUM_EPOCHS * 2 // 5)
        self.scaler = torch.cuda.amp.GradScaler()
        
 
      ####################
      # DATA RELATED HOOKS
      ####################

    
    def prepare_data(self):
         
        IMAGE_SIZE = config.IMAGE_SIZE
        train_csv_path = config.TRAIN_CSV_PATH
        test_csv_path = config.TEST_CSV_PATH
        
        self.train_dataset = YOLODataset(
            train_csv_path,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            transform=config.train_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        )
        print("Train dataset prepared")

        self.test_dataset = YOLODataset(
            test_csv_path,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        )
        print("Test dataset prepared")

        
        self.train_eval_dataset = YOLODataset(
            train_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        print("Validation dataset prepared")

    def train_dataloader(self):
        print("Train data loader")
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )
        return train_loader
    
    def val_dataloader(self):
        print("Validation data loader")
        train_eval_loader = DataLoader(
            dataset=self.train_eval_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )
        return train_eval_loader
    

    def test_dataloader(self):
         print("Test data loader")
         test_loader = DataLoader(
             dataset=self.test_dataset,
             batch_size=config.BATCH_SIZE,
             num_workers=config.NUM_WORKERS,
             pin_memory=config.PIN_MEMORY,
             shuffle=False,
             drop_last=False,
         )
         self.test_loader = test_loader
         return test_loader

    #Remove the loop and see if you can merge it with the test code.
    #Have conditions on epoc. Print gradcam of 10 images at EPOCH%10==0
    #You dont have to do only for incorrect ones. print 5- correct one, 5-incorrect ones
    
    def get_incorrect(self, batch):
        incorrect_examples = []
        incorrect_labels = []
        incorrect_pred = []
        incorrect_images = []
        
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True) 
        idxs_mask = ((pred == target.view_as(pred))==False).view(-1)
        if idxs_mask.numel(): 
            incorrect_examples.append(data[idxs_mask].squeeze().cpu().numpy())
            incorrect_labels.append(target[idxs_mask].cpu().numpy())
            incorrect_pred.append(pred[idxs_mask].squeeze().cpu().numpy())
            incorrect_images.append(data[idxs_mask])
            
        return incorrect_examples, incorrect_labels, incorrect_pred, incorrect_images
    
    
    def print_gradcam_images(self, batch, n=10):
        
        target_layers = [self.layers[-1]]
        cam = GradCAM(model=self, target_layers=target_layers)
        incorrect_examples, incorrect_labels, incorrect_pred, incorrect_image = self.get_incorrect(self, batch)
        
        classes = self.train_dataset.classes
        
        fig = plt.figure(figsize=(20, 16))
        for idx in np.arange(n):
            ax = fig.add_subplot(n//5, 5, idx+1, xticks=[], yticks=[])
            input_tensor = torch.tensor(incorrect_examples[0][idx:idx+1])
            targets = [ClassifierOutputTarget(incorrect_labels[0][idx])]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            rgb_img = incorrect_examples[0][idx]
            rgb_img = rgb_img/2 + 0.5
            rgb_img = np.clip(rgb_img, 0, 1)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            ax.set_title(f"Predicted label: {classes[incorrect_pred[0][idx]]}\n Actual label: {classes[incorrect_labels[0][idx]]}")
            plt.imshow(visualization)
   
from utils import check_class_accuracy, save_checkpoint

class PrintAccuracyAndLoss(Callback):
    def on_epoch_end(self, trainer, pl_module):
        
        if config.SAVE_MODEL:
            save_checkpoint(self, self.optimizer, filename=f"checkpoint.pth.tar")

        print(f"Currently epoch {self.current_epoch}")
        print("On Train Eval loader:")
        print("On Train loader:")
        check_class_accuracy(self, self.train_dataloader, threshold=config.CONF_THRESHOLD)

        train_acc = self.train_acc.compute()
        val_acc = pl_module.val_acc.compute()
        train_loss = trainer.callback_metrics['train_loss']
        val_loss = trainer.callback_metrics['val_loss']
        print(f"Epoch {trainer.current_epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")       

"""         
model = LT_model()
num_classes = 80
IMAGE_SIZE = 416
x = torch.randn((1, 3, 512, IMAGE_SIZE))
out = model(x)
trainer = Trainer(max_epochs=1, precision=16)
trainer.fit(model)
"""

