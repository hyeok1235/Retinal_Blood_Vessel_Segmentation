import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder blocks
        self.conv1 = self.conv_block(3, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 1024)

        # Downsampling (pooling)
        self.pool = nn.MaxPool2d(2)

        # Decoder blocks with attention gates
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = self.conv_block(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = self.conv_block(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = self.conv_block(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = self.conv_block(128, 64)

        # Final output layer
        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def attention_gate(self, encoder_feature, decoder_feature, filters):
        # Attention mechanism
        encoder_avg = F.adaptive_avg_pool2d(encoder_feature, 1)
        encoder_avg = F.interpolate(encoder_avg, size=decoder_feature.shape[2:], mode="bilinear", align_corners=False)

        attention = torch.cat([encoder_avg, decoder_feature], dim=1)
        attention = self.conv_block(attention.size(1), filters)(attention)
        attention = torch.sigmoid(nn.Conv2d(filters, 1, kernel_size=1)(attention))

        return attention * decoder_feature

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool(conv4)

        conv5 = self.conv5(pool4)

        # Decoder with attention gates
        up6 = self.up6(conv5)
        att6 = self.attention_gate(conv4, up6, 512)
        merge6 = torch.cat([att6, conv4], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        att7 = self.attention_gate(conv3, up7, 256)
        merge7 = torch.cat([att7, conv3], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        att8 = self.attention_gate(conv2, up8, 128)
        merge8 = torch.cat([att8, conv2], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        att9 = self.attention_gate(conv1, up9, 64)
        merge9 = torch.cat([att9, conv1], dim=1)
        conv9 = self.conv9(merge9)

        # Output layer
        output = torch.sigmoid(self.conv10(conv9))
        return output
