import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        
        # Extract configuration parameters
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.conv_channels = config['conv_channels']
        self.kernel_size = config['kernel_size']
        self.padding = config['padding']
        
        # Encoder (contracting path)
        self.enc_conv1 = self.conv_block(self.in_channels, self.conv_channels[0])
        self.enc_conv2 = self.conv_block(self.conv_channels[0], self.conv_channels[1])
        self.enc_conv3 = self.conv_block(self.conv_channels[1], self.conv_channels[2])
        self.enc_conv4 = self.conv_block(self.conv_channels[2], self.conv_channels[3])
        
        # Decoder (expansive path)
        self.dec_tconv4 = nn.ConvTranspose2d(self.conv_channels[3], self.conv_channels[2], kernel_size=2, stride=2)
        self.dec_conv4 = self.conv_block(self.conv_channels[3], self.conv_channels[2])
        self.dec_tconv3 = nn.ConvTranspose2d(self.conv_channels[2], self.conv_channels[1], kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(self.conv_channels[2], self.conv_channels[1])
        self.dec_tconv2 = nn.ConvTranspose2d(self.conv_channels[1], self.conv_channels[0], kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(self.conv_channels[1], self.conv_channels[0])
        
        # Output convolution
        self.out_conv = nn.Conv2d(self.conv_channels[0], self.out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder path
        enc1 = self.enc_conv1(x)
        enc2 = F.max_pool2d(enc1, kernel_size=2, stride=2)
        enc2 = self.enc_conv2(enc2)
        enc3 = F.max_pool2d(enc2, kernel_size=2, stride=2)
        enc3 = self.enc_conv3(enc3)
        enc4 = F.max_pool2d(enc3, kernel_size=2, stride=2)
        enc4 = self.enc_conv4(enc4)
        
        # Decoder path
        dec4 = self.dec_tconv4(enc4)
        dec4 = torch.cat((enc3, dec4), dim=1)
        dec4 = self.dec_conv4(dec4)
        dec3 = self.dec_tconv3(dec4)
        dec3 = torch.cat((enc2, dec3), dim=1)
        dec3 = self.dec_conv3(dec3)
        dec2 = self.dec_tconv2(dec3)
        dec2 = torch.cat((enc1, dec2), dim=1)
        dec2 = self.dec_conv2(dec2)
        
        # Output layer
        out = self.out_conv(dec2)
        out = torch.sigmoid(out)  # Use sigmoid activation for binary classification
        
        return out