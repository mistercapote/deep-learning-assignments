import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import functional as F

class UNetBinary(nn.Module):
    def __init__(self) :
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # ENCODER
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64x64
        self.pool = resnet.maxpool # 32x32
        self.enc2 = resnet.layer1  # 32x32, 64 canais
        self.enc3 = resnet.layer2  # 16x16, 128 canais
        self.enc4 = resnet.layer3  # 8x8, 256 canais

        # DECODER
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1) # 1 canal de saída (máscara binária)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.dec3(torch.cat([self.up3(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        return self.final_conv(self.up0(d1))


class UNetDDimensional(nn.Module):
    def __init__(self, D=1) :
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # ENCODER
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64x64
        self.pool = resnet.maxpool # 32x32
        self.enc2 = resnet.layer1  # 32x32, 64 canais
        self.enc3 = resnet.layer2  # 16x16, 128 canais
        self.enc4 = resnet.layer3  # 8x8, 256 canais

        # DECODER
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.semantic_head = nn.Conv2d(32, 1, kernel_size=1) 
        self.embed_head = nn.Conv2d(32, D, 1) # D canais de saída

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.dec3(torch.cat([self.up3(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.up0(d1)
        
        return self.semantic_head(d0),  self.embed_head(d0) 


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(size) for size in [1, 2, 3, 6]])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(4)])

    def forward(self, x):
        size = x.shape[-2:]
        features = [x]
        for pool, conv in zip(self.pools, self.convs):
            features.append(F.interpolate(conv(pool(x)), size=size, mode='bilinear', align_corners=False))
        return torch.cat(features, dim=1)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.ReLU())
        self.conv3x3_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0]), nn.ReLU())
        self.conv3x3_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1]), nn.ReLU())
        self.conv3x3_3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2]), nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, 1), nn.ReLU())

    def forward(self, x):
        x = torch.cat([self.conv1x1(x), self.conv3x3_1(x), self.conv3x3_2(x), self.conv3x3_3(x)], dim=1)
        return self.out_conv(x)


class DeepLabDDimensional(nn.Module):
    def __init__(self, D=2):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # ENCODER (Mesmo do UNetDDimensional)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool 
        self.enc2 = resnet.layer1  
        self.enc3 = resnet.layer2  
        self.enc4 = resnet.layer3  # Saída: 8x8, 256 canais

        # BOTTLENECK: Substitui as Skip Connections pelo ASPP
        self.aspp = ASPP(in_channels=256, out_channels=128)
        
        # DECODER: Upsampling direto (sem skip connections)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.ReLU()
        )
        
        self.semantic_head = nn.Conv2d(32, 1, kernel_size=1) 
        self.embed_head = nn.Conv2d(32, D, 1)


    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x4 = self.enc4(x)
        
        x_aspp = self.aspp(x4)
        d0 = self.up_conv(x_aspp)
        
        return self.semantic_head(d0), self.embed_head(d0)


class SegNetDDimensional(nn.Module):
    def __init__(self, D=2):
        super().__init__()
        
        # ENCODER (Estrutura padrão adaptada para extrair Pool Indices)
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # DECODER (Max Unpooling utilizando os índices memorizados)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU())

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        # HEADS (Semântico e Instância)
        self.semantic_head = nn.Conv2d(32, 1, kernel_size=1)
        self.embed_head = nn.Conv2d(32, D, 1)

    def forward(self, x):
        # Contração: Salva os tensores e os índices de pooling
        x1 = self.enc_conv1(x)
        size1 = x1.size()
        x1_pooled, idx1 = self.pool1(x1)

        x2 = self.enc_conv2(x1_pooled)
        size2 = x2.size()
        x2_pooled, idx2 = self.pool2(x2)
        
        x3 = self.enc_conv3(x2_pooled)
        size3 = x3.size()
        x3_pooled, idx3 = self.pool3(x3)

        # Expansão: Recupera a resolução usando os índices correspondentes
        d3 = self.unpool3(x3_pooled, idx3, output_size=size3)
        d3 = self.dec_conv3(d3)

        d2 = self.unpool2(d3, idx2, output_size=size2)
        d2 = self.dec_conv2(d2)

        d1 = self.unpool1(d2, idx1, output_size=size1)
        d1 = self.dec_conv1(d1)

        return self.semantic_head(d1), self.embed_head(d1)