import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class UNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.dec3(torch.cat([self.up3(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        return self.final_conv(self.up0(d1))



class UNetTernary(nn.Module):
	def __init__(self):
		super().__init__()
		resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
		
		self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
		self.pool = resnet.maxpool
		self.enc2 = resnet.layer1
		self.enc3 = resnet.layer2
		self.enc4 = resnet.layer3

		self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
		self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
		self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
		self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
		self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
		self.dec0 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())
		self.final_cls = nn.Conv2d(32, 3, kernel_size=1)
		self.final_dist = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

	def forward(self, x):
		x1 = self.enc1(x)
		x2 = self.enc2(self.pool(x1))
		x3 = self.enc3(x2)
		x4 = self.enc4(x3)

		d3 = self.dec3(torch.cat([self.up3(x4), x3], dim=1))
		d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
		d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
		d0 = self.dec0(self.up0(d1))

		logits_cls = self.final_cls(d0)
		dist_pred = self.final_dist(d0)
		return logits_cls, dist_pred








class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 2, 3]):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.ReLU())
        self.conv3x3_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0]), nn.ReLU())
        self.conv3x3_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1]), nn.ReLU())
        self.conv3x3_3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2]), nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, 1), nn.ReLU())

    def forward(self, x):
        x = torch.cat([self.conv1x1(x), self.conv3x3_1(x), self.conv3x3_2(x), self.conv3x3_3(x)], dim=1)
        return self.out_conv(x)



class PPM(nn.Module):
    def _make_stage(self, in_channels, out_channels,pool_size):
        out = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size)), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        return out
    
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = self._make_stage(in_channels, in_channels//4, 1)
        self.pool2 = self._make_stage(in_channels, in_channels//4, 2)
        self.pool3 = self._make_stage(in_channels, in_channels//4, 3)
        self.pool6 = self._make_stage(in_channels, in_channels//4, 6)
        self.proj = nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1)  # 

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.pool1(x)
        x1 = F.interpolate(x1, size=size, mode='bilinear', align_corners=False)
        x2 = self.pool2(x)
        x2 = F.interpolate(x2, size=size, mode='bilinear', align_corners=False)
        x3 = self.pool3(x)
        x3 = F.interpolate(x3, size=size, mode='bilinear', align_corners=False)
        x4 = self.pool6(x)
        x4 = F.interpolate(x4, size=size, mode='bilinear', align_corners=False)
        out  = torch.cat([x, x1, x2, x3, x4], dim=1)
        out = self.proj(out)
        return out 


class ParseModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_global = nn.Sequential(nn.AdaptiveAvgPool2d((1)), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.pool_global(x)
        x1 =  F.interpolate(x1, size=size, mode='bilinear', align_corners=False)
        out  = torch.cat([x, x1], dim=1)

        return out





# ==============================================================================
# EIXO 1: Mecanismo de Resolução 2 - Atrous Convolution + ASPP (DeepLab Ternary)
# Configurada com o MESMO encoder (ResNet-34) da U-Net Ternary
# ==============================================================================
class DeepLabTernary(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # ENCODER: idêntico ao da UNetTernary (ResNet-34)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool 
        self.enc2 = resnet.layer1  # 64 canais
        self.enc3 = resnet.layer2  # 128 canais
        self.enc4 = resnet.layer3  # 256 canais

        # BOTTLENECK: ASPP mantendo output stride
        self.aspp = ASPP(in_channels=256, out_channels=128)
        
        # DECODER: Upsampling direto (sem skip connections)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.ReLU()
        )
        
        # HEADS TERNÁRIOS
        self.final_cls = nn.Conv2d(32, 3, kernel_size=1)
        self.final_dist = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x4 = self.enc4(x)
        
        x_aspp = self.aspp(x4)
        d0 = self.up_conv(x_aspp)
        
        logits_cls = self.final_cls(d0)
        dist_pred = self.final_dist(d0)
        return logits_cls, dist_pred


# ==============================================================================
# EIXO 1: Mecanismo de Resolução 3 - Pool Indices (SegNet Ternary)
# ==============================================================================
class SegNetTernary(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER com MaxPool que memoriza índices
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # DECODER com MaxUnpooling usando índices
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        # HEADS TERNÁRIOS
        self.final_cls = nn.Conv2d(32, 3, kernel_size=1)
        self.final_dist = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.enc_conv1(x)
        size1 = x1.size()
        x1_p, idx1 = self.pool1(x1)

        x2 = self.enc_conv2(x1_p)
        size2 = x2.size()
        x2_p, idx2 = self.pool2(x2)

        x3 = self.enc_conv3(x2_p)
        size3 = x3.size()
        x3_p, idx3 = self.pool3(x3)

        d3 = self.unpool3(x3_p, idx3, output_size=size3)
        d3 = self.dec_conv3(d3)
        d2 = self.unpool2(d3, idx2, output_size=size2)
        d2 = self.dec_conv2(d2)
        d1 = self.unpool1(d2, idx1, output_size=size1)
        d1 = self.dec_conv1(d1)

        logits_cls = self.final_cls(d1)
        dist_pred = self.final_dist(d1)
        return logits_cls, dist_pred


# ==============================================================================
# EIXO 3: Contexto Global 1 - Image Pooling (ParseNet Ternary)
# ==============================================================================
class ParseNetTernary(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        self.parse = ParseModule(in_channels=256, out_channels=256)
        self.proj = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())

        self.final_cls = nn.Conv2d(32, 3, kernel_size=1)
        self.final_dist = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        bottleneck = self.proj(self.parse(x4))

        d3 = self.dec3(torch.cat([self.up3(bottleneck), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.dec0(self.up0(d1))

        logits_cls = self.final_cls(d0)
        dist_pred = self.final_dist(d0)
        return logits_cls, dist_pred


# ==============================================================================
# EIXO 3: Contexto Global 2 - Pyramid Pooling (PSPNet Ternary)
# ==============================================================================
class PSPNetTernary(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        self.ppm = PPM(in_channels=256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())

        self.final_cls = nn.Conv2d(32, 3, kernel_size=1)
        self.final_dist = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        bottleneck = self.ppm(x4)

        d3 = self.dec3(torch.cat([self.up3(bottleneck), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.dec0(self.up0(d1))

        logits_cls = self.final_cls(d0)
        dist_pred = self.final_dist(d0)
        return logits_cls, dist_pred