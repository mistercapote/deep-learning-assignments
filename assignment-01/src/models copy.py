import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import numpy as np
from skimage.measure import label, regionprops
import glob
from pathlib import Path
from scipy.ndimage import binary_dilation


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



class UNetTernaryDilated(nn.Module):
    def __init__(self, dilation=2):
        super().__init__()

        # ============================================================
        # ENCODER — MESMO RESNET-34 DA UNETTERNARY
        # ============================================================

        resnet = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1
        )

        self.enc1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.pool = resnet.maxpool

        self.enc2 = resnet.layer1   # 64 canais
        self.enc3 = resnet.layer2   # 128 canais
        self.enc4 = resnet.layer3   # 256 canais

        # ============================================================
        # ATRous CONVOLUTION
        #
        # Mantemos a resolução de saída de enc4 igual à de enc3,
        # removendo o downsampling do primeiro bloco.
        # ============================================================

        for i, block in enumerate(self.enc4):

            # Remove o downsampling do primeiro bloco
            if i == 0:
                block.conv1.stride = (1, 1)

                if block.downsample is not None:
                    block.downsample[0].stride = (1, 1)

            # Atrous / dilated convolution
            block.conv1.dilation = (dilation, dilation)
            block.conv1.padding = (dilation, dilation)

            block.conv2.dilation = (dilation, dilation)
            block.conv2.padding = (dilation, dilation)

        # ============================================================
        # DECODER
        #
        # Como enc4 agora possui a mesma resolução espacial de enc3,
        # up3 NÃO deve fazer upsampling.
        # ============================================================

        self.up3 = nn.Conv2d(
            256,
            128,
            kernel_size=1
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # H/8 -> H/4
        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # H/4 -> H/2
        self.up1 = nn.ConvTranspose2d(
            64,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # H/2 -> H
        self.up0 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec0 = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # ============================================================
        # HEADS DA TRILHA A
        # ============================================================

        # 3 classes:
        # 0 = background
        # 1 = interior
        # 2 = boundary
        self.final_cls = nn.Conv2d(
            32,
            3,
            kernel_size=1
        )

        # Mapa contínuo de distância
        self.final_dist = nn.Sequential(
            nn.Conv2d(
                32,
                1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        # ============================================================
        # ENCODER
        # ============================================================

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # ============================================================
        # DECODER
        # ============================================================

        # x4 e x3 possuem a mesma resolução espacial
        d3 = self.dec3(
            torch.cat(
                [self.up3(x4), x3],
                dim=1
            )
        )

        # H/8 -> H/4
        d2 = self.dec2(
            torch.cat(
                [self.up2(d3), x2],
                dim=1
            )
        )

        # H/4 -> H/2
        d1 = self.dec1(
            torch.cat(
                [self.up1(d2), x1],
                dim=1
            )
        )

        # H/2 -> H
        d0 = self.dec0(self.up0(d1))

        # ============================================================
        # SAÍDAS
        # ============================================================

        logits_cls = self.final_cls(d0)
        dist_pred = self.final_dist(d0)

        return logits_cls, dist_pred
    
def update_rf(r, jump, kernel, stride=1, dilation=1):
    """
    Atualiza receptive field e jump para uma operação convolucional/pooling.
    """
    r = r + ((kernel - 1) * dilation) * jump
    jump = jump * stride
    return r, jump

def resnet34_encoder_rf(model):
    r = 1
    jump = 1

    # conv1: 7x7, stride 2
    r, jump = update_rf(r, jump, kernel=7, stride=2)

    # MaxPool: 3x3, stride 2
    r, jump = update_rf(r, jump, kernel=3, stride=2)

    # layer1
    for block in model.enc2:
        r, jump = update_rf(
            r, jump,
            kernel=3,
            stride=block.conv1.stride[0],
            dilation=block.conv1.dilation[0]
        )
        r, jump = update_rf(
            r, jump,
            kernel=3,
            stride=block.conv2.stride[0],
            dilation=block.conv2.dilation[0]
        )

    # layer2
    for block in model.enc3:
        r, jump = update_rf(
            r, jump,
            kernel=3,
            stride=block.conv1.stride[0],
            dilation=block.conv1.dilation[0]
        )
        r, jump = update_rf(
            r, jump,
            kernel=3,
            stride=block.conv2.stride[0],
            dilation=block.conv2.dilation[0]
        )

    # layer3
    for block in model.enc4:
        r, jump = update_rf(
            r, jump,
            kernel=3,
            stride=block.conv1.stride[0],
            dilation=block.conv1.dilation[0]
        )
        r, jump = update_rf(
            r, jump,
            kernel=3,
            stride=block.conv2.stride[0],
            dilation=block.conv2.dilation[0]
        )

    return r, jump


IMG_SIZE=256


def object_diameters(train_dir, target_size=IMG_SIZE):
    """
    Calcula o diâmetro equivalente das instâncias na mesma escala
    espacial usada como entrada do modelo.

    As máscaras são redimensionadas para target_size x target_size
    antes da medição.
    """

    diameters = []

    mask_paths = glob.glob(
        str(train_dir / "*" / "masks" / "*.png")
    )

    for p in mask_paths:

        mask = np.array(Image.open(p))

        # Redimensiona usando nearest neighbor para preservar os rótulos
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (target_size, target_size),
                resample=Image.Resampling.NEAREST
            )
        )

        lab = label(mask_resized > 0)

        for region in regionprops(lab):
            diameters.append(
                region.equivalent_diameter_area
            )

    return np.array(diameters)




def instance_gt_to_ternary(instance_gt, boundary_width=2):
    """
    Converte o ground truth de instâncias em um rótulo ternário.

    Rótulos:
        0 = fundo
        1 = interior da instância
        2 = fronteira entre instâncias

    A fronteira é definida pelos pixels que pertencem à região
    de dilatação de pelo menos duas instâncias diferentes.
    """

    ids = np.unique(instance_gt)
    ids = ids[ids != 0]

    # Caso não existam instâncias
    if len(ids) == 0:
        return np.zeros(
            instance_gt.shape,
            dtype=np.int64
        )

    struct = np.ones((3, 3), dtype=bool)

    # Conta quantas instâncias dilatadas cobrem cada pixel
    dilated_sum = np.zeros(
        instance_gt.shape,
        dtype=np.uint16
    )

    for i in ids:

        inst_mask = instance_gt == i

        dilated = binary_dilation(
            inst_mask,
            structure=struct,
            iterations=boundary_width
        )

        dilated_sum += dilated.astype(np.uint16)

    # Região onde pelo menos duas instâncias se aproximam
    boundary = dilated_sum > 1

    # Interior: pertence a uma instância, mas não à fronteira
    interior = (instance_gt > 0) & ~boundary

    # Rótulo ternário
    ternary = np.zeros(
        instance_gt.shape,
        dtype=np.int64
    )

    ternary[interior] = 1
    ternary[boundary] = 2

    return ternary


def load_instance_gt(mask_dir):
    """Empilha as máscaras individuais do DSB2018 em um único array rotulado (H, W)."""
    mask_dir = Path(mask_dir)
    gt = None

    for i, p in enumerate(sorted(mask_dir.glob("*.png")), start=1):
        m = np.array(Image.open(p)) > 0

        if gt is None:
            gt = np.zeros(m.shape, dtype=np.int32)

        gt[m] = i

    return gt