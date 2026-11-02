
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os 
from sklearn.metrics import f1_score

class DatasetTrap(Dataset):
    def __init__(self,df: pd.DataFrame,raiz,augment=None,) -> None:
        super().__init__()
        self.df = df
        self.raiz = raiz
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        imagem_carregada = cv2.imread(os.path.join(self.raiz, linha['image_id'] ))

        rotulo = torch.tensor(linha['label'], dtype = torch.long)

        imagem_convertida = cv2.cvtColor(imagem_carregada, cv2.COLOR_BGR2RGB)
    
        imagem_transformada = self.augment(image = imagem_convertida)

        return imagem_transformada['image'], rotulo

def train_epoch(model, train_loader, val_loader, label, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    history = {'train_loss': [], 'val_loss': []}
    best_f1 = -1

    for epoch in range(epochs):
        model.train()
        accumulated_loss = torch.tensor(0.0, device=DEVICE)

        for images, rotulos in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterio(logits, masks)
            loss.backward()
            optimizer.step()
            accumulated_loss += loss.detach() * images.size(0)
        train_loss = accumulated_loss.item() / len(train_loader.dataset)

        
        model.eval()
        total_iou = torch.tensor(0.0, device=DEVICE)
        total_dice = torch.tensor(0.0, device=DEVICE)

        with torch.no_grad():
            previsoes, rotulos = [], []
            accumulated_loss_val = 0
            for images, rotulos_lote in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                rotulos_lote = rotulos_lote.to(DEVICE, non_blocking=True)
                logits = model(images)
                classe_prevista = torch.argmax(logits, dim =1 )
                previsoes.extend(classe_prevista.cpu().numpy())
                rotulos.extend(rotulos_lote.cpu().numpy())

                loss = criterio(logits, rotulos_lote)
                accumulated_loss_val += loss.detach()
            val_loss = accumulated_loss_val.item()/ len(val_loader.dataset)
            f1_score_here  = f1_score(previsoes, rotulos,average='macro' )

        val_iou = total_iou.item() / len(val_loader.dataset)
        val_dice = total_dice.item() / len(val_loader.dataset)
        scheduler.step(val_iou)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}')

        if f1_score_here > best_f1:
            best_f1 = f1_score_here
            torch.save(model.state_dict(), f'../models/best_model_{label}.pt')

    return history


transform_treino = A.Compose([
    # 1. Redimensionamento e Padding para 224x224
    A.LongestMaxSize(max_size=224),

    A.PadIfNeeded(
        min_height=224,
        min_width=224,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(0, 0, 0)
    ),

    # 2. Aumentação de dados
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.7,
    ),

    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5,
    ),

    # Espelhamento horizontal
    A.HorizontalFlip(p=0.5),

    # 3. Normalização (ImageNet)
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    ),

    # Conversão para tensor PyTorch
    ToTensorV2(),
])



df_treino = pd.read_csv("assignment-03/pa3_dados/labels/rotulos_5.csv ")
# Extrai a coluna de rótulos (vamos assumir que a coluna se chama 'label')
rotulos_treino = df_treino['label'].values


modelo = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
num_entradas = modelo.fc.in_features
modelo.fc = nn.Linear(num_entradas, 21)

# Calcula os pesos matematicamente
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(rotulos_treino), y=rotulos_treino)

# Converte para tensor e passa para a Loss
pesos_tensor = torch.tensor(pesos, dtype=torch.float)
criterio = nn.CrossEntropyLoss(weight=pesos_tensor)