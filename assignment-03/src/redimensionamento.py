
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
        imagem_carregada = cv2.imread(os.path.join(self.raiz, linha['image_id'] + ".jpg"))
        rotulo = torch.tensor(linha['label'], dtype = torch.long)

        imagem_convertida = cv2.cvtColor(imagem_carregada, cv2.COLOR_BGR2RGB)
    
        imagem_transformada = self.augment(image = imagem_convertida)

        return imagem_transformada['image'], rotulo

def train_epoch(model,device,criterio,   train_loader, val_loader, label, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    history = {'train_loss': [], 'val_loss': [], 'f1_score': []}
    best_f1 = -1

    for epoch in range(epochs):
        model.train()
        accumulated_loss = torch.tensor(0.0, device=device)

        for images, rotulos in train_loader:
            images = images.to(device, non_blocking=True)
            rotulos = rotulos.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterio(logits, rotulos)
            loss.backward()
            optimizer.step()
            accumulated_loss += loss.detach() * images.size(0)
        train_loss = accumulated_loss.item() / len(train_loader.dataset)

        
        model.eval()

        with torch.no_grad():
            previsoes, rotulos_val = [], []
            accumulated_loss_val = 0
            for images, rotulos_lote in val_loader:
                images = images.to(device, non_blocking=True)
                rotulos_lote = rotulos_lote.to(device, non_blocking=True)
                logits = model(images)
                classe_prevista = torch.argmax(logits, dim =1 )
                previsoes.extend(classe_prevista.cpu().numpy())
                rotulos_val.extend(rotulos_lote.cpu().numpy())

                loss = criterio(logits, rotulos_lote)
                accumulated_loss_val += loss.detach() * images.size(0)
            val_loss = accumulated_loss_val.item() / len(val_loader.dataset)
            f1_score_here = f1_score(rotulos_val, previsoes, average='macro')

       
        scheduler.step(f1_score_here)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['f1_score'].append(f1_score_here)

        print(f'Epoch {epoch+1}/{epochs} | F1: {f1_score_here:.4f} | Train Loss: {train_loss} | Val Loss: {val_loss} '  )

        if f1_score_here > best_f1:
            best_f1 = f1_score_here
            torch.save(model.state_dict(), f'models/best_model_{label}.pt')

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



transform_val = A.Compose([
    # 1. Redimensionamento e Padding para 224x224
    A.LongestMaxSize(max_size=224),

    A.PadIfNeeded(
        min_height=224,
        min_width=224,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(0, 0, 0)
    ),

    # 3. Normalização (ImageNet)
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    ),

    # Conversão para tensor PyTorch
    ToTensorV2(),
])


class CNN_pequena(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bloco_inicial = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels=16, kernel_size = 3, padding = 1 ),
         nn.ReLU(), nn.MaxPool2d(2), 
         nn.Conv2d(in_channels = 16 , out_channels=16, kernel_size = 3, padding = 1 ),

          nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 16, out_channels=16, kernel_size = 3, padding = 1 ),
          
                    nn.ReLU(), nn.MaxPool2d(2),

           nn.Flatten(), nn.Linear(12544, 21) )

    def forward(self, x):
        out = self.bloco_inicial(x)
        return out

def eval_model(modelo, loader, criterio, device):
    modelo.eval()

    with torch.no_grad():
                previsoes, rotulos = [], []
                accumulated_loss_test = 0
                for images, rotulos_lote in loader:
                    images = images.to(device, non_blocking=True)
                    rotulos_lote = rotulos_lote.to(device, non_blocking=True)
                    logits = modelo(images)
                    classe_prevista = torch.argmax(logits, dim =1 )
                    previsoes.extend(classe_prevista.cpu().numpy())
                    rotulos.extend(rotulos_lote.cpu().numpy())

                    loss = criterio(logits, rotulos_lote)
                    accumulated_loss_test += loss.detach()* images.size(0)
                test_loss = accumulated_loss_test.item()/ len(loader.dataset)
                f1_score_here  = f1_score(previsoes, rotulos,average='macro')

         
        
               
                print(f' F1: {f1_score_here:.4f} |  Teste Loss {test_loss} '  )
    return previsoes, rotulos