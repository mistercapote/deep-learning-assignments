import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as    cv
from pathlib import Path
import warnings

def to_tensors(image: np.ndarray, instance_masks: np.ndarray):
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

    binary_mask = (instance_masks.sum(axis=0) > 0).astype(np.float32)
    binary_mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

    instance_gt = np.zeros(instance_masks.shape[1:], dtype=np.int32)
    for i, mask in enumerate(instance_masks):
        instance_gt[mask > 0] = i + 1
        
    return image_tensor, binary_mask_tensor, instance_gt


class DSB2018Dataset(Dataset):
    """
    Dataset otimizado para o DSB2018.
    Aviso (Uso de RAM): Com `cache_in_memory=True`, os tensores processados são mantidos 
    em memória RAM. O dataset DSB2018 original tem cerca de 670 imagens, o que 
    geralmente cabe bem na memória (img_size=128). Para datasets maiores, monitore o uso.
    """
    def __init__(self, root_dir: str, img_size: int = 128, sample_ids: list = None,
                 cache_in_memory: bool = True, cache_in_disk: bool = True):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.sample_ids = sample_ids or sorted(
            p.name for p in self.root_dir.iterdir() if p.is_dir()
        )
        self.cache_in_memory = cache_in_memory
        self.cache_in_disk = cache_in_disk
        self.memory_cache = {}

        if self.cache_in_memory and len(self.sample_ids) > 1000:
            warnings.warn("Dataset grande (>1000 samples). O cache em memória pode esgotar a RAM.")

        # Otimização 1: Resolve glob() e armazena os caminhos apenas uma vez
        self.sample_paths = []
        for sid in self.sample_ids:
            sdir = self.root_dir / sid
            img_path = next((sdir / "images").glob("*.png"), None)
            mask_paths = sorted((sdir / "masks").glob("*.png"))
            self.sample_paths.append((img_path, mask_paths, sdir))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Otimização 2: Busca direto do cache em memória (se disponível)
        if self.cache_in_memory and idx in self.memory_cache:
            return self.memory_cache[idx]

        img_path, mask_paths, sample_dir = self.sample_paths[idx]
        
        # Otimização 3: Busca do cache em disco (evita reprocessamento entre kernels)
        disk_cache_path = sample_dir / f"cache_{self.img_size}.pt"
        if self.cache_in_disk and disk_cache_path.exists():
            tensors = torch.load(disk_cache_path, weights_only=False)
            if self.cache_in_memory:
                self.memory_cache[idx] = tensors
            return tensors

        # Processamento original preservado
        image = cv.imread(str(img_path), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size), interpolation=cv.INTER_LINEAR)

        instance_masks = np.zeros((len(mask_paths), self.img_size, self.img_size), dtype=np.uint8)
        for i, mask_path in enumerate(mask_paths):
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)
            instance_masks[i] = (mask > 127).astype(np.uint8)

        tensors = to_tensors(image, instance_masks)

        # Salva o resultado nos caches habilitados
        if self.cache_in_disk:
            torch.save(tensors, disk_cache_path)
        if self.cache_in_memory:
            self.memory_cache[idx] = tensors

        return tensors

