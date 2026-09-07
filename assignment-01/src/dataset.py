import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from pathlib import Path
import warnings
import scipy.ndimage as ndi


def to_tensors(image: np.ndarray, instance_masks: np.ndarray, part: int = 1):
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
    
    img_h, img_w = instance_masks.shape[1:]
    instance_gt = np.zeros((img_h, img_w), dtype=np.int32)
    
    if part == 1:
        binary_mask = (instance_masks.sum(axis=0) > 0).astype(np.float32)
        target_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
        
        for i, mask in enumerate(instance_masks):
            instance_gt[mask > 0] = i + 1
            
    elif part == 2:
        semantic_target = np.zeros((img_h, img_w), dtype=np.int64)
        
        for i, mask in enumerate(instance_masks):
            if mask.sum() == 0: 
                continue
            
            interior = ndi.binary_erosion(mask, iterations=2)
            boundary = (mask > 0) & (~interior)
            
            semantic_target[interior > 0] = 1
            semantic_target[boundary > 0] = 2
            instance_gt[mask > 0] = i + 1
            
        target_tensor = torch.from_numpy(semantic_target).long()
        
    return image_tensor, target_tensor, instance_gt


def generate_image(img_size: int, seed: int = None):
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=seed)
    num_ellipses = int(rng.integers(5, 21))
    instance_masks = np.zeros((num_ellipses, img_size, img_size), dtype=np.uint8)
    
    for idx in range(num_ellipses):
        center = tuple(rng.integers(0, img_size + 1, size=2).tolist())
        axes = tuple(rng.integers(5, 41, size=2).tolist())
        angle = int(rng.integers(0, 101))
        gray_intensity = int(rng.integers(50, 256))
        color = (gray_intensity, gray_intensity, gray_intensity)
        
        cv.ellipse(image, center, axes, angle, 0, 360, color, -1)
        cv.ellipse(instance_masks[idx], center, axes, angle, 0, 360, 1, -1)
        
    contrast = rng.random() + 0.5
    noise = rng.integers(-20, 20, size=(img_size, img_size, 3))
    image = (image * contrast) + noise
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, instance_masks


class SyntheticEllipseDataset(Dataset):
    def __init__(self, num_samples: int, img_size: int, part: int, seed: int = None) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed
        self.part = part

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = None if self.seed is None else self.seed + idx
        image, instance_masks = generate_image(self.img_size, seed=seed)            
        return to_tensors(image, instance_masks, self.part)


class DSB2018Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: int, part: int, sample_ids: list = None,
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
        self.part = part

        if self.cache_in_memory and len(self.sample_ids) > 1000:
            warnings.warn("Dataset grande (>1000 samples). O cache em memória pode esgotar a RAM.")

        self.sample_paths = []
        for sid in self.sample_ids:
            sdir = self.root_dir / sid
            img_path = next((sdir / "images").glob("*.png"), None)
            mask_paths = sorted((sdir / "masks").glob("*.png"))
            self.sample_paths.append((img_path, mask_paths, sdir))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.memory_cache:
            return self.memory_cache[idx]

        img_path, mask_paths, sample_dir = self.sample_paths[idx]
        
        disk_cache_path = sample_dir / f"cache_part{self.part}_{self.img_size}.pt"
        if self.cache_in_disk and disk_cache_path.exists():
            tensors = torch.load(disk_cache_path, weights_only=False)
            if self.cache_in_memory:
                self.memory_cache[idx] = tensors
            return tensors

        image = cv.imread(str(img_path), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size), interpolation=cv.INTER_LINEAR)

        instance_masks = np.zeros((len(mask_paths), self.img_size, self.img_size), dtype=np.uint8)
        for i, mask_path in enumerate(mask_paths):
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)
            instance_masks[i] = (mask > 127).astype(np.uint8)

        tensors = to_tensors(image, instance_masks, self.part)

        if self.cache_in_disk:
            torch.save(tensors, disk_cache_path)
        if self.cache_in_memory:
            self.memory_cache[idx] = tensors

        return tensors