import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

from scipy import ndimage
def generate_image(img_size: int, seed: int = None):
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=seed)
    num_ellipses = int(rng.integers(5, 21))
    instance_masks = np.zeros((num_ellipses, img_size, img_size), dtype= np.uint8)
    
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
    image = np.clip(image, 0 , 255 ).astype(np.uint8)

    return image, instance_masks


def to_tensors(image: np.ndarray, instance_masks: np.ndarray):
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

    binary_mask = (instance_masks.sum(axis=0) > 0).astype(np.float32)
    binary_mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

    instance_gt = np.zeros(instance_masks.shape[1:], dtype=np.int32)
    for i, mask in enumerate(instance_masks):
        instance_gt[mask > 0] = i + 1
        
    return image_tensor, binary_mask_tensor, instance_gt


class SyntheticEllipseDataset(Dataset):
    def __init__(self, num_samples: int, img_size: int = 128, seed: int = None) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = None if self.seed is None else self.seed + idx
        image, instance_masks = generate_image(self.img_size, seed=seed)            
        return to_tensors(image, instance_masks)


class DSB2018Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 128, sample_ids: list = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.sample_ids = sample_ids or sorted(
            p.name for p in self.root_dir.iterdir() if p.is_dir()
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_dir = self.root_dir / self.sample_ids[idx]

        image_path = next((sample_dir / "images").glob("*.png"))
        image = cv.imread(str(image_path), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size), interpolation=cv.INTER_LINEAR)

        mask_paths = sorted((sample_dir / "masks").glob("*.png"))
        instance_masks = np.zeros((len(mask_paths), self.img_size, self.img_size), dtype=np.uint8)
        for i, mask_path in enumerate(mask_paths):
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)
            instance_masks[i] = (mask > 127).astype(np.uint8)

        return to_tensors(image, instance_masks)


def build_dataset(dataset_type: str, **kwargs):
    if dataset_type == "synthetic":
        return SyntheticEllipseDataset(**kwargs)
    elif dataset_type == "real":
        return DSB2018Dataset(**kwargs)

def collate_fn_ternary(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    dists = torch.stack([b[2] for b in batch])
    instance_masks = [b[3] for b in batch]
    ids = [b[4] for b in batch]
    return images, labels, dists, instance_masks, ids




IMG_SIZE = 128
BOUNDARY_THICKNESS = 2  


def generate_ternary_and_distance(instance_masks, H, W, boundary_thickness=BOUNDARY_THICKNESS):
    """
    Gera, a partir da lista de máscaras binárias de instância (resolução
    original, ANTES do resize):
      - label   (H,W) int64  : 0=fundo, 1=interior, 2=fronteira
      - dist_map(H,W) float32: distância normalizada [0,1] ao fundo

    COMO A FRONTEIRA É GERADA:
      Para cada instância i, erodemos a máscara m_i com elemento estruturante
      de conectividade-8, `boundary_thickness` iterações. O anel de pixels
      removidos pela erosão (m_i AND NOT erode(m_i)) é a fronteira daquela
      instância. A classe "fronteira" global é a UNIÃO desses anéis de todas
      as instâncias, e tem PRIORIDADE sobre "interior": se um pixel cai em
      fronteira de uma instância e interior de outra (instâncias coladas),
      ele fica marcado como fronteira — que é justamente o pixel ambíguo que
      queremos que o modelo aprenda a reconhecer como "corte" entre núcleos.

    ESPESSURA (boundary_thickness=2 px, padrão):
      - Fina demais (1px): depois que a CNN suaviza a predição, o gap de 1px
        some facilmente e o watershed volta a fundir instâncias vizinhas.
      - Grossa demais (4-5px): em núcleos pequenos (diâmetro ~8-10px) a
        erosão come quase todo o interior, deixando poucos pixels de
        marcador confiável para o watershed, e a classe interior fica
        pequena/ruidosa demais para treinar bem.
      - 2px é o meio-termo padrão adotado em trabalhos de segmentação de
        núcleos/células com watershed marcado.

    MAPA DE DISTÂNCIA:
      Para cada instância, calculamos a distância euclidiana ao fundo
      (scipy.ndimage.distance_transform_edt) e normalizamos pelo máximo
      DENTRO daquela própria instância. Isso é importante: sem essa
      normalização por instância, núcleos grandes dominariam a loss (teriam
      valores de distância muito maiores que núcleos pequenos); normalizando
      cada um pelo seu próprio pico, todo núcleo contribui numa escala
      [0,1] comparável, do centro (valor 1) até a borda (valor ~0).
    """
    label = np.zeros((H, W), dtype=np.int64)
    dist_map = np.zeros((H, W), dtype=np.float32)
    struct = ndimage.generate_binary_structure(2, 2)

    interiors, boundaries = [], []
    for m in instance_masks:
        m_bool = m.astype(bool)
        if m_bool.sum() == 0:
            continue
        eroded = ndimage.binary_erosion(m_bool, structure=struct,
                                         iterations=boundary_thickness)
        ring = m_bool & ~eroded
        interiors.append(eroded)
        boundaries.append(ring)

        dt = ndimage.distance_transform_edt(m_bool)
        max_dt = dt.max()
        if max_dt > 0:
            dist_map = np.maximum(dist_map, dt / max_dt)

    for interior in interiors:
        label[interior] = 1
    for ring in boundaries:            # fronteira sobrescreve (prioridade máxima)
        label[ring] = 2

    return label, dist_map
class DSB2018TernaryDataset(Dataset):
    def __init__(self, root, img_size=IMG_SIZE, boundary_thickness=BOUNDARY_THICKNESS,
                 augment=None):
        self.root = Path(root)
        self.ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.img_size = img_size
        self.boundary_thickness = boundary_thickness
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _load_instance_masks(self, mask_dir):
        mask_files = sorted(mask_dir.glob('*.png'))
        return [(np.array(Image.open(mf).convert('L')) > 0).astype(np.uint8)
                for mf in mask_files]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.root / img_id / 'images' / f'{img_id}.png'
        mask_dir = self.root / img_id / 'masks'

        image = np.array(Image.open(img_path).convert('RGB'))
        instance_masks = self._load_instance_masks(mask_dir)
        H, W = image.shape[:2]

        # gera rótulo ternário e mapa de distância NA RESOLUÇÃO ORIGINAL,
        # antes de redimensionar — assim a erosão/distância não é afetada
        # por artefatos de interpolação do resize
        label, dist_map = generate_ternary_and_distance(
            instance_masks, H, W, self.boundary_thickness)

        image_r = cv2.resize(image, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)
        label_r = cv2.resize(label.astype(np.uint8), (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)
        dist_r = cv2.resize(dist_map, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)

        if self.augment is not None:
            aug = self.augment(image=image_r, masks=[label_r, dist_r])
            image_r, (label_r, dist_r) = aug['image'], aug['masks']

        image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
        label_t = torch.from_numpy(label_r).long()                       # (H,W)
        dist_t = torch.from_numpy(dist_r.astype(np.float32)).unsqueeze(0)  # (1,H,W)

        instance_masks_r = [
            cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            for m in instance_masks
        ]

        return image_t, label_t, dist_t, instance_masks_r, img_id
