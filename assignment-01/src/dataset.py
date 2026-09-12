import torch
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from pathlib import Path
from scipy import ndimage


def split(n, p=0.2):
    n_val = int(p * n)
    n_train = n - n_val
    return [n_train, n_val]


def collate_fn_binary(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    instance_masks = [b[2] for b in batch]   # lista de listas (tam. variável)
    ids = [b[3] for b in batch]
    return images, masks, instance_masks, ids


def collate_fn_ternary(batch):
	images = torch.stack([b[0] for b in batch])
	labels = torch.stack([b[1] for b in batch])
	dists = torch.stack([b[2] for b in batch])
	instances = [b[3] for b in batch]
	ids = [b[4] for b in batch]
	return images, labels, dists, instances, ids


def generate_synthetic_image(img_size: int = 128, seed: int = None):
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


class SyntheticBinaryDataset(Dataset):
    def __init__(self,num_samples: int,img_size: int = 128,seed: int = None,augment=None,) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed
        self.augment = augment

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = None if self.seed is None else self.seed + idx
        image, instance_masks = generate_synthetic_image(self.img_size, seed=seed)
        img_id = f"synth_{idx:05d}"

        # 1. Garante lista de arrays 2D uint8 para as instâncias
        if isinstance(instance_masks, np.ndarray):
            instance_masks_r = [m.astype(np.uint8) for m in instance_masks]
        else:
            instance_masks_r = [np.asarray(m, dtype=np.uint8) for m in instance_masks]

        # 2. Constrói a máscara binária agregada (H, W) uint8
        if len(instance_masks_r) > 0:
            binary_mask_r = (np.sum(instance_masks_r, axis=0) > 0).astype(np.uint8)
        else:
            h, w = image.shape[:2]
            binary_mask_r = np.zeros((h, w), dtype=np.uint8)

        image_r = image.copy()

        # 3. Augmentations (idêntico ao DSB2018Dataset)
        if self.augment is not None:
            aug = self.augment(image=image_r, mask=binary_mask_r)
            image_r, binary_mask_r = aug["image"], aug["mask"]

        # 4. Conversão para tensores PyTorch
        image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(binary_mask_r.astype(np.float32)).unsqueeze(0)

        return image_t, mask_t, instance_masks_r, img_id


class SyntheticTernaryDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        img_size: int = 128,
        boundary_thickness: int = 2,
        seed: int = None,
        augment=None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.boundary_thickness = boundary_thickness
        self.seed = seed
        self.augment = augment
        self._struct = ndimage.generate_binary_structure(2, 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = None if self.seed is None else self.seed + idx
        image, instance_masks = generate_synthetic_image(self.img_size, seed=seed)
        img_id = f"synth_{idx:05d}"

        # 1. Normaliza as instâncias como lista de arrays 2D uint8
        if isinstance(instance_masks, np.ndarray):
            instance_masks_r = [m.astype(np.uint8) for m in instance_masks if m.sum() > 0]
        else:
            instance_masks_r = [
                np.asarray(m, dtype=np.uint8) for m in instance_masks if np.sum(m) > 0
            ]

        # 2. Gera os mapas de rótulo ternário e distância Euclidiana
        label_r = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        dist_r = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        interiors, boundaries = [], []
        for m in instance_masks_r:
            m_bool = m.astype(bool)

            eroded = ndimage.binary_erosion(
                m_bool, structure=self._struct, iterations=self.boundary_thickness
            )

            # Distância Euclidiana individual normalizada ao interior do objeto [0, 1]
            dt = ndimage.distance_transform_edt(m_bool)
            max_dt = dt.max()
            if max_dt > 0:
                dist_r = np.maximum(dist_r, dt / max_dt)

            # Garante que núcleos muito pequenos não desapareçam por erosão total
            if eroded.sum() == 0:
                center_idx = np.unravel_index(np.argmax(dt), dt.shape)
                eroded[center_idx] = True

            ring = m_bool & ~eroded
            interiors.append(eroded)
            boundaries.append(ring)

        for interior in interiors:
            label_r[interior] = 1  # 1: Interior
        for ring in boundaries:
            label_r[ring] = 2      # 2: Fronteira (prioridade na sobreposição)

        image_r = image.copy()

        # 3. Augmentations em conjunto (imagem, label ternário e mapa de distâncias)
        if self.augment is not None:
            aug = self.augment(image=image_r, masks=[label_r, dist_r])
            image_r = aug["image"]
            label_r, dist_r = aug["masks"]

        # 4. Conversão para tensores PyTorch alinhados ao DSB2018TernaryDataset
        image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
        label_t = torch.from_numpy(label_r).long()
        dist_t = torch.from_numpy(dist_r).unsqueeze(0).float()

        return image_t, label_t, dist_t, instance_masks_r, img_id

    
class DSB2018BinaryDataset(Dataset):
    def __init__(self, root, img_size=128, augment=None,
                 cache_in_memory=True, cache_in_disk=True):
        self.root = Path(root)
        self.ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.img_size = img_size
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        self.cache_in_disk = cache_in_disk
        self.memory_cache = {}

        # pré-computa os paths uma única vez, evita re-glob a cada __getitem__
        self.sample_paths = []
        for img_id in self.ids:
            sdir = self.root / img_id
            img_path = sdir / 'images' / f'{img_id}.png'
            mask_paths = sorted((sdir / 'masks').glob('*.png'))
            self.sample_paths.append((img_path, mask_paths, sdir))

    def __len__(self):
        return len(self.ids)

    def _disk_cache_path(self, sample_dir):
        # nome inclui img_size para não misturar caches de resoluções diferentes
        return sample_dir / f'cache_{self.img_size}.pt'

    def _process_from_disk(self, img_path, mask_paths):
        """Lê PNGs originais e faz resize. Sem augment (fica p/ __getitem__)."""
        image = cv.imread(str(img_path), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_r = cv.resize(image, (self.img_size, self.img_size),
                              interpolation=cv.INTER_LINEAR)

        instance_masks_r = []
        binary_mask_r = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for mp in mask_paths:
            m = cv.imread(str(mp), cv.IMREAD_GRAYSCALE)
            m = cv.resize(m, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)
            m = (m > 127).astype(np.uint8)
            instance_masks_r.append(m)
            binary_mask_r = np.maximum(binary_mask_r, m)

        return {
            'image': image_r,
            'mask': binary_mask_r,
            'instances': instance_masks_r,
        }

    def _get_cached_data(self, idx):
        if self.cache_in_memory and idx in self.memory_cache:
            return self.memory_cache[idx]

        img_path, mask_paths, sample_dir = self.sample_paths[idx]
        disk_cache_path = self._disk_cache_path(sample_dir)

        if self.cache_in_disk and disk_cache_path.exists():
            data = torch.load(disk_cache_path, weights_only=False)
        else:
            data = self._process_from_disk(img_path, mask_paths)
            if self.cache_in_disk:
                torch.save(data, disk_cache_path)

        if self.cache_in_memory:
            self.memory_cache[idx] = data

        return data

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        data = self._get_cached_data(idx)

        # copiar antes de augmentar, pra não corromper o que fica em cache
        image_r = data['image'].copy()
        binary_mask_r = data['mask'].copy()
        instance_masks_r = data['instances']

        if self.augment is not None:
            aug = self.augment(image=image_r, mask=binary_mask_r)
            image_r, binary_mask_r = aug['image'], aug['mask']

        image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(binary_mask_r.astype(np.float32)).unsqueeze(0)

        return image_t, mask_t, instance_masks_r, img_id


def generate_ternary_and_distance(instance_masks, h, w, boundary_thickness=2):
	"""
	Como gerar o rótulo de fronteira a partir das máscaras individuais?
	- Para cada máscara de instância m_i, erodimos com conectividade-8 por `boundary_thickness` iterações;
	- O anel de fronteira de cada núcleo é: ring_i = m_i & ~eroded_i;
	- A classe FRONTEIRA tem prioridade sobre o interior na fusão.

	Que espessura?
	- 1 px tende a sumir devido ao pooling/convoluções da CNN e interpolação;
	- 4 px consome quase todo o interior em núcleos pequenos, destruindo os marcadores;
	- 2 pixels foi esoclhido .

	Mapa de Distância ao Fundo:
	- Distância euclidiana (EDT) normalizada pelo valor máximo dentro da própria instância [0, 1].
	"""
	label = np.zeros((h, w), dtype=np.int64)
	dist_map = np.zeros((h, w), dtype=np.float32)
	struct = ndimage.generate_binary_structure(2, 2)

	interiors, boundaries= [], []

	for m in instance_masks:
		m_bool = m.astype(bool)
		if m_bool.sum() == 0:
			continue

		eroded = ndimage.binary_erosion(m_bool, structure=struct, iterations=boundary_thickness)

		# Distância Euclidiana individual normalizada
		dt = ndimage.distance_transform_edt(m_bool)
		max_dt = dt.max()
		if max_dt > 0:
			dist_map = np.maximum(dist_map, dt / max_dt)

		if eroded.sum() == 0:
			center_idx = np.unravel_index(np.argmax(dt), dt.shape)
			eroded[center_idx] = True

		ring = m_bool & ~eroded
		interiors.append(eroded)
		boundaries.append(ring)


	for interior in interiors:
		label[interior] = 1  # Interior
	for ring in boundaries:
		label[ring] = 2  # Fronteira (prioridade)

	return label, dist_map


class DSB2018TernaryDataset(Dataset):
	def __init__(self,root,img_size=128,boundary_thickness=2,
			augment=None,cache_in_memory=True,cache_in_disk=True,):
		self.root = Path(root)
		self.ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
		self.img_size = img_size
		self.boundary_thickness = boundary_thickness
		self.augment = augment
		self.cache_in_memory = cache_in_memory
		self.cache_in_disk = cache_in_disk
		self.memory_cache = {}

		self.sample_paths = []
		for img_id in self.ids:
			sdir = self.root / img_id
			img_path = sdir / 'images' / f'{img_id}.png'
			mask_paths = sorted((sdir / 'masks').glob('*.png'))
			self.sample_paths.append((img_path, mask_paths, sdir))

	def __len__(self):
		return len(self.ids)

	def _disk_cache_path(self, sample_dir):
		return sample_dir / f'cache_{self.img_size}_ternary.pt'

	def _process_from_disk(self, img_path, mask_paths):
		image = cv.imread(str(img_path), cv.IMREAD_COLOR)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		image_r = cv.resize(
				image, (self.img_size, self.img_size), interpolation=cv.INTER_LINEAR
		)

		instance_masks_r = []
		for mp in mask_paths:
			m = cv.imread(str(mp), cv.IMREAD_GRAYSCALE)
			m = cv.resize(
					m, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST
			)
			m = (m > 127).astype(np.uint8)
			if m.sum() > 0:
				instance_masks_r.append(m)

		label_r, dist_r = generate_ternary_and_distance(
				instance_masks_r,
				self.img_size,
				self.img_size,
				self.boundary_thickness,
		)

		return {
				'image': image_r,
				'label': label_r,
				'dist': dist_r,
				'instances': instance_masks_r,
		}

	def _get_cached_data(self, idx):
		if self.cache_in_memory and idx in self.memory_cache:
			return self.memory_cache[idx]

		img_path, mask_paths, sample_dir = self.sample_paths[idx]
		disk_cache = self._disk_cache_path(sample_dir)

		if self.cache_in_disk and disk_cache.exists():
			data = torch.load(disk_cache, weights_only=False)
		else:
			data = self._process_from_disk(img_path, mask_paths)
			if self.cache_in_disk:
				torch.save(data, disk_cache)

		if self.cache_in_memory:
			self.memory_cache[idx] = data
		return data

	def __getitem__(self, idx):
		data = self._get_cached_data(idx)
		img_id = self.ids[idx]

		image_r = data['image'].copy()
		label_r = data['label'].copy()
		dist_r = data['dist'].copy()
		instance_masks_r = data['instances']

		if self.augment is not None:
			aug = self.augment(image=image_r, masks=[label_r, dist_r])
			image_r = aug['image']
			label_r, dist_r = aug['masks']

		image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
		label_t = torch.from_numpy(label_r).long()
		dist_t = torch.from_numpy(dist_r).unsqueeze(0).float()

		return image_t, label_t, dist_t, instance_masks_r, img_id