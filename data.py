"""Stratified split with exact-duplicate removal before splitting."""
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def split_dataset(root, seed=42):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    groups, seen = defaultdict(list), {}
    duplicates = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError('Images must be placed in class subdirectories.')
        label = relative.parts[0]
        # Pixel hash also catches identical images saved under different metadata.
        with Image.open(path) as image:
            image = image.convert('RGB')
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in seen:
            if seen[digest] != label:
                raise ValueError(f'Identical image assigned to different classes: {path}')
            duplicates += 1
            continue
        seen[digest] = label
        groups[label].append(str(relative))
    classes = sorted(groups)
    if len(classes) < 2:
        raise ValueError('At least two classes are required.')
    rng = random.Random(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for target, label in enumerate(classes):
        paths = groups[label][:]
        if len(paths) < 5:
            raise ValueError(f'{label}: need at least five unique images.')
        rng.shuffle(paths)
        count = max(1, round(len(paths) * 0.15))
        for split, selected in (('val', paths[:count]), ('test', paths[count:2 * count]),
                                ('train', paths[2 * count:])):
            splits[split].extend({'path': p, 'label': target} for p in selected)
    return {'root': str(root), 'seed': seed, 'classes': classes,
            'duplicates_removed': duplicates, 'splits': splits}


class LeafDataset(Dataset):
    def __init__(self, manifest, split, size=128):
        self.root = Path(manifest['root'])
        self.rows = manifest['splits'][split]
        steps = [transforms.Resize((size, size))]
        if split == 'train':
            steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                      transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)]
        steps += [transforms.ToTensor(), transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, row['label']
