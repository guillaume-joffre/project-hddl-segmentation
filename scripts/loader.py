from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch


class OxfordPetsDataset(Dataset):
    def __init__(self, transform = None, images_dir = "./data/images", masks_dir = "./data/annotations/trimaps"):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        self.transform = transform
        self.images = sorted(list(self.images_dir.glob("*.jpg")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / img_path.with_suffix(".png").name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # 1-channel

        # Convert mask to 0-background, 1-object
        # mask = torch.tensor(np.array(mask), dtype=torch.long)
        # mask = (mask == 1).long()  # or map 1=object, 2=border, 0=background

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask