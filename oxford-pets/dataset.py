from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import torch
    
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def get_mask_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.PILToTensor(),            # Keeps integer labels
    ])

class OxfordPetsDataset(Dataset):
    def __init__(self, 
            image_transform = get_image_transform(), 
            image_dir = "./data/oxford-pets/images", 
            mask_transform = get_mask_transform(),
            mask_dir = "./data/oxford-pets/annotations/trimaps"
        ):
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        
        self.image_transform = image_transform or get_image_transform()
        self.mask_transform  = mask_transform  or get_mask_transform()
        
        self.images = sorted(list(self.image_dir.glob("*.jpg")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.with_suffix(".png").name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) # 1-channel

        # Convert mask to 0-background, 1-object
        # mask = np.array(mask)
        # mask = (mask == 1).astype(np.int64)
        # mask = Image.fromarray(mask)

        # Apply transforms separately
        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze(0)  # shape: (H, W)

        return image, mask
    

# run with: python scripts/dataset.py on the (HDDLtorch) env
if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    full_dataset = OxfordPetsDataset()
    display_loader = DataLoader(full_dataset, batch_size=4, shuffle=True)
    images, masks = next(iter(display_loader))

    plt.figure(figsize=(16, 4))
    for i in range(4):
        ax = plt.subplot(2, 4, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())   # quick unnormalize for display
        plt.imshow(img)
        plt.axis("off")
        
        ax = plt.subplot(2, 4, i+5)
        plt.imshow(masks[i], cmap="gray")
        plt.title("Mask (Label)")
        plt.axis("off")

    plt.show()


