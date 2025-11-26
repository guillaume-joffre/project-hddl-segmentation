from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import torch
    
def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def get_mask_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.PILToTensor(),            # Keeps integer labels
    ])

class OxfordPetsDataset(Dataset):
    def __init__(self, transform = get_base_transforms(), images_dir = "./data/oxford-pets/images", masks_dir = "./data/oxford-pets/annotations/trimaps"):
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
    

# run with: python scripts/dataset.py on the (HDDLtorch) env
if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    full_dataset = OxfordPetsDataset()
    display_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(display_loader))
    print(labels)
    print(images)

    plt.figure(figsize=(16, 4))
    for i in range(N):
        ax = plt.subplot(2, N//2, i+1)
        img = unnormalize(images[i])
        plt.imshow(img)
        plt.axis("off")

    plt.show()


