
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET


class OxfordPetsDataset(Dataset):
    """
    PyTorch Dataset for Oxford Pets with optional masks and bounding boxes.
    Returns a dictionary with keys: 'image', 'animal_id', 'race_id', optional 'bbox' and 'mask'.
    """
      
    @staticmethod
    def default_base_transform() -> A.Compose:
        """
        The default transform that will be applied.
        Bounding boxes are expected in pascal_voc format [xmin, ymin, xmax, ymax].
        
        NOTES:
        Albumentation applies geometric transforms to images/masks/bboxes
        so that all follow the same coordinate space.
        Color changes and other specific transforms are applied to the 
        right data only, automatically (color-changes don't apply to masks/bboxes).
        """
        
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc")) # xmin ymin xmax ymax

      
    @staticmethod
    def default_transform_unnormalize(image):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        
        img = image * std + mean          # revert normalization
        img = img.permute(1,2,0).numpy()  # convert to HWC
        img = np.clip(img, 0, 1)          # ensure values in [0,1] for plt
        return img
      
      
    def __init__(self, 
        fetch_bboxes: bool = False,         # whether or not to fetch the bounding box
        fetch_masks: bool = False,          # whether or not to fetch the segmentation mask
        joint_transform: A.Compose = default_base_transform(),  # albumentation transform for image/bbox/mask
        dataset_root = Path("./data/oxford-pets"),
        identifiers = None
    ): 
        dataset_root = Path(dataset_root).resolve()

        self.images_path = dataset_root / "images"
        self.bboxes_path = dataset_root / "annotations/xmls"
        self.masks_path = dataset_root / "annotations/trimaps"

        self.fetch_bboxes = bool(fetch_bboxes)
        self.fetch_masks = bool(fetch_masks)
        
        self.joint_transform = joint_transform
        
        # data lists
        if identifiers is None:
            self.identifiers = self.get_identifiers() # the list of images in the Dataset
        else:
            self.identifiers = pd.Series(sorted(identifiers), name="identifier")
        
        self.animals = pd.Categorical( self.identifiers.str[0].str.islower().astype(int).map({0: "cat", 1: "dog"}) )
        self.races = pd.Categorical( self.identifiers.str.rpartition("_")[0].str.lower() )


    def __len__(self):
        return len(self.identifiers)
    
    
    def __getitem__(self, idx):
        identifier = self.identifiers[idx]
        
        components = {"image": self.get_image(identifier)}
        components["bboxes"] = [self.get_bbox(identifier)] if self.fetch_bboxes else []
        if self.fetch_masks: components["mask"] = self.get_mask(identifier)

        transformed = self.joint_transform(**components)

        outputs = {
            "image": transformed["image"],
            "animal_id": self.animals.codes[idx],
            "race_id": self.races.codes[idx]
        }
        if self.fetch_bboxes and transformed["bboxes"]: 
            outputs["bbox"] = transformed["bboxes"][0]
        if self.fetch_masks: 
            outputs["mask"] = transformed["mask"]
        
        return outputs
    
    
    def get_identifiers(self, force_all = False):
        ids = set(f.stem for f in self.images_path.glob("*.jpg"))

        # Optionally intersect with available bbox IDs
        if self.fetch_bboxes and not force_all:
            ids &= {f.stem for f in self.bboxes_path.glob("*.xml")}

        # Optionally intersect with available mask IDs
        if self.fetch_masks and not force_all:
            ids &= {f.stem for f in self.masks_path.glob("*.png")}

        # Keep only identifiers that exist in all required folders
        return pd.Series(sorted(ids), name="identifier")
        
        
    def get_image(self, identifier: str):
        filepath = self.images_path / (identifier + ".jpg")
        image = Image.open(filepath).convert("RGB")
        return np.array(image)
    
    
    def get_mask(self, identifier: str):
        filepath = self.masks_path / (identifier + ".png")
        mask = Image.open(filepath) # 1-channel
        return np.array(mask, dtype = np.uint8)
    
    
    def get_bbox(self, identifier: str):
        filepath = self.bboxes_path / (identifier + ".xml")
        root = ET.parse(filepath).getroot()

        obj = root.find("object")
        bbx = obj.find("bndbox")
        xmin = int(bbx.find("xmin").text)
        ymin = int(bbx.find("ymin").text)
        xmax = int(bbx.find("xmax").text)
        ymax = int(bbx.find("ymax").text)

        return [xmin, ymin, xmax, ymax]
    
    
    def animal2index(self, animal: str) -> int:
        return self.animals.categories.get_loc(animal)
    
    
    def index2animal(self, index: int) -> str:
        return self.animals.categories.values[index]
    
    
    def race2index(self, race: str) -> int:
        return self.races.categories.get_loc(race)
    
    
    def index2race(self, index: int) -> str:
        return self.races.categories.values[index]
    
    
    def get_dataframe(self):
        identifiers = self.get_identifiers(force_all = True)
        animals = pd.Categorical(
            identifiers.str[0].str.isupper().astype(int).map({1: "cat", 0: "dog"}),
            categories = self.animals.categories
        )
        races = pd.Categorical(
            identifiers.str.rpartition("_")[0].str.lower(),
            categories = self.races.categories
        )
        
        dataframe = pd.DataFrame({
            "identifier": identifiers,
            "animal": animals,
            "animal.id": animals.codes,
            "race": races,
            "race.id": races.codes,
            "imagefile": self.images_path / (identifiers + ".jpg"),
            "bboxfile":  self.bboxes_path / (identifiers + ".xml"),
            "maskfile":  self.masks_path  / (identifiers + ".png")
        })
        
        # sets all paths leading to non existing files to pd.NA
        for col in ["imagefile", "maskfile", "bboxfile"]:
            paths = dataframe[col].astype(str)
            exists = paths.apply(os.path.exists)
            dataframe.loc[~exists, col] = pd.NA
    
        # dataframe.index.name = "index"
        return dataframe #.reset_index()
    
        
# run with: python oxford-pets/dataset.py on the conda (HDDL.1) env
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    dset = OxfordPetsDataset(fetch_bboxes=True, fetch_masks=True)
    loader = DataLoader(dset, batch_size=5, shuffle=True)
    images, animals, races, bboxes, masks = next(iter(loader)).values()
    fig, axes = plt.subplots(3, 5, figsize=(15,7))

    for idx in range(5):

        img = unnormalize(images[idx])
        
        # ax1: image
        ax1 = axes[0][idx]
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title(f"{dset.index2animal(animals[idx])} / {dset.index2race(races[idx])}")

        # ax2: image + bounding box
        ax2 = axes[1][idx]
        ax2.imshow(img)
        ax2.axis("off")
        
        xmin = bboxes[0][idx]
        ymin = bboxes[1][idx]
        xmax = bboxes[2][idx]
        ymax = bboxes[3][idx]
        
        rect = patches.Rectangle(
            (xmin, ymin),       # (xmin, ymin)
            xmax - xmin,        # width
            ymax - ymin,        # height
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        ax2.add_patch(rect)

        # ax3: segmentation mask
        ax3 = axes[2][idx]
        ax3.imshow(masks[idx].squeeze(), cmap="gray")
        ax3.axis("off")

    fig.supxlabel("Test: Some images with their bounding box and segmentation mask", size=15.5, style="italic")
    fig.subplots_adjust(hspace=0.05)
    fig.tight_layout()
    plt.show()

