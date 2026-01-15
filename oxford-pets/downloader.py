import os
from pathlib import Path

import tarfile
import requests
from tqdm import tqdm

# Urls for Oxford-IIIT Pet Dataset 
URLS = {
    "images.tar.gz":      "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}

def get_data_directory() -> Path:
    
    # ----- Check if launched from root directory -----
    cwd = Path.cwd()
    if not (cwd / "install" / "environment.yml").exists():
        raise FileNotFoundError("Please launch this script from the root project folder !")

    # ----- Creating the data directory ---------------
    data_dir = cwd / "data"
    data_dir.mkdir(exist_ok=True)
    
    data_dir = data_dir / "oxford-pets"
    data_dir.mkdir(exist_ok=True)
    
    return data_dir

def download_archive(data_dir: Path, file: str, url: str) -> Path:
    tar_file = data_dir / file
    
    if tar_file.exists():
        print(f"{file} déja téléchargé dans {tar_file}, utilisation de l'archive directement.")
        
    else: # - downloading -
        print(f"téléchargement de {url} ...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with open(tar_file, 'wb') as f, tqdm(
            desc=tar_file.name, total=total, unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

        print(f"fichier {file} téléchargé dans: {data_dir}")
        
    return tar_file

def extract_archive(archive: Path, destination: Path):
    
    # - extracting -
    if not destination.exists():
        print(f"extraction de {archive.name} vers {destination}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=destination.parent)
    else:
        print(f"dataset déjà présent dans: {destination}")
        
    return destination

def remove_archive(archive: Path):
    archive.unlink()

def download_dataset():
    data_dir = get_data_directory()
    
    for file, url in URLS.items():
        destination = data_dir / file.removesuffix('.tar.gz')
        if destination.exists():
            print(f"dataset déjà présent dans: ./data/oxford-pets/{file.removesuffix('.tar.gz')}")
            continue
        
        archive = download_archive(data_dir, file, url)
        extract_archive(archive, destination)
        remove_archive(archive)
        

### Run from root with: python oxford-pets/downloader.py
if __name__ == "__main__":
    download_dataset()