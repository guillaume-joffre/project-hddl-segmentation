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

def download_dataset():
    
    # ----- Check if launched from root directory -----
    cwd = Path.cwd()
    if not (cwd / "install" / "environment.yml").exists():
        raise FileNotFoundError("Please launch this function from the root project folder !")

    # ----- Creating the data directory ---------------
    data_dir = cwd / "data/oxford-pets"
    data_dir.mkdir(exist_ok=True)
    
    # ----- Downloading and extracting ----------------
    for file, url in URLS.items():
        tar_file = data_dir / file
        extract_dir = data_dir / file.removesuffix('.tar.gz')

        if extract_dir.exists():
            print(f"dataset déjà présent dans: ./data/oxford-pets/{file.removesuffix('.tar.gz')}")
            continue
        
        if tar_file.exists():
            print(f"{file} déja téléchargé dans {tar_file}, utilisation du cache.")
            
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
            
        # - extracting -
        if not extract_dir.exists():
            print(f"extraction de {tar_file.name} vers {extract_dir}")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=data_dir)
        else:
            print(f"dataset déjà présent dans: {extract_dir}")
        
if __name__ == "__main__":
    download_dataset()