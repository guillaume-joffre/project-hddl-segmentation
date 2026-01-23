# Projets d'HDDL

Les trois mini-projets d'hddl sont diponibles dans les dossiers correspondants, avec:
- mini-project-1: Chien/Chats (classification et segmentation)
- mini-project-2: Conditional VAE
- mini-project-3: CNN vs ViT

## Installation:

Pour installer l'environnement:
> conda env create -f install/environment.yml --verbose

Puis pour l'activer:
> conda activate HDDL.1

Enfin pour rendre l'environnement visible pour marimo/jupyter:
> python -m ipykernel install --user --name HDDL.1 --display-name "HDDL.1"
