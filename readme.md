## Installation:

Pour installer l'environnement faites tourner la commande:
> conda env create -f install/environment.yml --verbose

Puis activez l'env avec:
> conda activate HDDL.1

Enfin rendez l'environnement visible pour marimo/jupyter avec:
> python -m ipykernel install --user --name HDDL.1 --display-name "HDDL.1 Segmentation"

Si vous avez pas un GPU nvidia vous aurez pas cuda et ce 
sera plus lent mais au pire on se débrouillera comme ça
avec ce qu'on a. 