# ProjetIA_m1_s2
Réalisation d'un arbitre virtuel pour force athlétique dans le cadre de compétitions professionnelles nationales. Ce logiciel permet d'analyser et d'enregistrer les mouvements des athlètes grâce à l'utilisation d'un modèle de détection de la barre et de tracking de personnes.


## Conda Environemnt
Utilisation d'un envrionnement conda pour les dépendances :

**Utilisation:**
conda activate envProjectIA

**Création:**
conda env create -f env.yml  #Créer un nouvel environnement

**Suppression :**
conda env remove -n envProjectIA  #Supprimer l'environnement existant

**Update de l’environnement :**
conda env update -f env.yml --prune #Prune permet de supprimer les dépendances qui ne sont plus nécessaires

## Utilisation du logiciel

'''python3 GUI.py''' pour lancer le programme


## AWS :
Utilisation d'AWS S3. Il est intégrer au sein du logiciel avec la possibilité de rentrer vos crédentials afin d'accéder aux vidéos d'entraînements.

**À savoir (from aws website):**

AWS Free Tier : 
As part of the AWS Free Tier, you can get started with Amazon S3 for free. Upon sign-up, new AWS customers receive 5GB of Amazon S3 storage in the S3 Standard storage class; 20,000 GET Requests; 2,000 PUT, COPY, POST, or LIST Requests; and 100 GB of Data Transfer Out each month.

Try out the AWS Pricing Calculator : https://calculator.aws/#/
https://aws.amazon.com/s3/pricing/?loc=ft#AWS_Free_Tier
# Projet-IAtheles
