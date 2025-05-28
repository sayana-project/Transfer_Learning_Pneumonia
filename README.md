# Détection de la Pneumonie à partir de Radios Thoraciques

![ courbe Précision rappel](image/intro.jpeg)

## 1. Introduction

Ce projet vise à développer un modèle de classification binaire capable de détecter des cas de pneumonie à partir d'images de radios thoraciques. À partir du dataset *Chest X-Ray* fourni par Kaggle et d’un modèle CNN préentraîné, le projet applique les techniques de transfert learning. L’ensemble du processus est documenté dans un notebook clair, avec des visualisations et des outils de suivi comme MLflow pour tracer les expérimentations.

## 2. Objectif du projet

Détecter automatiquement les cas de pneumonie à partir d’images médicales (radiographies thoraciques) en exploitant le transfert learning sur un modèle CNN préentraîné.

## 3. Installation

### a. Cloner le projet :
```bash
git clone https://github.com/sayana-project/Transfer_Learning_Pneumonia.git
```

### b. Créer un environnement virtuel :
```bash
python -m venv .venv
```

### c. Activer le venv :
- Windows : `.\.venv\Scriptsactivate`
- macOS/Linux : `source .venv/bin/activate`

### d. Installer les dépendances :
```bash
pip install -r requirements.txt
```
### f. lancer le serveur local mlflow :
```bash
mlflow server --host 127.0.0.1 --port 8080
```


## 4. Bibliothèques utilisées

- `numpy`, `matplotlib`, `seaborn`
- `tensorflow`, `keras`
- `opencv-python`
- `sklearn`
- `mlflow`
- `pathlib`, `os`, `cv2`,`random`

## 5. Structure du projet

```
.
├── data/chest_xray
├       ├──test/train/val                        # Images (train/test/val)
├── targeimg                                     # Image target
├── image                                        # Image courbe/graphique
├── notebook_jupyter_transfer_learning_CNN.ipynb # Notebook principal
├── mlruns/                                      # test model MLflow
├── requirements.txt
└── README.md
```

## 6. Description du modèle

- Modèle de base : `DenseNet121`
- Couches personnalisées : GlobalAveragePooling2D, BatchNormalization, Dense
- Optimiseur : `Adam`
- Fonction de perte : `binary_crossentropy`
- Activation finale : `sigmoid`

## 7. Résultats
Résultat de notebook_jupyter_transfer_learning_CNN2.ipynb
- Accuracy validation/test : **100%**
-courbe precisionet perte
![ courbe precision &perte](image/courbe_precision_perte2.png)
- Matrice de confusion
![ Matrice de confusion](image/confusion_matrix2.png)
- Courbe ROC
![ courbe ROC](image/roc_curve2.png)
- Courbe précision-rappel
![ courbe Précision rappel](image/precision_recall_curve2.png)
- Rapport de classification (`sklearn`)
```
              precision    recall  f1-score   support

      NORMAL       0.84      0.89      0.87       234
   PNEUMONIA       0.93      0.90      0.92       390

    accuracy                           0.90       624
   macro avg       0.89      0.90      0.89       624
weighted avg       0.90      0.90      0.90       624
```

## 8. Suivi avec MLflow

- Suivi automatique : pertes, métriques, paramètres.
- Sauvegarde des modèles.
- Tracking avec `mlflow.tensorflow.autolog()`

## 9.🔍 How to Use – Prédiction d'une Image Unique
Ce projet vous permet de prédire automatiquement si une radiographie thoracique contient des signes de pneumonie ou non à l'aide d'un modèle CNN pré-entraîné.

Étapes à suivre :
1. Placez une seule image au format JPG/PNG dans le dossier targetimg/.

Exemple : targetimg/radio_test1.jpeg

2. Exécutez la cellule dans votre notebook Jupyter en dessous du titre TARGET 

3. Le résultat s'affiche directement avec :

Le nom de l’image,

Le diagnostic prédit (NORMAL ou PNEUMONIA),

Le score de confiance du modèle.

## 10. À améliorer

- Déploiement API Flask.
- Ajustement des callbacks.
- Data augmentation plus avancée.
- Amélioration de la performance. (ROC/AUC et matrice de confusion)

## 11. Dataset

- Source : [Kaggle - Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Model : [Modéle CNN](https://keras.io/api/applications/)
- Trois répertoires : `train`, `val`, `test`
- Deux classes : `NORMAL`, `PNEUMONIA`

## 12. Auteur

Projet personnel mené dans un cadre d’apprentissage du deep learning appliqué à la santé.  
**@sayana-project**
