# Segmentation d'Images Médicales 3D - Entraînement et Analyse

Ce répertoire contient des notebooks Jupyter pour l'entraînement et l'analyse de modèles de segmentation d'images médicales 3D sur des scanners CT.

## 📋 Contenu

- `train_unet.ipynb` - Notebook d'entraînement du modèle U-Net
- `segmentation_detection_analysis.ipynb` - Notebook d'évaluation et de visualisation du modèle

## 🎯 Aperçu

Ce projet implémente un modèle U-Net 2D pour la segmentation multi-organes à partir de scanners CT. Le modèle peut segmenter 117 structures anatomiques différentes incluant :
- Organes majeurs (cerveau, cœur, foie, reins, poumons, etc.)
- Structures squelettiques (vertèbres, côtes, etc.)
- Système vasculaire (aorte, artères, veines, etc.)
- Groupes musculaires

## 🔧 Prérequis

### Configuration Système
- Python 3.10+
- GPU compatible CUDA (recommandé) ou CPU
- 8Go+ de RAM (16Go+ recommandé)

### Dépendances Python
Toutes les dépendances sont listées dans `../requirements.txt` :
```
numpy
nibabel
scikit-image
torch
matplotlib
plotly
pandas
tqdm
scipy
jupyter
jupyterlab
ipywidgets
```

## 📦 Installation

1. **Installer les dépendances** (si pas encore installées) :
```bash
cd /home/hzhang02/dataset
pip3 install -r requirements.txt
```

2. **Configurer le noyau Jupyter** :
```bash
python3 -m ipykernel install --user --name=claude_env --display-name="Python 3 (claude_env)"
```

3. **Vérifier l'installation** :
```bash
python3 -c "import numpy, torch, nibabel, plotly; print('Tous les packages sont installés avec succès !')"
```

## 🚀 Démarrage Rapide

### Étape 1 : Entraîner le Modèle

1. Lancer JupyterLab :
```bash
cd /home/hzhang02/dataset/scripts
jupyter lab train_unet.ipynb
```

2. Sélectionner le noyau : **"Python 3 (claude_env)"**

3. Exécuter toutes les cellules pour :
   - Charger et préparer le jeu de données
   - Construire le modèle U-Net
   - Entraîner pour le nombre d'époques spécifié
   - Sauvegarder les points de contrôle dans `../outputs/`

**Configuration d'Entraînement** (ajustable dans la Section 6) :
- `EPOCHS = 5` - Nombre d'époques d'entraînement
- `BATCH_SIZE = 8` - Taille du batch pour l'entraînement
- `LEARNING_RATE = 1e-3` - Taux d'apprentissage
- `TARGET_SHAPE = (256, 256)` - Taille de l'image

**Sorties Attendues** :
- `outputs/label_map.json` - Cartographie des structures anatomiques vers les canaux
- `outputs/checkpoint_epochX.pth` - Points de contrôle du modèle
- `outputs/training_history.json` - Métriques d'entraînement
- `outputs/training_history.png` - Visualisation des courbes d'entraînement

### Étape 2 : Évaluer et Analyser

1. Lancer le notebook d'analyse :
```bash
jupyter lab segmentation_detection_analysis.ipynb
```

2. Sélectionner le noyau : **"Python 3 (claude_env)"**

3. Exécuter toutes les cellules pour :
   - Charger le modèle entraîné
   - Effectuer l'inférence sur les données de test
   - Calculer les métriques d'évaluation
   - Générer des visualisations

**Métriques d'Évaluation** :
- Coefficient de Dice
- IoU (Intersection sur Union)
- Taux de Réussite (à différents seuils)
- Analyse de performance par structure

**Sorties Générées** :
- `outputs/evaluation_results_*.csv` - Métriques détaillées par structure
- `outputs/metrics_distribution_*.png` - Distributions des métriques
- `outputs/structure_ranking_*.png` - Classements de performance
- `outputs/segmentation_visualization_*.png` - Résultats de segmentation 2D
- `outputs/3d_*.html` - Visualisations 3D interactives

## 📊 Structure du Jeu de Données

Structure de répertoire attendue :
```
/home/hzhang02/dataset/
├── s0000/
│   ├── ct.nii.gz                    # Scanner CT
│   └── segmentations/               # Masques de vérité terrain
│       ├── liver.nii.gz
│       ├── heart.nii.gz
│       ├── kidney_left.nii.gz
│       └── ... (117 structures)
├── s0001/
├── s0002/
└── ...
```

## 📈 Flux de Travail Typique

1. **Entraînement Initial** (2-5 époques pour tester) :
```bash
# Exécuter train_unet.ipynb avec EPOCHS=2
```

2. **Évaluation Rapide** :
```bash
# Exécuter segmentation_detection_analysis.ipynb
```

3. **Entraînement Complet** (si les résultats sont prometteurs) :
```bash
# Augmenter EPOCHS à 20-50 dans train_unet.ipynb
```

4. **Analyse Complète** :
```bash
# Ré-exécuter segmentation_detection_analysis.ipynb avec le meilleur point de contrôle
```

## 🎨 Exemples de Visualisation

Les notebooks génèrent diverses visualisations :

### Progression de l'Entraînement
- Courbes de perte sur les époques
- Tendances du score Dice de validation
- Prédictions d'échantillons vs. vérité terrain

### Résultats d'Évaluation
- Comparaisons de coupes 2D (CT + superposition)
- Rendus de maillage 3D (HTML interactif)
- Histogrammes de distribution de performance
- Graphiques de classement des structures

## 🔍 Dépannage

### Problème : "ModuleNotFoundError: No module named 'numpy'"
**Solution** : Assurez-vous de sélectionner le bon noyau :
- Dans Jupyter : Kernel → Change Kernel → "Python 3 (claude_env)"

### Problème : Mémoire CUDA saturée
**Solution** : Réduire la taille du batch dans le notebook d'entraînement :
```python
BATCH_SIZE = 4  # ou même 2
```

### Problème : L'entraînement est trop lent
**Solution** :
- Réduire la taille de l'image : `TARGET_SHAPE = (128, 128)`
- Réduire la complexité du modèle : `features=[16, 32, 64, 128]`
- Utiliser moins d'époques pour tester : `EPOCHS = 2`

### Problème : Scores Dice faibles
**Causes possibles** :
- Époques d'entraînement insuffisantes (essayer 20-50)
- Taux d'apprentissage trop élevé/faible (essayer 1e-4 ou 5e-4)
- Jeu de données trop petit (considérer l'augmentation de données)

## 📚 Architecture du Modèle

**U-Net 2D** :
- Encodeur : 4 blocs de sous-échantillonnage [32, 64, 128, 256 caractéristiques]
- Goulot d'étranglement : 512 caractéristiques
- Décodeur : 4 blocs de sur-échantillonnage avec connexions de saut
- Sortie : 117 canaux (un par structure anatomique)

**Fonction de Perte** : Entropie Croisée Binaire avec Logits (BCEWithLogitsLoss)

**Optimiseur** : Adam (lr=1e-3)

**Métrique d'Évaluation** : Coefficient de Dice

## 💡 Conseils pour de Meilleurs Résultats

1. **Augmentation de Données** : Considérer l'ajout de :
   - Rotation aléatoire (±15°)
   - Retournement aléatoire (horizontal/vertical)
   - Déformation élastique
   - Échelle d'intensité

2. **Entraînement Avancé** :
   - Planification du taux d'apprentissage (ReduceLROnPlateau)
   - Arrêt anticipé
   - Écrêtage du gradient
   - Entraînement en précision mixte (pour un entraînement GPU plus rapide)

3. **Améliorations du Modèle** :
   - Essayer U-Net 3D au lieu de 2D
   - Utiliser des mécanismes d'attention (Attention U-Net)
   - Expérimenter avec différentes fonctions de perte (Dice Loss, Focal Loss)

4. **Méthodes d'Ensemble** :
   - Entraîner plusieurs modèles avec différentes graines aléatoires
   - Moyenner les prédictions pour de meilleurs résultats

## 📖 Références

- **Jeu de Données** : TotalSegmentator v2.0.1
- **Modèle** : U-Net (Ronneberger et al., 2015)
- **Framework** : PyTorch 2.9.0

## 📝 Notes

- Le temps d'entraînement dépend du GPU/CPU et de la taille du jeu de données
- La première époque est généralement plus lente en raison du chargement des données
- Les points de contrôle sont sauvegardés après chaque époque (peuvent être de gros fichiers)
- La validation est effectuée après chaque époque d'entraînement

## 🤝 Contribution

Pour étendre ce projet :
1. Ajouter de nouvelles structures anatomiques à la carte des labels
2. Implémenter U-Net 3D pour un meilleur contexte spatial
3. Ajouter plus de métriques d'évaluation (distance de Hausdorff, distance de surface)
4. Intégrer avec des visualiseurs d'imagerie médicale (3D Slicer, ITK-SNAP)

## 📄 Licence

Ce projet est à des fins éducatives et de recherche.

---

**Créé** : Novembre 2025
**Dernière Mise à Jour** : Novembre 2025
