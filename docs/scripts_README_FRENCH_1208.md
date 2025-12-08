# Segmentation d'Images Médicales 3D - Système d'Entraînement et d'Analyse (Version Améliorée)

Ce répertoire contient des scripts Python pour l'entraînement et l'analyse de modèles de segmentation d'images médicales 3D sur des scanners CT.

**Dernières Mises à Jour (2025-12-08)** :
- ✅ Ajout du mécanisme d'arrêt précoce (Early Stopping)
- ✅ Ajout de l'arrêt automatique basé sur un seuil de précision
- ✅ Sauvegarde automatique du meilleur modèle
- ✅ Surveillance et visualisation améliorées de l'entraînement

## 📋 Liste des Scripts

### Scripts d'Entraînement Principaux
- **`train_unet_enhanced.py`** - Script d'entraînement U-Net amélioré (Recommandé)
  - Support de l'arrêt précoce et du seuil de précision
  - Surveillance en temps réel (Weights & Biases)
  - Sauvegarde automatique du meilleur modèle
  - Enregistrement détaillé des métriques de performance

- **`train_unet.py`** - Script d'entraînement U-Net de base
  - Processus d'entraînement simple et direct
  - Adapté pour des expériences rapides

### Scripts d'Analyse et d'Évaluation
- **`segmentation_detection_analysis.py`** - Script d'évaluation et de visualisation du modèle
  - Calcul des métriques de performance
  - Visualisation des résultats
  - Analyse des erreurs

## 🎯 Aperçu du Projet

Ce projet implémente un modèle de segmentation multi-organes basé sur U-Net 2D, capable de segmenter **117 structures anatomiques différentes** à partir de scanners CT, incluant :

### Systèmes d'Organes Principaux
- **Système Digestif** : foie, estomac, pancréas, rate, côlon, intestin grêle, duodénum, vésicule biliaire, œsophage
- **Système Respiratoire** : lobes supérieurs/moyens/inférieurs des poumons gauche/droit, trachée
- **Système Circulatoire** : cœur, aorte, veine cave supérieure/inférieure, veine porte, veine pulmonaire et diverses branches vasculaires
- **Système Urogénital** : reins gauche/droit, kystes rénaux, prostate, vessie, glandes surrénales gauche/droit
- **Système Nerveux** : cerveau, moelle épinière
- **Système Endocrinien** : glande thyroïde

### Système Squelettique
- **Colonne Vertébrale** : vertèbres cervicales (C1-C7), thoraciques (T1-T12), lombaires (L1-L5), sacrées (S1), sacrum
- **Cage Thoracique** : côtes gauche/droit (1-12), sternum, clavicules, cartilages costaux
- **Membres** : humérus gauche/droit, fémurs, articulations de la hanche
- **Autres** : crâne, omoplates

### Système Musculaire
- Grand fessier, moyen fessier, petit fessier (gauche/droit)
- Iliopsoas (gauche/droit)
- Muscles du dos (gauche/droit)

## 🔧 Configuration Système Requise

### Matériel Requis
- **CPU** : Processeur multi-cœurs (8 cœurs ou plus recommandé)
- **GPU** : GPU NVIDIA compatible CUDA (recommandé)
  - Mémoire minimale : 8GB
  - Mémoire recommandée : 24GB ou plus
- **RAM** : 16GB ou plus (32GB ou plus recommandé)
- **Stockage** : Au moins 50GB d'espace disponible

### Logiciels Requis
- **Système d'Exploitation** : Linux / macOS / Windows
- **Python** : 3.10+
- **CUDA** : 11.0+ (si utilisation de GPU)

### Dépendances Python
Toutes les dépendances sont listées dans `requirements.txt` :
```
numpy>=1.21.0          # Calcul numérique
nibabel>=3.2.0         # Lecture/écriture de fichiers NIfTI
scikit-image>=0.19.0   # Traitement d'images
torch>=2.0.0           # Framework de deep learning
torchvision>=0.15.0    # Outils de vision par ordinateur
matplotlib>=3.5.0      # Visualisation 2D
plotly>=5.0.0          # Visualisation 3D interactive
pandas>=1.3.0          # Analyse de données
tqdm>=4.62.0           # Barres de progression
scipy>=1.7.0           # Calcul scientifique
wandb>=0.15.0          # Surveillance d'entraînement en temps réel
```

## 📦 Démarrage Rapide

### 1. Installation des Dépendances

```bash
cd /local/hzhang02/data/dataset
pip install -r requirements.txt
```

### 2. Préparation des Données

Assurez-vous que les données sont organisées selon la structure suivante :
```
/local/hzhang02/data/
├── s0000/
│   ├── ct.nii.gz
│   └── segmentations/
│       ├── liver.nii.gz
│       ├── heart.nii.gz
│       └── ... (117 fichiers de segmentation)
├── s0001/
│   ├── ct.nii.gz
│   └── segmentations/
│       └── ...
└── ... (plus de sujets)
```

### 3. Lancement de l'Entraînement

**Méthode Recommandée (script amélioré)** :
```bash
cd /local/hzhang02/data/dataset/scripts
python train_unet_enhanced.py
```

**Méthode de Base** :
```bash
python train_unet.py
```

## ⚙️ Explication de la Configuration

### Configuration de Base (lignes 374-380)

```python
DATA_ROOT = '/local/hzhang02/data'           # Répertoire racine des données
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs'  # Répertoire de sortie
TARGET_SHAPE = (256, 256)                    # Taille d'image
BATCH_SIZE = 16                              # Taille du batch
LEARNING_RATE = 1e-3                         # Taux d'apprentissage initial
EPOCHS = 20                                  # Nombre maximum d'époques
```

### Configuration de l'Arrêt Précoce (lignes 382-385) ⭐ Nouvelle Fonctionnalité

```python
USE_EARLY_STOPPING = True                    # Activer l'arrêt précoce
EARLY_STOP_PATIENCE = 5                      # Nombre d'époques de tolérance
EARLY_STOP_MIN_DELTA = 0.001                 # Seuil d'amélioration minimal
```

**Principe de Fonctionnement** :
- Surveille le coefficient Dice de validation
- Arrête automatiquement si l'amélioration est < 0.1% pendant 5 époques consécutives
- Prévient le surapprentissage et le gaspillage de ressources

**Paramètres Recommandés** :
- Expériences rapides : `PATIENCE = 3`
- Entraînement normal : `PATIENCE = 5` (par défaut)
- Ajustement fin : `PATIENCE = 7-10`

### Configuration du Seuil de Précision (lignes 387-390) ⭐ Nouvelle Fonctionnalité

```python
USE_ACCURACY_THRESHOLD = True                # Activer l'arrêt par seuil
ACCURACY_THRESHOLD = 0.93                    # Coefficient Dice cible
ACCURACY_THRESHOLD_PATIENCE = 2              # Époques de confirmation de stabilité
```

**Principe de Fonctionnement** :
- Commence le comptage lorsque Dice de validation ≥ 0.93
- Arrête l'entraînement après 2 époques consécutives atteignant le seuil
- Assure une performance stable et fiable

**Suggestions de Seuil** :
- Validation rapide : `0.85-0.90`
- Environnement de production : `0.93-0.95` (recommandé)
- Recherche d'excellence : `0.95-0.98`

### Configuration du Planificateur de Taux d'Apprentissage (lignes 392-401)

```python
USE_SCHEDULER = True                         # Activer le planificateur
SCHEDULER_TYPE = 'cosine'                    # 'cosine' ou 'plateau'

# Paramètres de recuit cosinus
COSINE_T_MAX = 20                            # Longueur de période
COSINE_ETA_MIN = 1e-6                        # Taux d'apprentissage minimal

# Paramètres de réduction adaptative
PLATEAU_FACTOR = 0.5                         # Facteur de décroissance
PLATEAU_PATIENCE = 3                         # Époques de tolérance
PLATEAU_MIN_LR = 1e-6                        # Taux d'apprentissage minimal
```

### Configuration Weights & Biases (lignes 403-407)

```python
USE_WANDB = True                             # Activer la surveillance en temps réel
WANDB_PROJECT = 'medical-segmentation-unet'  # Nom du projet
WANDB_RUN_NAME = 'unet-2d-training-enhanced' # Nom de l'exécution
```

## 🚀 Scénarios d'Utilisation

### Scénario 1 : Expérience Rapide (Économie de Temps)

```python
# Modifier la configuration
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3

USE_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 0.90
ACCURACY_THRESHOLD_PATIENCE = 1

BATCH_SIZE = 8  # Si mémoire GPU insuffisante
```

**Résultats Attendus** :
- Temps d'entraînement : environ 6-8 heures
- Arrêt prévu : Époque 8-10
- Adapté pour : validation initiale, itération rapide

### Scénario 2 : Entraînement Standard (Recommandé)

```python
# Utiliser la configuration par défaut
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 5

USE_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 0.93
ACCURACY_THRESHOLD_PATIENCE = 2

BATCH_SIZE = 16
```

**Résultats Attendus** :
- Temps d'entraînement : environ 7-9 jours
- Arrêt prévu : Époque 12-15
- Adapté pour : environnement de production, entraînement formel

### Scénario 3 : Recherche de Performance Optimale

```python
# Configuration haute qualité
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 7

USE_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 0.95
ACCURACY_THRESHOLD_PATIENCE = 3

SCHEDULER_TYPE = 'plateau'  # Ajustement plus flexible du taux d'apprentissage
```

**Résultats Attendus** :
- Temps d'entraînement : environ 10-13 jours
- Arrêt prévu : Époque 15-18
- Adapté pour : recherche scientifique, applications haute précision

## 📊 Surveillance de l'Entraînement

### Sortie Console

```
============================================================
Epoch 12/20
============================================================
Training: 100%|██████████| 1250/1250 [2:15:30<00:00]
Validation: 100%|██████████| 312/312 [0:25:15<00:00]

Epoch 12 Résultats:
  Perte d'entraînement: 0.0012 | Dice d'entraînement: 0.9450
  Perte de validation: 0.0018 | Dice de validation: 0.9315 | IoU: 0.9182
  Écart de surapprentissage: 0.0135 (normal)
  Norme du gradient: 0.0024
  Temps d'époque: 55432.5 secondes
  ✓ Amélioration du Dice de validation! Nouveau meilleur: 0.9315
  ✓ Seuil de précision atteint 0.9300! (2/2 époques)
  Taux d'apprentissage mis à jour: 0.00034567

  Top 5 des meilleures structures:
    liver: 0.9856
    spleen: 0.9782
    kidney_left: 0.9745
    kidney_right: 0.9723
    heart: 0.9698

  Top 5 des pires structures:
    rib_left_12: 0.7234
    vertebrae_C1: 0.7456
    thyroid_gland: 0.7589
    prostate: 0.7623
    adrenal_gland_left: 0.7701

============================================================
Arrêt par seuil de précision déclenché!
  Dice de validation atteint 0.9315 >= 0.9300
  Et maintenu stable pendant 2 époques
============================================================
```

### Surveillance en Temps Réel Weights & Biases

Après le démarrage de l'entraînement, visitez wandb.ai pour voir :
- Courbes d'entraînement/validation en temps réel
- Changements du taux d'apprentissage
- Surveillance de la norme du gradient
- Performance par classe
- Visualisation des prédictions d'échantillons
- Utilisation des ressources système

### Fichiers de Sortie

Le processus d'entraînement génère automatiquement les fichiers suivants :

```
outputs/
├── checkpoint_enhanced_epoch{N}.pth      # Point de contrôle par époque
├── best_model.pth                        # Meilleur modèle ⭐
├── training_history_enhanced.json        # Données d'historique d'entraînement
├── training_history_enhanced.png         # Graphiques de courbes d'entraînement
├── test_inference_enhanced.png           # Résultats d'inférence de test
├── sample_visualization_enhanced.png     # Visualisation d'échantillons
├── label_map.json                        # Mappage des étiquettes
├── ct_slices.png                         # Affichage de tranches CT
└── ct_mesh.html                          # Visualisation de maillage 3D
```

## 🔍 Évaluation du Modèle

### Chargement du Meilleur Modèle

```python
import torch
from scripts.train_unet_enhanced import UNet2D

# Charger le modèle
checkpoint = torch.load('outputs/best_model.pth')
model = UNet2D(in_ch=1, out_ch=117, features=[32, 64, 128, 256])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Meilleur coefficient Dice: {checkpoint['val_dice']:.4f}")
print(f"Meilleur IoU: {checkpoint['val_iou']:.4f}")
print(f"Nombre d'époques: {checkpoint['epoch']}")
```

### Métriques de Performance

Basé sur les données d'entraînement réelles :

| Métrique | Valeur Initiale | Valeur Finale | Amélioration |
|----------|----------------|---------------|--------------|
| Dice d'entraînement | 0.8200 | 0.9764 | +19.1% |
| Dice de validation | 0.8358 | 0.9317 | +11.5% |
| IoU de validation | 0.8356 | 0.9184 | +9.9% |
| Perte d'entraînement | 0.0604 | 0.0003 | -99.5% |

## ⏱️ Optimisation des Performances

### Économie de Temps

| Configuration | Époques Prévues | Durée d'Entraînement | Économie |
|---------------|----------------|---------------------|----------|
| Sans arrêt précoce | 20 | ~308 heures | 0% |
| Seuil 0.90 | 10-12 | ~170 heures | 45% |
| Seuil 0.93 | 12-15 | ~215 heures | 30% |
| Seuil 0.95 | 15-18 | ~260 heures | 15% |

### Astuces pour Accélérer l'Entraînement

1. **Augmenter la taille du batch** (si la mémoire le permet)
   ```python
   BATCH_SIZE = 32  # Augmenter de 16 à 32
   ```

2. **Utiliser l'entraînement en précision mixte**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

3. **Augmenter les threads de chargement de données**
   ```python
   num_workers=8  # Augmenter de 4 à 8
   ```

4. **Utiliser un planificateur de taux d'apprentissage plus rapide**
   ```python
   SCHEDULER_TYPE = 'plateau'  # Plus flexible que cosine
   ```

## 📚 Documentation Associée

- `docs/training_report_detailed.md` - Rapport d'entraînement détaillé
- `docs/training_issues_and_improvements.md` - Analyse des problèmes et suggestions d'amélioration
- `docs/early_stopping_guide.md` - Guide d'utilisation du mécanisme d'arrêt précoce
- `docs/MODEL_GUIDE_FRENCH_1114.md` - Explication détaillée de l'architecture du modèle

## 🔧 Dépannage

### Problème 1 : Mémoire GPU Insuffisante (Out of Memory)

**Symptôme** : `RuntimeError: CUDA out of memory`

**Solutions** :
```python
BATCH_SIZE = 8  # Réduire la taille du batch
TARGET_SHAPE = (128, 128)  # Réduire la taille d'image
```

### Problème 2 : Arrêt Prématuré de l'Entraînement

**Symptôme** : Arrêt après seulement 2-3 époques

**Solutions** :
```python
USE_ACCURACY_THRESHOLD = False  # Désactiver temporairement le seuil
# Ou
ACCURACY_THRESHOLD = 0.95  # Augmenter le seuil
ACCURACY_THRESHOLD_PATIENCE = 3  # Augmenter les époques de confirmation
```

### Problème 3 : Vitesse d'Entraînement Lente

**Symptôme** : Chaque époque prend beaucoup de temps

**Causes Possibles et Solutions** :
- Chargement lent des données : Augmenter `num_workers`
- Goulot d'étranglement CPU : Réduire les opérations d'augmentation de données
- I/O disque lent : Copier les données sur SSD
- Faible utilisation GPU : Augmenter `BATCH_SIZE`

### Problème 4 : Échec de Connexion à Weights & Biases

**Solutions** :
```python
USE_WANDB = False  # Désactiver wandb
# Ou connexion manuelle
import wandb
wandb.login(key='your-api-key')
```

### Problème 5 : Données Introuvables

**Symptôme** : `Trouvé 0 sujets`

**Solutions** :
```python
# Vérifier le chemin des données
DATA_ROOT = '/local/hzhang02/data'  # Assurer que le chemin est correct
# Vérifier le nommage des dossiers (doit commencer par 's', ex: s0000, s0001)
```

## 💡 Meilleures Pratiques

### 1. Liste de Vérification Avant l'Entraînement

- [ ] Chemin des données correct
- [ ] Mémoire GPU suffisante (au moins 8GB)
- [ ] Espace disque suffisant (au moins 50GB)
- [ ] Configuration d'arrêt précoce et de seuil raisonnable
- [ ] Weights & Biases configuré (optionnel)

### 2. Surveillance Pendant l'Entraînement

- [ ] Consulter régulièrement les courbes wandb
- [ ] Surveiller l'utilisation du GPU
- [ ] Vérifier l'espace disque
- [ ] Observer la tendance du Dice de validation

### 3. Analyse Après l'Entraînement

- [ ] Consulter les graphiques d'historique d'entraînement
- [ ] Analyser les meilleures/pires classes
- [ ] Vérifier le surapprentissage
- [ ] Sauvegarder le meilleur modèle

## 📞 Support Technique

En cas de problème, vérifiez :
1. Documentation pertinente (répertoire `docs/`)
2. Sortie des logs d'entraînement
3. Rapport Weights & Biases
4. Utilisation GPU/mémoire

## 📄 Licence

Ce projet est destiné uniquement à la recherche académique.

---

**Dernière Mise à Jour** : 2025-12-08
**Version** : v2.0 (Version Améliorée)
**Mainteneur** : hzhang02
