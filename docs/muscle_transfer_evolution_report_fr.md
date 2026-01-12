# Rapport d'Evolution des Methodes d'Apprentissage par Transfert Musculaire

## Apercu

Ce rapport documente l'evolution complete des methodes d'apprentissage par transfert musculaire de V1 a V3, incluant la conception, les details d'implementation, les problemes rencontres et les ameliorations apportees.

**Objectif Principal** : Utiliser les muscles paravertebraux deja etiquetes (autochthon, gluteus, iliopsoas) pour apprendre les caracteristiques musculaires, puis segmenter tous les tissus musculaires du corps, en particulier les muscles abdominaux non etiquetes.

---

## Phase 1 : V1 - Architecture Teacher-Student

### Philosophie de Conception

V1 a adopte une **architecture Teacher-Student**, dont l'idee centrale est :

```
Existant : Modele pre-entraine pour les muscles paravertebraux (Teacher)
Objectif : Entrainer un nouveau modele capable de segmenter tous les muscles (Student)
Methode : Le Student reste coherent avec le Teacher dans les zones connues, juge par lui-meme ailleurs
```

### Architecture du Reseau

**Modele Student : MuscleTransferNet**

```python
class MuscleTransferNet(nn.Module):
    """
    Entree : 3 canaux
        - Image CT
        - Carte de caracteristiques HU (dans la plage musculaire ou non)
        - Prediction du Teacher (zones musculaires connues)
    Sortie :
        - Segmentation musculaire du corps entier
        - Vecteur de caracteristiques musculaires (32 dimensions)
    """
```

| Composant | Description |
|-----------|-------------|
| Encodeur | 4 couches de sous-echantillonnage, canaux [32, 64, 128, 256] |
| Module d'attention | AttentionBlock, aide a se concentrer sur les caracteristiques musculaires |
| Decodeur | 4 couches de sur-echantillonnage, avec skip connections |
| Encodeur de caracteristiques | Extrait un vecteur de 32 dimensions du goulot |

### Fonction de Perte

```python
class MuscleTransferLoss:
    # Quatre termes de perte
    consistency = 2.0   # Coherence avec le Teacher
    hu_prior = 1.0      # Prior sur les valeurs HU
    coverage = 1.0      # Couverture (ne pas perdre les muscles connus)
    boundary = 0.5      # Lissage des contours
```

### Problemes Rencontres

**Probleme Fatal : Incompatibilite de Structure du Modele**

```
RuntimeError: Error(s) in loading state_dict for UNet2D:
    Unexpected key(s) in state_dict: "encoder.0.net.3.weight"...
```

Cause : Le modele original utilisait `DoubleConv` avec Dropout, tandis que le modele Teacher defini dans V1 n'avait pas de couche Dropout, causant une incompatibilite d'indices.

```python
# Modele original (avec Dropout)
nn.Sequential(
    nn.Conv2d(...),        # index 0
    nn.BatchNorm2d(...),   # index 1
    nn.ReLU(...),          # index 2
    nn.Dropout2d(...),     # index 3  ← Couche supplementaire!
    nn.Conv2d(...),        # index 4
    ...
)

# Teacher defini dans V1 (sans Dropout)
nn.Sequential(
    nn.Conv2d(...),        # index 0
    nn.BatchNorm2d(...),   # index 1
    nn.ReLU(...),          # index 2
    nn.Conv2d(...),        # index 3  ← Indices incompatibles
    ...
)
```

### Resume V1

| Element | Statut |
|---------|--------|
| Innovations | Architecture Teacher-Student, mecanisme d'attention, extraction de vecteurs |
| Probleme | Incompatibilite de structure, impossible de charger les poids |
| Resultat | Echec d'execution |

---

## Phase 2 : V2 - Utilisation Directe des Etiquettes TotalSegmentator

### Ameliorations de Conception

Face aux problemes de V1, V2 a apporte des changements fondamentaux :

```
V1 : Predictions du Teacher comme pseudo-etiquettes → Necessite le chargement du modele → Incompatibilite
V2 : Etiquettes reelles de TotalSegmentator → Aucune dependance aux modeles pre-entraines
```

### Modifications Principales

**1. Suppression de la Dependance au Modele Teacher**

```python
# V2 - Lecture directe des resultats de segmentation TotalSegmentator
KNOWN_MUSCLE_FILES = [
    'autochthon_left.nii.gz',
    'autochthon_right.nii.gz',
    'gluteus_maximus_left.nii.gz',
    'gluteus_maximus_right.nii.gz',
    'gluteus_medius_left.nii.gz',
    'gluteus_medius_right.nii.gz',
    'gluteus_minimus_left.nii.gz',
    'gluteus_minimus_right.nii.gz',
    'iliopsoas_left.nii.gz',
    'iliopsoas_right.nii.gz',
]
```

**2. Renforcement des Contraintes HU**

```python
# Definition des zones d'exclusion stricte
AIR_HU_MAX = -200      # Air, doit etre exclu
BONE_HU_MIN = 300      # Os, doit etre exclu

# Poids de perte
LABEL_WEIGHT = 2.0            # Coherence des zones etiquetees
HU_CONSTRAINT_WEIGHT = 3.0    # Contrainte HU
EXCLUSION_WEIGHT = 5.0        # Zone d'exclusion (le plus fort!)
BOUNDARY_WEIGHT = 0.3         # Lissage des contours
```

**3. Nouvelle Fonction de Perte**

```python
class MuscleTransferLossV2:
    def forward(self, pred_logits, label_mask, hu_slice, body_mask, exclusion_mask):
        # 1. Perte de coherence des etiquettes
        label_loss = BCE(pred, label) * weight

        # 2. Perte de zone d'exclusion (le plus important! Air/os doivent etre 0)
        exclusion_loss = (pred_prob * exclusion_mask).mean()

        # 3. Perte de contrainte HU (zones HU invalides ne peuvent pas etre predites comme muscles)
        hu_loss = (pred_prob * hu_invalid * body_mask).mean() * 10

        # 4. Perte de lissage des contours
        boundary_loss = TV(pred_prob)
```

**4. Filtrage de la Region Abdominale**

Le retour utilisateur indiquait de se concentrer sur la region abdominale, en excluant les cuisses :

```python
# Marqueurs d'organes abdominaux
ABDOMINAL_ORGANS = ['liver', 'spleen', 'kidney_left', 'kidney_right', ...]

# Marqueurs d'exclusion (cuisses)
EXCLUDE_MARKERS = ['femur_left', 'femur_right']

# Logique de filtrage
keep_slice = has_abdominal_organ AND NOT has_femur
```

Effet du filtrage :
- Ensemble d'entrainement : 6352/7695 coupes conservees (82.5%)
- Ensemble de validation : 2119/2540 coupes conservees (83.4%)

**5. Conception Simplifiee des Entrees (2 canaux)**

```python
input_tensor = np.stack([
    ct_resized,       # Canal 0 : Image CT
    muscle_resized,   # Canal 1 : Etiquettes musculaires connues
], axis=0)
```

### Resultats d'Entrainement

| Metrique | Valeur |
|----------|--------|
| Conformite HU | 99.5% |
| Score Dice | 0.80 |
| Perte d'exclusion | Proche de 0 |

### Problemes Identifies

**Modele trop conservateur** : Malgre de bons indicateurs, l'analyse pratique a revele que la zone musculaire predite etait insuffisante en couverture. Meme les zones musculaires deja etiquetees n'etaient pas entierement couvertes.

Retour utilisateur :
> "Je souhaite qu'il couvre plus de muscles, pas comme maintenant ou meme les muscles dans la base d'echantillons ne sont pas entierement couverts."

### Resume V2

| Element | Statut |
|---------|--------|
| Ameliorations | Suppression dependance Teacher, contraintes HU strictes, filtrage abdominal |
| Succes | Entrainement reussi, bons indicateurs |
| Probleme | Modele trop conservateur, couverture insuffisante |

---

## Phase 3 : V3 - Methode "Expansion puis Affinage"

### Philosophie de Conception

Face au probleme de couverture insuffisante de V2, l'utilisateur a propose l'approche "Expansion puis Affinage" :

```
Expansion : Couvrir tous les pixels dont les valeurs HU sont dans la plage musculaire (rappel eleve)
Affinage : Ajuster les contours par alignement avec les etiquettes connues (precision elevee)
```

Citation de l'utilisateur :
> "Tu peux d'abord definir une plage HU, puis couvrir tous les points de l'image CT dans cette plage. Ensuite, faire une legere reduction pour que ca corresponde aux contours des muscles deja echantillonnes."

### Modifications Principales

**1. Conception Etendue des Entrees (4 canaux)**

```python
input_tensor = np.stack([
    ct_resized,           # Canal 0 : Image CT
    hu_coarse,            # Canal 1 : Segmentation HU grossiere (tous les pixels HU dans -29~150) ← Nouveau!
    muscle_resized,       # Canal 2 : Etiquettes musculaires connues
    1 - exclusion         # Canal 3 : Masque de zone non-exclue ← Nouveau!
], axis=0)
```

**Raisons de conception** :
- `hu_coarse` indique au modele "ces pixels pourraient physiquement etre des muscles", encourageant l'expansion
- `1 - exclusion` marque clairement les zones impossibles pour les muscles

**2. Nouvelle Fonction de Perte : ExpandThenRefineLoss**

```python
class ExpandThenRefineLoss(nn.Module):
    """
    Trois termes de perte, implementant "Expansion puis Affinage" :
    1. label_alignment (poids 3.0) - Alignement precis des zones etiquetees
    2. coverage_reward (poids 1.0) - Encourager la couverture des zones HU valides non etiquetees
    3. exclusion (poids 5.0) - Exclusion stricte de l'air/os
    """

    def forward(self, pred_logits, label_mask, hu_coarse, exclusion_mask, body_mask):
        # 1. Perte d'alignement des etiquettes - assurer la precision des contours
        has_label = (label_mask > 0.5).float()
        label_region_loss = (BCE * has_label).sum() / has_label.sum()
        losses['label_alignment'] = label_region_loss * 3.0

        # 2. Recompense de couverture - encourager la couverture des zones HU valides non etiquetees (cle!)
        unlabeled_hu_valid = hu_coarse * (1 - has_label) * body_mask * (1 - exclusion_mask)
        coverage_loss = ((1 - pred_prob) * unlabeled_hu_valid).mean()
        losses['coverage_reward'] = coverage_loss * 1.0

        # 3. Perte de zone d'exclusion - interdiction absolue de predire l'air/os
        exclusion_loss = (pred_prob * exclusion_mask).mean()
        losses['exclusion'] = exclusion_loss * 5.0
```

**Comparaison des Fonctions de Perte** :

| Terme de perte | V2 | V3 |
|----------------|----|----|
| Alignement etiquettes | BCE sur tout | BCE uniquement sur zones etiquetees (poids 3.0) |
| Contrainte HU | Penalite zones HU invalides | Recompense couverture zones HU valides |
| Exclusion | Exclure air/os | Exclure air/os (identique) |
| Lissage contours | Perte TV | Supprime (depend de l'alignement) |

### Configuration V3

```python
# Hyperparametres
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
IMAGE_SIZE = 256

# Plage HU
HU_MIN = -29
HU_MAX = 150

# Seuils d'exclusion
AIR_THRESHOLD = -200
BONE_THRESHOLD = 300

# Poids de perte
LABEL_ALIGNMENT_WEIGHT = 3.0   # Alignement etiquettes (precision)
COVERAGE_REWARD_WEIGHT = 1.0   # Recompense couverture (rappel)
EXCLUSION_WEIGHT = 5.0         # Contrainte d'exclusion
```

### Etat de l'Entrainement

- **Heure de demarrage** : 2026-01-11 01:08
- **PID du processus** : 2333264
- **Donnees d'entrainement** : 6 352 coupes abdominales
- **Donnees de validation** : 2 119 coupes abdominales
- **WandB** : https://wandb.ai/haochen-zhang-insa-lyon/muscle-transfer-learning

### Effets Attendus

1. **Rappel plus eleve** : Couvrir davantage de regions musculaires
2. **Maintien de la precision des contours** : Via la perte label_alignment
3. **Eviter les faux positifs** : Via la perte d'exclusion

---

## Resume de l'Evolution des Versions

```
V1 : Architecture Teacher-Student
    ├─ Innovation : Mecanisme d'attention, extraction de vecteurs
    ├─ Probleme : Incompatibilite de structure du modele
    └─ Statut : Echec

         ↓ Suppression dependance Teacher, utilisation etiquettes reelles

V2 : Utilisation directe des etiquettes TotalSegmentator
    ├─ Ameliorations : Contraintes HU strictes, filtrage abdominal
    ├─ Resultats : Conformite HU 99.5%, Dice 0.80
    ├─ Probleme : Modele trop conservateur, couverture insuffisante
    └─ Statut : Succes mais insatisfaisant

         ↓ Ajout recompense de couverture, "Expansion puis Affinage"

V3 : Methode "Expansion puis Affinage"
    ├─ Ameliorations : 4 canaux d'entree, perte de recompense de couverture
    ├─ Objectif : Rappel eleve + contours precis
    └─ Statut : En cours d'entrainement
```

### Comparaison Technique

| Caracteristique | V1 | V2 | V3 |
|-----------------|----|----|-----|
| Source des etiquettes | Predictions du Teacher | Fichiers TotalSegmentator | Fichiers TotalSegmentator |
| Canaux d'entree | 3 (CT+carac HU+Teacher) | 2 (CT+etiquettes) | 4 (CT+HU grossier+etiquettes+non-exclusion) |
| Contrainte principale | Coherence Teacher | Contrainte HU stricte | Recompense couverture+alignement |
| Direction d'optimisation | Equilibree | Priorite precision | Priorite rappel |
| Filtrage abdominal | Non | Oui | Oui |
| Statut | Echec | Succes | En cours |

---

## Liste des Fichiers

| Fichier | Version | Description |
|---------|---------|-------------|
| `scripts/train_muscle_transfer.py` | V1 | Architecture Teacher-Student (obsolete) |
| `scripts/train_muscle_transfer_v2.py` | V2 | Etiquettes TotalSegmentator |
| `scripts/train_muscle_transfer_v3.py` | V3 | Methode "Expansion puis Affinage" |
| `outputs_muscle_transfer/` | V1 | Repertoire de sortie V1 |
| `outputs_muscle_transfer_v2/` | V2 | Repertoire de sortie V2 |
| `outputs_muscle_transfer_v3/` | V3 | Repertoire de sortie V3 |

---

*Date de generation du rapport : 2026-01-11*
