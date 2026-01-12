# Rapport de Modification - Apprentissage par Transfert Musculaire V2

## Apercu

Ce rapport documente le processus complet de mise a niveau de la version V1 vers la version V2, incluant l'analyse des problemes, la conception et l'implementation.

## Contexte et Problematique

### Problemes de la Version V1

La version V1 utilisait une architecture Teacher-Student, avec un modele musculaire pre-entraine comme Teacher pour generer des pseudo-etiquettes. Cependant, des problemes serieux ont ete rencontres lors de l'execution :

**Incompatibilite de structure du modele** : Le modele UNet2D original entraine ne correspondait pas a la structure du modele Teacher definie dans le code V1, empechant le chargement correct des poids.

```
Exemple de message d'erreur :
Unexpected key(s) in state_dict: "encoder.0.net.3.weight"...
```

La cause etait que le module `DoubleConv` du modele original contenait des couches Dropout, absentes dans le code V1.

### Defauts de Conception

1. **Dependance aux predictions du modele** : Utilisation des predictions du modele comme pseudo-etiquettes, risquant de propager des erreurs
2. **Architecture Teacher-Student complexe** : Augmentation de la difficulte de debogage
3. **Manque de contraintes strictes** : Mecanisme d'exclusion insuffisant pour l'air et les os

## Solution V2

### Ameliorations Principales

```
Utilisation directe des etiquettes reelles de TotalSegmentator
↓
Plus de dependance aux predictions du modele pre-entraine
↓
Ajout de contraintes strictes sur les valeurs HU
↓
Ajout du filtrage de la region abdominale
```

### Comparaison des Methodes

| Caracteristique | Version V1 | Version V2 |
|-----------------|------------|------------|
| Source des etiquettes | Predictions du modele pre-entraine (pseudo-etiquettes) | Segmentations reelles de TotalSegmentator |
| Architecture | Teacher-Student | Modele Student unique |
| Contraintes HU | Contraintes souples | Contraintes strictes (air/os doivent etre exclus) |
| Filtrage de region | Aucun | Filtrage abdominal (exclusion des cuisses) |

## Modifications Detaillees

### 1. Suppression de la Dependance au Modele Teacher

V1 necessitait le chargement du modele pre-entraine :
```python
# V1 - Necessite le chargement du modele Teacher
teacher = DoubleConvTeacher(...)  # Structure doit correspondre exactement
teacher.load_state_dict(checkpoint['model_state_dict'])
```

V2 utilise directement les fichiers d'etiquettes :
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

### 2. Renforcement des Contraintes HU

```python
# Definition des zones d'exclusion stricte
AIR_HU_MAX = -200      # Air/arriere-plan, doit etre exclu
BONE_HU_MIN = 300      # Os, doit etre exclu

# Configuration des poids de perte
LABEL_WEIGHT = 2.0            # Coherence des zones etiquetees
HU_CONSTRAINT_WEIGHT = 3.0    # Poids de contrainte HU
EXCLUSION_WEIGHT = 5.0        # Contrainte de zone d'exclusion (la plus forte!)
BOUNDARY_WEIGHT = 0.3         # Lissage des contours
```

### 3. Nouvelle Conception de la Fonction de Perte

```python
class MuscleTransferLossV2(nn.Module):
    """
    Quatre termes de perte :
    1. label - Coherence des zones etiquetees
    2. exclusion - Air/os doivent predire 0
    3. hu_constraint - Contrainte de plage HU
    4. boundary - Lissage des contours
    """

    def forward(self, pred_logits, label_mask, hu_slice, body_mask, exclusion_mask):
        # 1. Perte de coherence des etiquettes
        label_loss = (bce_loss * weight * body_mask).sum() / (body_mask.sum() + 1e-6)

        # 2. Perte de zone d'exclusion (la plus importante!)
        exclusion_loss = (pred_prob * exclusion_mask).mean()

        # 3. Perte de contrainte HU
        hu_violation = pred_prob * hu_invalid * body_mask * (1 - exclusion_mask)
        hu_loss = hu_violation.mean() * 10

        # 4. Perte de lissage des contours
        tv_loss = tv_h + tv_w
```

### 4. Filtrage de la Region Abdominale

Le retour utilisateur indiquait que le modele devait se concentrer sur la region abdominale plutot que sur les cuisses. V2 a ajoute un filtrage anatomique :

```python
# Marqueurs d'organes abdominaux (les coupes avec ces organes sont abdominales)
ABDOMINAL_ORGANS = [
    'liver.nii.gz',
    'spleen.nii.gz',
    'kidney_left.nii.gz',
    'kidney_right.nii.gz',
    'pancreas.nii.gz',
    'stomach.nii.gz',
    'colon.nii.gz',
    'small_bowel.nii.gz',
]

# Marqueurs d'exclusion (les coupes avec ces os sont des cuisses)
EXCLUDE_MARKERS = [
    'femur_left.nii.gz',
    'femur_right.nii.gz',
]

# Logique de filtrage
has_abdominal = any(organ present in slice)
has_thigh = any(femur present in slice)
keep_slice = has_abdominal and not has_thigh
```

**Effet du filtrage** :
- Ensemble d'entrainement : 6352/7695 coupes conservees (82.5%)
- Ensemble de validation : 2119/2540 coupes conservees (83.4%)

### 5. Conception des Entrees (2 canaux)

```python
input_tensor = np.stack([
    ct_resized,           # Canal 0: Image CT (normalisee)
    muscle_resized,       # Canal 1: Etiquettes musculaires connues
], axis=0)
```

### 6. Integration WandB

Ajout de la journalisation complete avec WandB :
- Courbes de perte entrainement/validation
- Suivi des composantes de perte
- Metriques de validation (Dice, conformite HU)
- Visualisation periodique des echantillons

## Resultats d'Entrainement

L'entrainement V2 a obtenu de bons indicateurs sur l'ensemble de validation :

| Metrique | Valeur |
|----------|--------|
| Conformite HU | 99.5% |
| Score Dice | 0.80 |
| Perte etiquette | Diminution stable |
| Perte exclusion | Proche de 0 |

## Problemes Identifies (menant a V3)

Malgre les bons indicateurs de V2, l'analyse pratique a revele :

**Modele trop conservateur** : La zone musculaire predite etait insuffisante en couverture. Meme les zones musculaires deja etiquetees n'etaient pas entierement couvertes.

Ce probleme a motive la conception de la methode V3 "Expansion puis Affinage".

## Modifications de Fichiers

| Fichier | Operation | Description |
|---------|-----------|-------------|
| `scripts/train_muscle_transfer_v2.py` | Nouveau | Implementation complete V2 |
| `.gitignore` | Modifie | Ajout des regles pour `outputs_muscle_transfer_v2/` |
| `outputs_muscle_transfer_v2/` | Nouveau | Repertoire de sortie pour l'entrainement V2 |

## Nettoyage de la Dette Technique

V2 a egalement corrige plusieurs problemes techniques de V1 :

1. **Erreur multiprocessus CUDA** : Configuration `num_workers=0` pour resoudre "Cannot re-initialize CUDA in forked subprocess"
2. **Erreur de serialisation JSON** : Conversion des types numpy en types natifs Python
3. **Correspondance de structure de modele** : Plus besoin de correspondre a la structure du modele pre-entraine

---

*Date de generation du rapport : 2026-01-11*
