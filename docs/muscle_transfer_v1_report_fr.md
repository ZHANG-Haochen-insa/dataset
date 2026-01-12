# Rapport sur la Methode V1 d'Apprentissage par Transfert Musculaire

## Apercu

Ce rapport documente la conception, l'implementation et les problemes rencontres dans la version V1 de l'apprentissage par transfert musculaire. V1 etait la premiere tentative d'utilisation de l'apprentissage par transfert pour segmenter les muscles du corps entier.

## Concept Fondamental

V1 a adopte une **architecture Teacher-Student**, dont l'idee centrale est :

```
Existant : Modele de segmentation musculaire pre-entraine pour les muscles paravertebraux (Teacher)
Objectif : Entrainer un nouveau modele capable de segmenter tous les muscles du corps (Student)
Methode : Le Student reste coherent avec le Teacher dans les zones connues, et juge par lui-meme dans les zones inconnues
```

### Philosophie de Conception

1. **Du local au global** : Extension d'un modele de muscles paravertebraux a tout le corps
2. **Transfert de caracteristiques** : Apprendre les "caracteristiques musculaires" (distribution HU, texture, etc.)
3. **Extension auto-supervisee** : Segmentation autonome dans les zones non etiquetees

## Architecture du Reseau

### Modele Student : MuscleTransferNet

```python
class MuscleTransferNet(nn.Module):
    """
    Reseau de transfert de caracteristiques musculaires

    Entree : 3 canaux
        - Image CT (1 canal)
        - Carte de caracteristiques HU (1 canal, indiquant si dans la plage musculaire)
        - Prediction du Teacher (1 canal, zones musculaires connues)
    Sortie :
        - Segmentation musculaire du corps entier (1 canal)
        - Vecteur de caracteristiques musculaires (32 dimensions, pour analyse)
    """
```

**Caracteristiques de l'architecture** :

| Composant | Description |
|-----------|-------------|
| Encodeur | 4 couches de sous-echantillonnage, canaux [32, 64, 128, 256] |
| Module d'attention | AttentionBlock, aide le modele a se concentrer sur les caracteristiques musculaires |
| Decodeur | 4 couches de sur-echantillonnage, avec skip connections |
| Encodeur de caracteristiques | Extrait un vecteur de 32 dimensions du goulot d'etranglement |
| Dropout | Encodeur 0.1, goulot 0.2 |

### Modele Teacher : UNet2D

```python
class UNet2D(nn.Module):
    """Structure du modele Teacher (identique a l'entrainement original)"""
    # Structure UNet standard, sans Dropout
    # Entree : 1 canal CT
    # Sortie : Segmentation musculaire multi-canaux
```

## Conception de la Fonction de Perte

```python
class MuscleTransferLoss(nn.Module):
    """
    Quatre termes de perte :
    1. consistency - Coherence avec le Teacher
    2. hu_prior - Prior sur les valeurs HU
    3. coverage - Couverture des zones connues
    4. boundary - Lissage des contours
    """
```

### Configuration des Poids de Perte

| Terme de perte | Poids | Fonction |
|----------------|-------|----------|
| consistency | 2.0 | Coherence avec le Teacher dans ses zones de prediction |
| hu_prior | 1.0 | Valeurs HU dans la plage pour les zones predites comme muscles |
| coverage | 1.0 | Ne pas perdre les muscles predits par le Teacher |
| boundary | 0.5 | Contrainte de lissage des contours |

### Details de la Fonction de Perte

```python
def forward(self, pred_logits, teacher_mask, hu_slice, body_mask):
    # 1. Perte de coherence - Poids plus eleve dans les zones du Teacher
    weight_map = 1 + teacher_region * 2.0  # Poids 3x dans les zones Teacher
    consistency_loss = (bce_loss * weight_map * body_mask).sum() / body_mask.sum()

    # 2. Perte de prior HU - Penalise les predictions musculaires hors plage HU
    hu_violation = pred_prob * (1 - hu_in_range) * body_mask
    hu_prior_loss = hu_violation.mean()

    # 3. Perte de couverture - Assure la couverture des muscles connus
    coverage = (pred_prob * teacher_region).sum() / teacher_region.sum()
    coverage_loss = 1 - coverage

    # 4. Perte de lissage des contours - Variation totale (TV)
    tv_h = |pred[:,:,1:,:] - pred[:,:,:-1,:]|
    tv_w = |pred[:,:,:,1:] - pred[:,:,:,:-1]|
    boundary_loss = tv_h + tv_w
```

## Conception du Dataset

### Composition des Entrees (3 canaux)

```python
input_tensor = np.stack([
    ct_resized,           # Canal 0 : Image CT normalisee
    hu_muscle_feature,    # Canal 1 : Carte HU (dans la plage musculaire ou non)
    teacher_pred,         # Canal 2 : Prediction du modele Teacher
], axis=0)
```

### Obtention des Predictions du Teacher

```python
def _get_teacher_prediction(self, ct_normalized, original_shape):
    """Obtenir la prediction du modele Teacher"""
    img_t = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = torch.sigmoid(self.teacher_model(img_t))
        # Combiner toutes les categories musculaires
        pred_combined = (pred[0].sum(dim=0) > 0.5).float()

    return pred_combined
```

## Metriques d'Evaluation

```python
def compute_metrics(pred, teacher, hu_slice, body_mask):
    """
    Cinq metriques d'evaluation :
    1. teacher_coverage - Taux de couverture des zones Teacher
    2. teacher_dice - Score Dice avec les predictions Teacher
    3. hu_compliance - Taux de conformite HU
    4. expansion_ratio - Ratio d'expansion par rapport au Teacher
    5. overflow_rate - Taux de debordement vers les zones non-musculaires
    """
```

## Definition des Plages de Valeurs HU

```python
# Plage HU des muscles
MUSCLE_HU_MIN = -29
MUSCLE_HU_MAX = 150
MUSCLE_HU_OPTIMAL_MIN = 0   # Muscle typique
MUSCLE_HU_OPTIMAL_MAX = 100

# Autres tissus
BONE_HU_MIN = 200   # Os
FAT_HU_MIN = -190   # Graisse
FAT_HU_MAX = -30
AIR_HU_MAX = -500   # Air
```

## Configuration d'Entrainement

```python
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS = 30

# Configuration de visualisation
VIS_EVERY_N_EPOCHS = 2
VIS_NUM_SAMPLES = 4
```

## Problemes Rencontres

### 1. Incompatibilite de Structure du Modele (Probleme Fatal)

**Description du probleme** :
Lors du chargement du modele Teacher pre-entraine, une erreur d'incompatibilite des poids s'est produite :

```
RuntimeError: Error(s) in loading state_dict for UNet2D:
    Unexpected key(s) in state_dict: "encoder.0.net.3.weight"...
```

**Analyse de la cause** :
Le modele original utilisait `DoubleConv` avec Dropout :

```python
# Structure du modele original
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # Dropout ici!
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
```

Le modele Teacher defini dans le code V1 n'avait pas de couche Dropout, causant une incompatibilite d'indices de couches.

### 2. Dependance aux Predictions du Modele comme Pseudo-etiquettes

**Probleme** : L'utilisation des predictions du modele Teacher comme signal de supervision peut propager des erreurs.

### 3. Manque de Contraintes Strictes d'Exclusion

**Probleme** : Mecanisme d'exclusion insuffisant pour l'air et les os, pouvant entrainer des faux positifs.

### 4. Complexite Architecturale Elevee

**Probleme** :
- Necessite de maintenir simultanement deux modeles Teacher et Student
- L'inference du Teacher est requise lors de la construction du dataset, augmentant le temps de traitement
- Difficulte de debogage accrue

## Directions d'Amelioration (menant a V2)

1. **Suppression de la dependance au modele Teacher** : Utilisation directe des etiquettes reelles de TotalSegmentator
2. **Ajout de contraintes strictes** : Penalites fortes pour l'air et les os
3. **Simplification de l'architecture** : Entrainement d'un seul modele Student
4. **Ajout du filtrage de region** : Concentration sur la region abdominale, exclusion des cuisses

## Informations sur le Fichier

| Attribut | Valeur |
|----------|--------|
| Chemin du fichier | `scripts/train_muscle_transfer.py` |
| Date de creation | 2026-01-10 |
| Lignes de code | ~700 lignes |
| Statut | Remplace par V2 |

## Resume

La version V1 a propose un cadre innovant d'apprentissage par transfert Teacher-Student, mais n'a pas pu fonctionner avec succes en raison du probleme d'incompatibilite de structure du modele. Ce probleme a motive la conception de la version V2, qui utilise directement les etiquettes TotalSegmentator au lieu des predictions du modele, simplifiant ainsi l'ensemble du processus.

Bien que V1 n'ait pas pu fonctionner avec succes, ses idees de conception (du local au global, transfert de caracteristiques, extension auto-supervisee) ont pose les bases des versions suivantes.

---

*Date de generation du rapport : 2026-01-11*
