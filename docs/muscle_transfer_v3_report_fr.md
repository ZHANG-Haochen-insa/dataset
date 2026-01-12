# Rapport de Modification - Apprentissage par Transfert Musculaire V3

## Apercu

Ce rapport documente le processus complet de mise a niveau de la version V2 vers la version V3 avec l'approche "Expansion puis Affinage", incluant l'analyse des problemes, la conception et l'implementation.

## Contexte et Problematique

### Limitations de la Version V2

La version V2 de l'apprentissage par transfert a obtenu de bons indicateurs sur l'ensemble de validation (conformite HU 99.5%, Dice 0.80), mais l'analyse pratique a revele un probleme fondamental :

**Modele trop conservateur** : La zone musculaire predite par le modele etait insuffisante en termes de couverture. Meme les regions musculaires deja etiquetees dans les echantillons d'entrainement n'etaient pas entierement couvertes. Cela va a l'encontre de notre objectif final : decouvrir et segmenter tous les tissus musculaires.

### Besoins de l'Utilisateur

L'utilisateur a clairement indique que le modele devrait :
1. **Couvrir davantage de regions musculaires**, sans se limiter aux parties deja etiquetees
2. Determiner d'abord toutes les zones candidates par la plage de valeurs HU
3. Puis ajuster precisement les contours par alignement avec les etiquettes connues

## Conception : Methode "Expansion puis Affinage"

### Concept Fondamental

```
Expansion : Couvrir tous les pixels dont les valeurs HU sont dans la plage musculaire (rappel eleve)
Affinage : Ajuster les contours par alignement avec les etiquettes connues (precision elevee)
```

### Comparaison des Methodes

| Caracteristique | Version V2 | Version V3 |
|-----------------|------------|------------|
| Canaux d'entree | 2 (CT + etiquettes connues) | 4 (CT + segmentation HU grossiere + etiquettes connues + zone non-exclue) |
| Conception des pertes | BCE + conformite HU + Dice | Alignement des etiquettes + recompense de couverture + penalite d'exclusion |
| Objectif | Correspondance precise avec les etiquettes connues | Maximiser la couverture tout en maintenant la precision des contours |
| Direction d'optimisation | Priorite a la precision | Priorite au rappel, contours precis |

## Modifications Detaillees

### 1. Conception des Entrees (4 canaux)

```python
input_tensor = np.stack([
    ct_resized,           # Canal 0: Image CT (normalisee)
    hu_coarse,            # Canal 1: Segmentation HU grossiere (tous les pixels HU dans -29~150)
    muscle_resized,       # Canal 2: Etiquettes musculaires connues
    1 - exclusion         # Canal 3: Masque de zone non-exclue
], axis=0)
```

**Raisons de conception** :
- `hu_coarse` fournit les zones candidates physiques pour les muscles, indiquant au modele "ces pixels pourraient physiquement etre des muscles"
- `muscle_resized` fournit les etiquettes correctes connues pour l'apprentissage des contours
- `1 - exclusion` marque clairement les zones qui ne peuvent pas etre des muscles (air, os)

### 2. Conception de la Fonction de Perte

```python
class ExpandThenRefireLoss(nn.Module):
    """
    Fonction de perte Expansion puis Affinage

    Objectifs :
    1. Correspondance precise dans les regions etiquetees (poids eleve)
    2. Encourager la couverture dans les regions HU valides non etiquetees (recompense de couverture)
    3. Exclure strictement les regions d'air et d'os
    """
```

#### Composition des Pertes

| Terme de perte | Poids | Fonction |
|----------------|-------|----------|
| label_alignment | 3.0 | Assurer l'alignement precis avec les etiquettes musculaires connues |
| coverage_reward | 1.0 | Encourager la couverture des regions HU valides non etiquetees |
| exclusion | 5.0 | Penaliser strictement les predictions erronees sur l'air/os |

#### Logique du Code Cle

```python
# 1. Perte d'alignement des regions etiquetees - poids eleve pour la precision
label_region_loss = (bce_all * has_label).sum() / (has_label.sum() + 1e-6)
losses['label_alignment'] = label_region_loss * 3.0

# 2. Recompense de couverture - encourager la couverture des regions HU valides non etiquetees
unlabeled_hu_valid = hu_coarse * (1 - has_label) * body_mask * (1 - exclusion_mask)
coverage_loss = ((1 - pred_prob) * unlabeled_hu_valid).mean()
losses['coverage_reward'] = coverage_loss * 1.0

# 3. Perte d'exclusion - interdiction absolue de predire l'air/os
exclusion_loss = (pred_prob * exclusion_mask).mean()
losses['exclusion'] = exclusion_loss * 5.0
```

### 3. Configuration d'Entrainement

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

# Poids des pertes
LABEL_ALIGNMENT_WEIGHT = 3.0
COVERAGE_REWARD_WEIGHT = 1.0
EXCLUSION_WEIGHT = 5.0
```

### 4. Filtrage Abdominal

La logique de filtrage abdominal de V2 a ete conservee, ne traitant que les coupes de la region abdominale (excluant les cuisses) :

```python
# Determination de la region abdominale par marqueurs anatomiques
abdominal_organs = [1, 2, 3, 5, 6]  # Rate, reins, foie, etc.
thigh_bones = [74, 76]  # Femurs gauche et droit

is_abdominal = any(organ in labels for organ in abdominal_organs)
has_thigh = any(bone in labels for bone in thigh_bones)

# Conserver les coupes abdominales, exclure les coupes de cuisse
keep_slice = is_abdominal and not has_thigh
```

## Modifications de Fichiers

| Fichier | Operation | Description |
|---------|-----------|-------------|
| `scripts/train_muscle_transfer_v3.py` | Nouveau | Implementation complete de la methode V3 "Expansion puis Affinage" |
| `.gitignore` | Modifie | Ajout des regles pour `outputs_muscle_transfer_v3/` |
| `outputs_muscle_transfer_v3/` | Nouveau | Repertoire de sortie pour l'entrainement V3 |

## Etat de l'Entrainement

- **Heure de demarrage** : 2026-01-11 01:08
- **PID du processus** : 2333264
- **Donnees d'entrainement** : 6 352 coupes abdominales (30 sujets)
- **Donnees de validation** : 2 119 coupes abdominales (8 sujets)
- **Lien WandB** : https://wandb.ai/haochen-zhang-insa-lyon/muscle-transfer-learning/runs/ok1dch31

## Effets Attendus

1. **Rappel plus eleve** : Le modele devrait couvrir davantage de regions musculaires
2. **Maintien de la precision des contours** : Grace a la perte label_alignment, les contours des regions connues devraient rester precis
3. **Eviter les faux positifs** : La perte d'exclusion garantit que l'air et les os ne seront pas mal classifies

## Directions d'Amelioration Futures

1. Si la couverture reste insuffisante, augmenter `COVERAGE_REWARD_WEIGHT`
2. Si les contours ne sont pas assez precis, augmenter `LABEL_ALIGNMENT_WEIGHT`
3. Envisager d'ajouter des contraintes de connectivite pour eviter les petites regions isolees

---

*Date de generation du rapport : 2026-01-11*
