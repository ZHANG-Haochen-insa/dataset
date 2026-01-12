# Rapport Technique - Apprentissage par Transfert Musculaire V3

## Apercu

La version V3 adopte l'approche "Expansion puis Affinage" (Expand then Refine) pour resoudre le probleme de la version V2 ou le modele etait trop conservateur avec une couverture insuffisante. L'idee centrale est d'utiliser d'abord la plage de valeurs HU pour une segmentation grossiere afin d'obtenir un rappel eleve, puis d'apprendre a affiner les contours en utilisant les regions musculaires deja etiquetees comme reference.

## Contexte et Problematique

### Limitations de la Version V2

La version V2 a obtenu de bons indicateurs sur l'ensemble de validation (conformite HU 99.5%, Dice 0.80), mais l'analyse pratique a revele un probleme fondamental :

**Modele trop conservateur** : La zone musculaire predite par le modele etait insuffisante en termes de couverture. Meme les regions musculaires deja etiquetees n'etaient pas entierement couvertes.

## Conception de la Methode

### Concept Fondamental

```
Expansion : Segmentation grossiere par plage HU → Couvrir toutes les regions potentiellement musculaires (rappel eleve)
↓
Affinage : Reference aux etiquettes musculaires connues → Apprendre a affiner les contours
```

### Conception de la Fonction de Perte

La V3 utilise une nouvelle fonction de perte `ExpandThenRefireLoss` comprenant cinq composantes :

| Composante de perte | Poids | Fonction |
|---------------------|-------|----------|
| Alignement des etiquettes (label_alignment) | 3.0 | Les regions etiquetees doivent correspondre precisement aux etiquettes |
| Recompense de couverture (coverage_reward) | 1.0 | Encourager la couverture des regions HU valides non etiquetees |
| Contrainte d'exclusion (exclusion) | 5.0 | Exclusion forcee des regions air/os |
| Penalite hors plage HU (hu_violation) | 1.0 | Penalite legere pour les predictions hors plage HU |
| Lissage des contours (smoothness) | 0.2 | Maintenir des contours de prediction lisses |

### Architecture du Reseau

Utilisation de `MuscleRefineNet`, base sur l'architecture U-Net :

- **Nombre de canaux d'entree** : 4
  - Canal 0 : Image CT (normalisee)
  - Canal 1 : Segmentation HU grossiere (region candidate)
  - Canal 2 : Etiquettes musculaires connues (reference)
  - Canal 3 : Masque de region non-exclue
- **Canaux de feature maps** : [32, 64, 128, 256]
- **Couche goulot** : 512 canaux
- **Dropout** : Encodeur 0.1, goulot 0.2

## Configuration des Hyperparametres

### Parametres d'Entrainement

| Parametre | Valeur |
|-----------|--------|
| Taux d'apprentissage (Learning Rate) | 1e-4 |
| Decroissance des poids (Weight Decay) | 1e-5 |
| Taille de lot (Batch Size) | 16 |
| Nombre d'epoques (Epochs) | 30 |
| Optimiseur | AdamW |
| Planification du taux d'apprentissage | CosineAnnealingLR |

### Contraintes de Valeurs HU

| Parametre | Valeur HU | Description |
|-----------|-----------|-------------|
| Limite inferieure HU muscle | -29 | Valeur HU minimale du tissu musculaire |
| Limite superieure HU muscle | 150 | Valeur HU maximale du tissu musculaire |
| Seuil air | -200 | En dessous = air, exclusion forcee |
| Seuil os | 300 | Au dessus = os, exclusion forcee |
| Seuil corps | -500 | En dessous = arriere-plan |

### Configuration des Donnees

- **Nombre de sujets** : 50
- **Taille d'image cible** : 256×256
- **Filtrage abdominal** : Active (exclusion de la region des cuisses)
- **Ratio entrainement/validation** : 80%/20%

## Resultats d'Entrainement

### Metriques Finales

| Metrique | Valeur finale | Description |
|----------|---------------|-------------|
| Rappel des etiquettes | 100% | Muscles etiquetes entierement couverts |
| Dice des etiquettes | 31.4% | Chevauchement avec les etiquettes |
| Couverture HU | 99.99% | Degre de couverture des regions HU valides |
| Debordement d'exclusion | 0.0046% | Proportion de prediction dans les zones exclues |
| Ratio d'expansion | 11.7x | Expansion de la zone predite par rapport aux etiquettes |

### Convergence des Pertes

- Perte d'entrainement : de 0.907 a 0.024
- Perte de validation : de 0.672 a 0.025

### Courbes d'Entrainement

![Courbes d'entrainement](v3_training_history.png)

Les courbes d'entrainement montrent :
1. **Courbe de perte** : Les pertes d'entrainement et de validation convergent rapidement et se stabilisent
2. **Rappel des etiquettes** : Proche de 100% des la premiere epoque
3. **Dice des etiquettes** : Stable autour de 31%, refletant l'intention de la strategie d'expansion
4. **Couverture HU** : Atteint rapidement et maintient plus de 99%
5. **Debordement d'exclusion** : Reste constamment a un niveau tres bas
6. **Ratio d'expansion** : Stable autour de 11.7 fois

### Visualisation des Predictions

![Visualisation des predictions](v3_visualization_epoch30.png)

Explication de la visualisation (de gauche a droite) :
1. **CT Image** : Image CT originale
2. **HU Coarse (Candidate)** : Zone candidate de segmentation HU grossiere (vert)
3. **Known Labels** : Etiquettes musculaires connues (rouge)
4. **Prediction** : Resultat de prediction du modele (bleu)
5. **R:Miss G:New B:Match** : Rouge=manque, Vert=nouvelle couverture, Bleu=correspondance correcte
6. **R:Missed G:Good B:Bad** : Rouge=HU valide mais non predit, Vert=HU valide et predit, Bleu=HU invalide mais predit

## Analyse des Resultats

### Avantages

1. **Rappel eleve** : 100% de couverture des regions musculaires etiquetees
2. **Forte couverture HU** : Couverture quasi totale de toutes les regions HU valides
3. **Faible debordement d'exclusion** : Tres peu de predictions dans les zones os/air
4. **Entrainement stable** : Convergence fluide des pertes, sans oscillation

### Problemes et Reflexions

1. **Dice faible** : Le score Dice d'environ 31% reflete la sur-expansion due a la strategie "expansion d'abord"
2. **Ratio d'expansion eleve** : Une expansion de 11.7 fois signifie que la zone predite est bien plus grande que les etiquettes reelles
3. **Precision des contours** : Bien que la couverture soit elevee, la precision des contours reste a ameliorer

### Comparaison avec V2

| Caracteristique | V2 | V3 |
|-----------------|-----|-----|
| Strategie | Prediction conservatrice | Couverture aggressive |
| Rappel | Faible | 100% |
| Dice | ~80% | ~31% |
| Ratio d'expansion | <1x | 11.7x |
| Probleme de sous-detection | Severe | Resolu |
| Sur-segmentation | Non | Presente |

## Conclusion

La version V3 a reussi a resoudre le probleme de sous-detection de V2, atteignant un rappel de 100% des etiquettes. La conception "expansion puis affinage" permet au modele de couvrir toutes les regions musculaires potentielles. Cependant, la version actuelle a des capacites limitees d'affinage des contours dans la phase "affinage", conduisant a un ratio d'expansion eleve.

Directions d'amelioration futures :
1. Renforcer le poids de la perte d'alignement des contours
2. Introduire un post-traitement morphologique
3. Ajouter un module de detection des contours
4. Strategie d'entrainement multi-etapes

---

*Date de generation du rapport : 2026-01-12*
