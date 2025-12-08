# Rapport d'Entraînement du Modèle de Segmentation d'Images Médicales CT 3D

## Aperçu du Projet

Ce rapport documente en détail le processus d'entraînement, les performances et les découvertes clés d'un modèle de segmentation multi-organes d'images médicales CT 3D basé sur l'architecture U-Net.

### Informations de Base
- **Architecture du Modèle**: U-Net 2D
- **Méthode d'Entraînement**: Entraînement sur tranches axiales 2D
- **Type de Données**: Images CT au format NIfTI
- **Tâche de Segmentation**: Segmentation multi-canal de 117 structures anatomiques
- **Nombre d'Époques**: 20 epochs
- **Durée d'Entraînement**: Environ 309 heures (12.9 jours)
- **Durée Moyenne par Époque**: 15.4 heures
- **Outil de Surveillance**: Weights & Biases (wandb)

---

## 1. Informations sur le Dataset

### 1.1 Étiquettes des Structures Anatomiques
Le modèle peut identifier et segmenter **117 structures anatomiques différentes**, incluant :

#### Systèmes d'Organes Principaux
- **Système Digestif**: foie, estomac, pancréas, rate, côlon, intestin grêle, duodénum, vésicule biliaire, œsophage
- **Système Respiratoire**: lobes supérieurs/moyens/inférieurs des poumons gauche/droit, trachée
- **Système Circulatoire**: cœur, aorte, veines caves supérieure/inférieure, veine porte, veine pulmonaire, diverses branches artérielles et veineuses
- **Système Urogénital**: reins gauche/droit, kystes rénaux, prostate, vessie, glandes surrénales gauche/droit
- **Système Nerveux**: cerveau, moelle épinière
- **Système Endocrinien**: glande thyroïde

#### Système Squelettique
- **Colonne Vertébrale**: vertèbres cervicales (C1-C7), thoraciques (T1-T12), lombaires (L1-L5), sacrées (S1), sacrum
- **Cage Thoracique**: côtes gauche/droit (1-12), sternum, clavicules, cartilages costaux
- **Membres**: humérus gauche/droit, fémurs, articulations de la hanche
- **Autres**: crâne, omoplates

#### Système Musculaire
- Grand fessier, moyen fessier, petit fessier (gauche/droit)
- Iliopsoas (gauche/droit)
- Muscles du dos (gauche/droit)

---

## 2. Analyse des Performances d'Entraînement

### 2.1 Perte d'Entraînement (Training Loss)

La perte d'entraînement a diminué rapidement de **0.0604** initial à **0.000332** final, soit une réduction totale de **99.45%**.

**Phases Clés**:
- **Époque 1-2**: Diminution drastique (0.0604 → 0.0068), réduction de 88.7%
- **Époque 2-5**: Convergence rapide (0.0068 → 0.0017), réduction de 75.0%
- **Époque 5-10**: Optimisation stable (0.0017 → 0.00066), réduction de 61.2%
- **Époque 10-20**: Ajustement fin (0.00066 → 0.00033), réduction de 50.0%

### 2.2 Coefficient Dice d'Entraînement (Training Dice Score)

Le coefficient Dice a augmenté de **0.8200** à **0.9764**, soit une amélioration de **15.64 points de pourcentage**.

**Jalons de Performance**:
- **Époque 1**: 0.8200 (point de départ)
- **Époque 5**: 0.9126 (franchissement du seuil 0.9)
- **Époque 10**: 0.9452 (stabilisation à 0.94+)
- **Époque 15**: 0.9701 (franchissement du seuil 0.97)
- **Époque 20**: 0.9764 (performance optimale)

### 2.3 Performance sur l'Ensemble de Validation

#### Coefficient Dice de Validation (Validation Dice Score)
- **Valeur Initiale**: 0.8358
- **Meilleure Valeur**: 0.9318 (Époque 18)
- **Valeur Finale**: 0.9317 (Époque 20)
- **Amélioration Totale**: 9.59 points de pourcentage

#### IoU de Validation (Validation Intersection over Union)
- **Valeur Initiale**: 0.8356
- **Meilleure Valeur**: 0.9185 (Époque 18)
- **Valeur Finale**: 0.9184 (Époque 20)
- **Amélioration Totale**: 8.28 points de pourcentage

### 2.4 Perte de Validation (Validation Loss)
- **Valeur Initiale**: 0.0093
- **Valeur Minimale**: 0.0015 (Époque 9)
- **Valeur Finale**: 0.0018 (Époque 20)
- **Réduction**: 80.3%

---

## 3. Analyse de la Stratégie d'Entraînement

### 3.1 Planification du Taux d'Apprentissage

Utilisation d'une stratégie de **recuit cosinus du taux d'apprentissage**:
- **Taux d'Apprentissage Initial**: 0.001
- **Taux d'Apprentissage Final**: 7.15e-06 (réduction de 99.3%)
- **Méthode de Planification**: Décroissance en courbe cosinus lisse

**Avantages**:
- Utilisation d'un taux d'apprentissage élevé en début pour une convergence rapide
- Utilisation d'un petit taux d'apprentissage en fin pour un ajustement fin
- Évite les changements brusques des planificateurs par paliers

### 3.2 Surveillance de la Norme du Gradient

La norme du gradient est restée stable pendant l'entraînement, dans la plage **0.0006 - 0.0079**:
- **Valeur Moyenne**: 0.0024
- **Valeur Maximale**: 0.0079 (Époque 2)
- **Valeur Minimale**: 0.0006 (Époque 20)

**Une norme de gradient stable indique**:
- Absence d'explosion ou de disparition du gradient
- Entraînement stable et sain du modèle
- Paramètres d'optimiseur bien configurés

### 3.3 Statistiques de Temps d'Entraînement

- **Durée d'Entraînement par Époque**: 54,421 - 56,856 secondes
- **Durée Moyenne**: 55,498 secondes (environ 15.4 heures)
- **Durée Totale d'Entraînement**: 1,109,960 secondes (environ 308.9 heures / 12.9 jours)
- **Stabilité Temporelle**: Variation de la durée par époque inférieure à 5%

---

## 4. Évaluation des Performances du Modèle

### 4.1 Analyse de la Convergence

**Performance sur l'Ensemble d'Entraînement**:
- Amélioration continue, sans surapprentissage évident
- Dice final de 0.9764, performance excellente

**Performance sur l'Ensemble de Validation**:
- Tend à se stabiliser après l'Époque 9
- Légères fluctuations entre les Époques 15-20 (0.924-0.932)
- Performance optimale atteinte à l'Époque 18

**Capacité de Généralisation**:
- Écart Dice entre entraînement et validation d'environ 0.045 (4.5%)
- Écart dans une plage raisonnable, bonne capacité de généralisation
- Absence de surapprentissage sévère

### 4.2 Comparaison avec les Standards de Performance

Selon les standards du domaine de la segmentation d'images médicales:

| Plage Dice Score | Niveau de Performance | Performance de ce Modèle |
|------------------|----------------------|--------------------------|
| > 0.90           | Excellent            | ✓ Validation: 0.9317 |
| 0.80 - 0.90      | Bon                  | ✓ Phase initiale |
| 0.70 - 0.80      | Acceptable           | - |
| < 0.70           | À améliorer          | - |

**Ce modèle a atteint un niveau excellent sur la tâche complexe de segmentation de 117 structures anatomiques.**

### 4.3 Tendance Performance Entraînement vs Validation

```
Époque   Train Dice    Val Dice    Écart
--------------------------------------------
1        0.8200        0.8358      -0.0158
5        0.9126        0.9019      +0.0107
10       0.9452        0.9185      +0.0267
15       0.9701        0.9305      +0.0396
20       0.9764        0.9317      +0.0447
```

**Observations**:
- Performance de validation légèrement supérieure à l'entraînement en début (effet d'augmentation de données)
- Performance d'entraînement commence à dépasser la validation en milieu de parcours
- Écart stabilisé à 4-5% en fin, indiquant une bonne généralisation

---

## 5. Découvertes Clés et Insights

### 5.1 Caractéristique de Convergence Rapide
- Les **5 premières époques** ont accompli l'apprentissage principal (Dice passant de 0.82 à 0.91)
- Les **15 époques suivantes** ont servi à l'ajustement fin (Dice passant de 0.91 à 0.98)
- Indique une bonne adéquation entre l'architecture du modèle et la tâche

### 5.2 Processus d'Entraînement Stable
- Absence d'explosion ou de disparition du gradient
- Courbe de perte décroissant en douceur
- Stratégie de planification du taux d'apprentissage efficace

### 5.3 Bonne Capacité de Généralisation
- Écart entraînement-validation maintenu sous 5%
- Performance de validation stable, sans fluctuations brutales
- Métrique IoU atteignant également 0.918, validant davantage la qualité du modèle

### 5.4 Efficacité Computationnelle
- Durée d'entraînement moyenne de 15.4 heures par époque
- Total de 20 époques en environ 13 jours
- Efficacité d'entraînement raisonnable pour une tâche de segmentation complexe à 117 canaux

---

## 6. Description des Fichiers de Sortie

### 6.1 Points de Contrôle du Modèle
- **Format de Fichier**: `checkpoint_enhanced_epoch{N}.pth`
- **Nombre Sauvegardé**: 20 (un par époque)
- **Taille de Fichier**: Environ 89MB/point de contrôle
- **Stockage Total**: Environ 1.78GB

**Recommandations**:
- L'Époque 18 représente le point de contrôle avec la meilleure performance de validation
- L'Époque 20 représente le point de contrôle final d'entraînement
- Les points de contrôle intermédiaires peuvent être supprimés pour économiser l'espace

### 6.2 Fichiers de Visualisation

| Nom du Fichier | Description | Usage |
|----------------|-------------|-------|
| `training_history_enhanced.png` | Graphiques des courbes d'historique d'entraînement | Analyse des tendances de performance |
| `test_inference_enhanced.png` | Visualisation de l'inférence de test | Démonstration des effets de segmentation |
| `sample_visualization_enhanced.png` | Visualisation d'échantillons | Vérification de la qualité des données |
| `ct_slices.png` | Affichage de tranches CT | Aperçu des données brutes |
| `ct_mesh.html` | Visualisation de maillage 3D | Visualisation 3D interactive |

### 6.3 Fichiers de Données

- **`training_history_enhanced.json`**: Enregistrement complet des métriques d'entraînement
- **`label_map.json`**: Mappage des étiquettes des 117 structures anatomiques

---

## 7. Recommandations et Travaux Futurs

### 7.1 Recommandations d'Utilisation du Modèle
1. **Utiliser le point de contrôle de l'Époque 18** comme modèle de production (meilleure performance de validation)
2. Des ajustements ciblés peuvent être nécessaires pour des scénarios d'application spécifiques
3. Recommandé de valider et tester sur de nouvelles données

### 7.2 Directions d'Amélioration Potentielles

**Optimisations à Court Terme**:
- Implémenter un mécanisme d'arrêt précoce (la performance de validation se stabilise après l'Époque 15)
- Augmenter les stratégies d'augmentation de données pour améliorer davantage la généralisation
- Essayer différentes combinaisons de fonctions de perte (comme Focal Loss + Dice Loss)

**Explorations à Long Terme**:
- Essayer l'architecture U-Net 3D pour mieux exploiter l'information spatiale
- Introduire des mécanismes d'attention pour améliorer la précision de segmentation des petits organes
- Explorer des modèles légers pour améliorer la vitesse d'inférence
- Stratégie d'entraînement multi-échelle

### 7.3 Considérations de Déploiement
- **Performance d'Inférence**: Besoin d'évaluer le temps d'inférence par tranche
- **Besoins en Mémoire**: Taille du modèle d'environ 89MB, besoins en mémoire raisonnables pour l'inférence
- **Post-traitement**: Opérations morphologiques potentiellement nécessaires pour optimiser les contours de segmentation
- **Contrôle Qualité**: Recommandé d'ajouter un mécanisme d'évaluation de confiance

---

## 8. Conclusion

Cet entraînement a réussi à atteindre un niveau de performance excellent sur la tâche complexe de segmentation d'images médicales incluant 117 structures anatomiques :

**Réalisations Principales**:
- Coefficient Dice d'entraînement atteignant **0.9764**
- Coefficient Dice de validation atteignant **0.9317**
- IoU de validation atteignant **0.9184**
- Processus d'entraînement stable avec bonne capacité de généralisation

**Points Forts Techniques**:
- Stratégie efficace de planification du taux d'apprentissage
- Optimisation stable du gradient
- Surveillance et visualisation complètes

**Valeur Pratique**:
Ce modèle peut être utilisé pour des tâches automatisées de segmentation multi-organes d'images CT, avec de larges perspectives d'application dans l'analyse d'imagerie médicale, la planification de radiothérapie, la recherche anatomique et d'autres domaines.

---

## Annexe : Spécifications Techniques

**Environnement Matériel**:
- Entraînement sur GPU (modèle spécifique à vérifier dans les logs d'entraînement)
- Durée d'entraînement par époque d'environ 15.4 heures

**Dépendances Logicielles**:
- Framework de deep learning PyTorch
- NiBabel (traitement de fichiers NIfTI)
- Weights & Biases (suivi d'expériences)
- Bibliothèques de calcul scientifique NumPy, scikit-image, etc.

**Traitement des Données**:
- Entrée: Scans CT au format NIfTI
- Prétraitement: Extraction de tranches 2D, normalisation
- Sortie: Masques de segmentation à 117 canaux

**Paramètres du Modèle**:
- Optimiseur: Adam
- Taux d'apprentissage initial: 0.001
- Planification du taux d'apprentissage: CosineAnnealingLR
- Taille de batch: Non mentionnée dans ce rapport (à vérifier dans la configuration d'entraînement)

---

**Date de Génération du Rapport**: 2025-12-08
**Source des Données**: `/local/hzhang02/data/dataset/outputs/training_history_enhanced.json`
**Script d'Entraînement**: `/local/hzhang02/data/dataset/scripts/train_unet_enhanced.py`
