# Rapport Preliminaire - Apprentissage par Transfert Musculaire V4

## Resume

La version V4 represente une amelioration majeure de V3. L'innovation principale est **l'importation de toutes les donnees de segmentation**, permettant une exclusion precise des regions non musculaires (organes, os, vaisseaux) tout en recherchant des tissus musculaires similaires uniquement dans les regions non annotees.

## Contexte et Problematique

### Limitations de la Version V3

V3 a adopte la strategie "expansion puis raffinement", atteignant avec succes un taux de rappel de 100% et un ratio d'expansion de 11.7x. Cependant, certaines limitations persistent :

1. **Exclusion basee uniquement sur les valeurs HU** : Seules les valeurs HU (air<-200, os>300) sont utilisees pour l'exclusion
2. **Risque de confusion avec les organes** : Les valeurs HU du foie, de la rate, etc. peuvent se situer dans la plage musculaire (-29 a 150)
3. **Absence de contraintes anatomiques** : Les annotations existantes d'organes/os ne sont pas exploitees

### Probleme Central

De nombreux organes internes (foie, reins) et tissus mous ont des valeurs HU qui chevauchent celles des muscles. L'utilisation exclusive des valeurs HU ne permet pas une distinction efficace.

## Solution V4

### Concept Central

```
Importer tous les segments → Construire le masque d'exclusion non-musculaire → Rechercher les muscles uniquement dans les regions non annotees
```

### Ameliorations Cles

#### 1. Importation Complete des Donnees de Segmentation

V4 importe **107 fichiers** de segmentation non-musculaire de TotalSegmentator :

| Categorie | Nombre | Exemples |
|-----------|--------|----------|
| Organes (visceres) | 29 | Foie, rate, reins, pancreas, estomac, intestins, coeur, poumons, etc. |
| Os | 55+ | Vertebres (C1-L5), cotes (24), femur, bassin, crane, etc. |
| Vaisseaux | 16 | Aorte, veine cave, veine porte, arteres iliaques, etc. |

#### 2. Mecanisme d'Exclusion a Trois Niveaux

```
Zone d'exclusion = Exclusion HU ∪ Annotations non-musculaires

Ou :
- Exclusion HU : Air(<-200) ∪ Os(>300)
- Annotations non-musculaires : Organes ∪ Os annotes ∪ Vaisseaux
```

#### 3. Conscience des Regions Annotees

Ajout d'un "masque de toutes les regions annotees". Le reseau recherche les muscles similaires uniquement dans les **regions non annotees** :

```
Zone de recherche = Corps - Muscles connus - Annotations non-musculaires - Exclusion HU
```

### Amelioration de l'Architecture Reseau

Les canaux d'entree passent de 4 a 5 :

| Canal | Contenu | Description |
|-------|---------|-------------|
| 0 | Image CT | CT normalise |
| 1 | Segmentation grossiere HU | Zone candidate dans la plage HU musculaire |
| 2 | Labels musculaires connus | 10 types de muscles annotes |
| 3 | Zone non-exclue | 1 - (Exclusion HU ∪ Annotations non-musculaires) |
| 4 | **Zone non annotee** | 1 - Toutes les regions annotees (nouveau) |

### Amelioration de la Fonction de Perte

Trois nouveaux composants ajoutes a la base V3 :

| Composant de perte | Poids | Role |
|--------------------|-------|------|
| Penalite non-musculaire (non_muscle_penalty) | 7.5 | Interdiction stricte de predire des muscles dans les organes/os/vaisseaux |
| Penalite de similarite non-musculaire (similarity_non_muscle) | 1.0 | Maintenir la carte de similarite basse dans les regions non-musculaires |
| Recompense de couverture (amelioree) | 1.0 | Calculee uniquement dans les regions non annotees |

Fonction de perte complete :

```
L_total = L_label_alignment (3.0)
        + L_coverage_reward (1.0)      # Uniquement zones non annotees
        + L_exclusion (5.0)
        + L_non_muscle_penalty (7.5)   # Nouveau
        + L_hu_violation (1.0)
        + L_smoothness (0.2)
        + L_similarity_consistency (1.5)
        + L_similarity_supervision (0.5)
        + L_similarity_non_muscle (1.0) # Nouveau
```

## Configuration des Hyperparametres

### Parametres d'Entrainement

| Parametre | Valeur |
|-----------|--------|
| Taux d'apprentissage | 1e-4 |
| Decroissance des poids | 1e-5 |
| Taille de batch | 8 |
| Nombre d'epoques | 30 |
| Optimiseur | AdamW |
| Planificateur LR | CosineAnnealingLR |
| Canaux d'entree | 5 |

### Configuration des Donnees

| Element | Valeur |
|---------|--------|
| Nombre de sujets | 50 |
| Ensemble d'entrainement | 30 sujets, 6352 coupes |
| Ensemble de validation | 8 sujets, 2119 coupes |
| Taille d'image cible | 256×256 |
| Filtrage abdominal | Active |
| Fichiers musculaires connus | 10 |
| Fichiers d'exclusion non-musculaires | 107 |

### Details des Fichiers d'Exclusion Non-Musculaires

#### Organes (29)
- Systeme digestif : Foie, rate, pancreas, estomac, colon, intestin grele, duodenum, vesicule biliaire, oesophage
- Systeme urinaire : Reins (G/D), kystes renaux (G/D), vessie, prostate
- Systeme respiratoire : Poumons (5 lobes), trachee
- Endocrinien : Glandes surrenales (G/D), thyroide
- Autres : Coeur, appendice auriculaire, cerveau, moelle epiniere

#### Os (55+)
- Vertebres : Cervicales C1-C7, Thoraciques T1-T12, Lombaires L1-L5, Sacrum S1
- Cotes : 12 gauches, 12 droites
- Os des membres : Femurs (G/D), Humerus (G/D)
- Bassin : Os iliaques (G/D)
- Autres : Crane, sternum, clavicules, scapulas, cartilages costaux

#### Vaisseaux (16)
- Arteres : Aorte, tronc brachiocephalique, carotides communes, sous-clavieres, iliaques
- Veines : Veine cave superieure/inferieure, veines brachiocephaliques, veine porte, veines pulmonaires, veines iliaques

## Nouvelle Metrique d'Evaluation

| Metrique | Description |
|----------|-------------|
| non_muscle_overflow | Proportion de predictions debordant dans les regions non-musculaires (organes/os/vaisseaux) |

Cette metrique reflete directement si le modele predit incorrectement des organes, os ou vaisseaux comme etant des muscles.

## Amelioration de la Visualisation

La visualisation V4 passe de 7 a 8 colonnes :

1. **CT Image** : Image CT originale
2. **HU Coarse** : Zone candidate par segmentation HU grossiere
3. **R:Muscle B:NonMuscle** : Rouge=Muscles connus, Bleu=Regions non-musculaires annotees
4. **Similarity Map** : Carte de similarite
5. **Prediction** : Resultat de prediction du modele
6. **R:Miss G:New B:Match** : Comparaison prediction vs labels connus
7. **R:Overflow G:Good B:Known** : Rouge=Debordement non-musculaire (probleme), Vert=Prediction en zone non annotee (bien), Bleu=Muscle connu
8. **New Muscle Found** : Nouveaux muscles decouverts dans les zones completement non annotees

## Effets Attendus

### Ameliorations par Rapport a V3

| Aspect | V3 | V4 Attendu |
|--------|-----|------------|
| Mecanisme d'exclusion | HU uniquement | HU + 107 annotations anatomiques |
| Confusion avec organes | Possible | Fortement reduite |
| Zone de recherche | Toutes zones HU valides | Zones non annotees uniquement |
| Contraintes anatomiques | Aucune | Presentes (annotations existantes) |

### Objectifs Principaux

1. **Maintenir 100% de rappel** : Les 10 types de muscles annotes doivent etre entierement couverts
2. **Reduire significativement le debordement non-musculaire** : Les predictions ne doivent pas envahir les organes, os, vaisseaux
3. **Decouverte precise de nouveaux muscles** : Recherche uniquement dans les zones non annotees anatomiquement coherentes

## Statut de l'Entrainement

- **Statut** : Entrainement en cours
- **Progression actuelle** : Epoque 1/30
- **Lien wandb** : [Voir l'entrainement en temps reel](https://wandb.ai/haochen-zhang-insa-lyon/muscle-transfer-learning)

## Travaux Futurs

A completer apres l'entrainement :
1. Metriques d'entrainement finales
2. Graphiques des courbes d'entrainement
3. Resultats de visualisation des predictions
4. Analyse comparative detaillee avec V3

---

*Date de generation du rapport preliminaire : 2026-01-12*
*Statut de l'entrainement : En cours*
