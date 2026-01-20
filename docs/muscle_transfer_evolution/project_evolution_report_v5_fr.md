# Rapport d'Évolution du Projet de Segmentation Musculaire : De l'Exploration RL à l'Exclusion d'Organes V5

**Date** : 20 Janvier 2026  
**Auteur** : Agent Gemini (Basé sur l'analyse du code et de la documentation)

## 1. Contexte et Motivation du Projet

Après avoir réalisé la segmentation musculaire initiale avec TotalSegmentator, l'équipe a discuté et constaté que les étiquettes existantes étaient insuffisantes pour répondre aux besoins de la recherche. Nous avions besoin de segmenter **plus** de parties musculaires, au-delà des étiquettes anatomiques standard.

**Défi Principal** :
Nous devions utiliser les étiquettes musculaires existantes comme "graines" pour "trouver" des régions dans l'image qui n'étaient pas étiquetées mais qui appartenaient aux muscles. Il ne s'agit pas seulement d'une tâche de segmentation, mais plutôt d'une tâche de "découverte" semi-supervisée.

---

## 2. Exploration Précoce : Apprentissage par Renforcement (RL) & V1/V2

### 2.1 Tentative d'Apprentissage par Renforcement (`scripts/train_muscle_rl.py`)
Pour atteindre l'objectif de "trouver des muscles supplémentaires", nous avons initialement tenté l'apprentissage par renforcement (algorithme PPO).
*   **Idée** : Considérer le modèle de segmentation comme un agent qui apprend en interagissant avec l'environnement.
*   **Méthode** :
    *   **Enseignant (Teacher)** : Utiliser les modèles de 10 muscles existants comme "enseignant".
    *   **Récompense (Reward)** :
        *   `teacher_consistency` : Récompense positive pour la cohérence avec les prédictions de l'enseignant.
        *   `hu_range` : Récompense si la valeur HU de la zone prédite est dans la plage musculaire.
        *   `exploration` : Encourager l'exploration dans les zones non étiquetées par l'enseignant (tant que la valeur HU est raisonnable).
*   **Conclusion** : C'était une tentative très exploratoire pour laisser le modèle "apprendre" de manière autonome ce qu'est un muscle.

### 2.2 V1 & V2 (Fondations)
Comme indiqué dans le "Rapport d'Évolution des Méthodes de Transfert d'Apprentissage Musculaire" :
*   **V1** : Tentative d'architecture Teacher-Student, mais limitée par des problèmes de correspondance de structure de modèle.
*   **V2** : Retour aux fondamentaux, entraînement supervisé direct utilisant les étiquettes TotalSegmentator, avec ajout de contraintes strictes sur les valeurs HU (Hard Constraint).
    *   **Résultat** : La précision du modèle était élevée (Dice 0.80), mais **trop conservatrice**. Il n'apprenait qu'à segmenter les muscles déjà étiquetés et n'osait pas prédire les zones non étiquetées, échouant ainsi à atteindre l'objectif de "découvrir plus de muscles".

---

## 3. V3 : Extension Radicale ("Expand Then Refine")

Pour résoudre le problème "trop conservateur" de la V2, la V3 a adopté une stratégie radicale.

*   **Analyse du Code** : `scripts/train_muscle_transfer_v3.py`
*   **Logique Centrale** :
    *   Introduction de la **Récompense de Couverture (Coverage Reward)**. Tant que la valeur HU d'un pixel est dans la plage musculaire (-29 ~ 150) et qu'il ne s'agit ni d'os ni d'air, le modèle ne sera pas pénalisé pour l'avoir prédit comme muscle, mais sera au contraire encouragé.
    *   **Entrée** : 4 canaux (CT + Segmentation HU grossière + Étiquette + Masque d'exclusion).
*   **Fonction de Perte** : `ExpandThenRefineLoss`
    ```python
    # Récompense de couverture : Encourager la prédiction comme muscle dans les zones non étiquetées
    coverage_loss = ((1 - pred_prob) * unlabeled_hu_valid).mean()
    losses['coverage_reward'] = coverage_loss * COVERAGE_REWARD_WEIGHT
    ```
*   **Problème Rencontré ("Problème des Viscères")** :
    Bien que la V3 ait réussi à "découvrir" plus de zones, elle en a **trop découvert**.
    *   **Cause** : Les valeurs HU des organes viscéraux abdominaux (comme le foie, les reins, la rate) sont très proches de celles des muscles.
    *   **Résultat** : Comme la fonction de perte de la V3 encourageait la couverture de toutes les zones "ressemblant à du muscle", le modèle a incorrectement segmenté une grande quantité d'organes viscéraux comme étant des muscles.

---

## 4. V5 : Exclusion Précise & Focalisation L3 (Solution Actuelle)

La V5 est une version corrective directe du problème de "mauvaise classification des viscères" de la V3. Notre objectif est : **conserver la capacité de découvrir de nouveaux muscles, mais établir des frontières en supprimant les organes viscéraux de l'entraînement.**

*   **Analyse du Code** : `scripts/train_muscle_transfer_v5_l3.py`

### 4.1 Stratégies d'Amélioration Principales

1.  **Exclusion Explicite des Échantillons Négatifs (Organ Exclusion)** :
    Ne plus se fier uniquement aux valeurs HU. Nous chargeons explicitement tous les fichiers de segmentation non musculaires (`ORGAN_FILES`, tels que foie, rate, reins, etc.) et les fusionnons en un `non_muscle_mask`.
    *   **Logique** : Si une zone est le foie, même si sa valeur HU ressemble à du muscle, ce n'est absolument pas du muscle.

2.  **Focalisation Vertébrale L3 (L3 Focus)** :
    *   Utilisation de `vertebrae_L3.nii.gz` pour localiser la vertèbre L3.
    *   L'entraînement se concentre uniquement sur les coupes proches de L3. C'est le niveau standard de référence pour l'analyse des muscles abdominaux, ce qui réduit les interférences d'autres niveaux.

3.  **Mécanisme d'Attention (Attention Mechanism)** :
    Introduction de `AttentionMuscleNet`, incluant l'encodage positionnel et l'auto-attention, pour aider le modèle à comprendre la structure anatomique (par exemple : "les muscles sont généralement à la périphérie du corps, les organes à l'intérieur").

### 4.2 Analyse Approfondie : Fonction de Perte de la V5

La V5 utilise une `AttentionAwareLoss` complexe, spécialement conçue pour trouver des muscles tout en excluant les organes.

| Terme de Perte | Poids | Fonction | Analyse de la Logique du Code |
| :--- | :--- | :--- | :--- |
| **Pénalité Non-Muscle** | **Élevé (1.5 * 5.0)** | **Correction Centrale** : Pénalise sévèrement les zones prédites comme organes. | `(pred_prob * non_muscle_mask).sum()`. C'est le changement majeur de la V5 par rapport à la V3, tuant directement les erreurs de classification des viscères. |
| **Alignement d'Étiquette** | 3.0 | Base : Garantit une segmentation précise des muscles connus (ex : psoas). | Basé sur la perte BCE, calculé uniquement dans les zones étiquetées. |
| **Récompense de Couverture** | 1.0 | Héritage V3 : Continue d'encourager la découverte de nouveaux muscles. | **Restriction Clé** : Récompense calculée uniquement dans `unlabeled_region` (zones non étiquetées ET non organes). |
| **Cohérence de Similarité** | 1.5 | Nouvelle fonctionnalité : Trouver des muscles basés sur la similarité visuelle. | Force le modèle : Si une zone non étiquetée ressemble (en caractéristiques) à un muscle connu, elle devrait être prédite comme muscle. |

### 4.3 Résumé : Le Saut de la V3 à la V5

*   **V3** : "Tant que la valeur HU ressemble à du muscle, devine que c'est du muscle." -> **Conduit à la sélection erronée des viscères**.
*   **V5** : "La valeur HU ressemble à du muscle, **ET** on sait que ce n'est pas foie/rein/rate, **ET** sa position anatomique est similaire au muscle (via Attention), alors seulement devine que c'est du muscle."

De cette manière, la V5 vise à atteindre l'objectif final de l'équipe : étendre précisément notre carte de segmentation musculaire dans un environnement propre et sans interférence viscérale.
