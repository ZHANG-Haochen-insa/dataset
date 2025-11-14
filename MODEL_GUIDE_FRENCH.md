# Modèle de Segmentation d'Images Médicales 3D - Guide Théorique et Pratique

## Table des Matières
1. [Principes du Modèle : Pourquoi Cette Approche](#principes-du-modèle--pourquoi-cette-approche)
2. [Implémentation Technique : Comment Faire](#implémentation-technique--comment-faire)
3. [Guide Pratique](#guide-pratique)
4. [Optimisation des Performances](#optimisation-des-performances)

---

## Principes du Modèle : Pourquoi Cette Approche

### 1. Pourquoi Choisir l'Architecture U-Net ?

#### 1.1 Les Défis Spécifiques de la Segmentation d'Images Médicales

La segmentation d'images médicales diffère fondamentalement de la segmentation d'images ordinaires :

**Défis** :
- **Exigence de haute précision** : Les contours des organes doivent être précis, toute erreur peut affecter le diagnostic
- **Caractéristiques multi-échelles** : Nécessité de capturer simultanément les détails (petits vaisseaux) et l'ensemble (grands organes)
- **Rareté des données** : Coût élevé et quantité limitée des données médicales annotées
- **Déséquilibre des classes** : Certains organes (comme le cœur) occupent beaucoup d'espace, d'autres structures (petits vaisseaux) très peu

**Avantages de U-Net** :
```
Encodeur (sous-échantillonnage) → Capture l'information contextuelle
     ↓
Couche Goulot → Compression des caractéristiques globales
     ↓
Décodeur (sur-échantillonnage) + Connexions de saut → Restauration des détails spatiaux
```

- **Connexions de Saut (Skip Connections)** : Transmettent directement les caractéristiques haute résolution de l'encodeur au décodeur, préservant les détails
- **Structure Symétrique** : Conception symétrique encodeur-décodeur garantissant l'équilibre entre extraction et reconstruction des caractéristiques
- **Efficace avec peu de données** : Comparé à d'autres réseaux profonds, U-Net obtient de bons résultats même sur de petits ensembles de données

#### 1.2 Principes Fondamentaux de la Conception Architecturale

```
Entrée: Coupe CT (1, 256, 256)
         ↓
    [Chemin Encodeur]
Conv(32) → Capture les caractéristiques de texture de base
    ↓ MaxPool
Conv(64) → Capture les contours des organes
    ↓ MaxPool
Conv(128) → Capture les formes des organes
    ↓ MaxPool
Conv(256) → Capture les relations entre organes
    ↓ MaxPool

    [Couche Goulot]
Conv(512) → Compression du contexte global

    [Chemin Décodeur]
UpConv(256) ← + Skip[256] → Restauration des relations entre organes
    ↑
UpConv(128) ← + Skip[128] → Restauration des formes
    ↑
UpConv(64) ← + Skip[64] → Restauration des contours
    ↑
UpConv(32) ← + Skip[32] → Restauration des textures fines
    ↑
Sortie: Masque 117 canaux (117, 256, 256)
```

**Pourquoi ces nombres de couches ?**
- 4 sous-échantillonnages : 256→128→64→32→16, capture le global sans perdre trop de détails
- Doublement des canaux (32→64→128→256→512) : Compense la perte d'information due à la réduction de résolution spatiale

### 2. Pourquoi 2D plutôt que 3D U-Net ?

#### 2.1 Compromis de Conception

| Aspect | U-Net 2D | U-Net 3D |
|--------|----------|----------|
| **Besoin mémoire** | Faible (traite une seule coupe) | Élevé (traite tout le volume) |
| **Vitesse d'entraînement** | Rapide | 10-20× plus lent |
| **Contexte spatial** | Limité (plan unique) | Complet (espace 3D) |
| **Cas d'usage** | Grandes données, ressources limitées | Données modérées, GPU haute performance |

**Raisons de notre choix** :
- **Grande échelle de données** : 117 structures × plusieurs échantillons, 2D permet une itération plus rapide
- **Contraintes de ressources** : GPU standard (8GB VRAM) suffit pour l'entraînement
- **Facilité de débogage** : Visualisation et analyse plus intuitives
- **Développement progressif** : Valider d'abord la faisabilité en 2D, puis passer au 3D

#### 2.2 Gestion du Contexte Spatial en 2D

Bien qu'une coupe unique soit 2D, nous introduisons l'information 3D par :

1. **Traitement multi-coupes** : Échantillonnage depuis différentes positions durant l'entraînement
2. **Post-traitement** : Lissage 3D possible sur les prédictions de coupes adjacentes
3. **Extensions futures** : U-Net 2.5D (entrée de 3 coupes adjacentes) ou U-Net 3D complet

### 3. Pourquoi Utiliser BCE Loss ?

#### 3.1 Choix de la Fonction de Perte

**Entropie Croisée Binaire (BCE) vs Dice Loss** :

```python
# BCE Loss (celui que nous utilisons)
BCE = -[y*log(p) + (1-y)*log(1-p)]
Avantages :
- Optimisation pixel par pixel, gradients stables
- Calcul indépendant pour chaque structure, adapté au multi-classe
- Convergence rapide en début d'entraînement

# Dice Loss (alternative)
Dice = 1 - (2*|X∩Y|)/(|X|+|Y|)
Avantages :
- Optimise directement la métrique de segmentation
- Insensible au déséquilibre des classes
Inconvénients :
- Gradients instables en début d'entraînement
```

**Pourquoi choisir BCE ?**
- **Problème multi-label** : 117 structures nécessitent des prédictions indépendantes (non exclusives)
- **Entraînement stable** : Gradients BCE plus stables en début d'entraînement
- **BCEWithLogitsLoss** : Intègre sigmoid, plus stable numériquement

**Amélioration future** : Possibilité d'essayer une perte combinée BCE + Dice

### 4. Pourquoi Ce Prétraitement des Données ?

#### 4.1 Stratégie de Normalisation

```python
# Les valeurs CT en unités HU : -1000 (air) à +3000 (os)
# Notre normalisation :
windowed = np.clip(ct_data, -200, 300)  # Fenêtrage
normalized = (windowed + 200) / 500     # [0, 1]
```

**Principe** :
- **Fenêtrage (Windowing)** : Focus sur la plage des tissus mous (-200 à 300 HU)
  - Poumons : -500 à -400 HU
  - Tissus mous : -100 à 100 HU
  - Os : +400 à +1000 HU
- **Pourquoi [-200, 300] ?** Couvre la plage HU de la plupart des organes
- **Normalisation [0,1]** : Entraînement plus stable du réseau neuronal

#### 4.2 Normalisation de Taille

```python
TARGET_SHAPE = (256, 256)
```

**Pourquoi 256×256 ?**
- **Puissance de 2** : S'adapte à 4 sous-échantillonnages (256→128→64→32→16)
- **Équilibre performance** : 128 trop petit perd les détails, 512 trop grand manque de VRAM
- **Standard** : Taille courante en traitement d'images médicales

---

## Implémentation Technique : Comment Faire

### 1. Conception du Pipeline de Données

#### 1.1 Organisation du Dataset

```python
class CTSegmentationDataset:
    """
    Philosophie de conception :
    - Chargement paresseux : Ne lit que lorsque nécessaire, économise la mémoire
    - Mécanisme de cache : Évite les lectures répétées sur disque
    - Prétraitement dynamique : Support pour l'augmentation de données
    """

    def __getitem__(self, idx):
        # 1. Lire la coupe CT
        ct_volume = nib.load(ct_path).get_fdata()
        slice_2d = ct_volume[:, :, slice_idx]

        # 2. Lire les 117 masques de segmentation
        masks = []
        for structure in all_structures:
            mask = nib.load(mask_path).get_fdata()
            masks.append(mask[:, :, slice_idx])

        # 3. Prétraitement
        ct_normalized = self.normalize(slice_2d)
        masks_stacked = np.stack(masks, axis=0)  # (117, H, W)

        return ct_normalized, masks_stacked
```

**Points clés de conception** :
- **Sélection de coupes** : Échantillonner plusieurs coupes par volume CT, augmente les échantillons d'entraînement
- **Masques multi-canaux** : Empiler 117 masques binaires en tenseur (117, H, W)
- **Optimisation mémoire** : Utiliser float32 au lieu de float64, réduit la mémoire de 50%

#### 1.2 Stratégie de Batch

```python
train_loader = DataLoader(
    dataset,
    batch_size=8,      # Pourquoi 8 ?
    shuffle=True,      # Pourquoi mélanger ?
    num_workers=4      # Pourquoi 4 workers ?
)
```

**Principe de sélection des paramètres** :
- **batch_size=8** :
  - Trop petit (1-2) : Bruit de gradient élevé, entraînement instable
  - Trop grand (32+) : Manque de VRAM, mise à jour lente du gradient
  - 8 est l'équilibre optimal pour GPU 8GB
- **shuffle=True** : Mélange pour éviter le surapprentissage d'un ordre spécifique
- **num_workers=4** : Chargement parallèle multi-thread, 50% des cœurs CPU

### 2. Détails d'Implémentation du Modèle

#### 2.1 Modules Cœur de U-Net

```python
class DoubleConv(nn.Module):
    """
    Principe du bloc de double convolution :
    Conv → BN → ReLU → Conv → BN → ReLU

    Pourquoi deux convolutions ?
    - Augmente la capacité de transformation non-linéaire
    - Élargit le champ récepteur
    - Design standard du papier original U-Net
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Stabilise l'entraînement
            nn.ReLU(inplace=True),         # Activation non-linéaire
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

**Pourquoi utiliser BatchNorm ?**
- Accélère la convergence (réduit le décalage de covariance interne)
- Permet un taux d'apprentissage plus élevé
- Effet de régularisation léger

**Pourquoi padding=1 ?**
- Maintient les dimensions spatiales (convolution 3×3 + padding=1 → taille inchangée)
- Évite la perte d'information aux frontières

#### 2.2 Connexion Encodeur-Décodeur

```python
class UNet(nn.Module):
    def forward(self, x):
        # Chemin encodeur - Extraction de caractéristiques
        x1 = self.enc1(x)          # (B, 32, 256, 256)
        x2 = self.enc2(self.pool(x1))  # (B, 64, 128, 128)
        x3 = self.enc3(self.pool(x2))  # (B, 128, 64, 64)
        x4 = self.enc4(self.pool(x3))  # (B, 256, 32, 32)

        # Couche goulot
        bottleneck = self.bottleneck(self.pool(x4))  # (B, 512, 16, 16)

        # Chemin décodeur - Reconstruction du masque
        u4 = self.up4(bottleneck)              # (B, 256, 32, 32)
        u4 = torch.cat([u4, x4], dim=1)        # Connexion de saut
        u4 = self.dec4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3(u3)

        # ... continuer sur-échantillonnage

        return self.final_conv(u1)  # (B, 117, 256, 256)
```

**Signification mathématique des connexions de saut** :
```
Sortie = Décodeur(Sur-échantillonné) + Encodeur(résolution originale)
            ↑                               ↑
    Caractéristiques sémantiques      Détails spatiaux
```

### 3. Stratégie d'Entraînement

#### 3.1 Configuration de l'Optimiseur

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,           # Pourquoi 0.001 ?
    betas=(0.9, 0.999),  # Pourquoi ces valeurs ?
    eps=1e-8,
    weight_decay=1e-5    # Régularisation L2
)
```

**Analyse des paramètres Adam** :
- **lr=1e-3** : Point de départ standard, ni trop grand (pas de divergence) ni trop petit (pas trop lent)
- **beta1=0.9** : Taux de décroissance exponentielle du moment d'ordre 1 (gradient)
- **beta2=0.999** : Taux de décroissance exponentielle du moment d'ordre 2 (carré du gradient)
- **weight_decay** : Prévient le surapprentissage, pénalise les grands poids

#### 3.2 Planification du Taux d'Apprentissage

```python
# Stratégie de taux d'apprentissage recommandée
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',        # Métrique à maximiser (coefficient Dice)
    factor=0.5,        # Réduction de 50%
    patience=3,        # Réduire après 3 époques sans amélioration
    verbose=True
)

# Dans la boucle d'entraînement
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(...)
    val_dice = validate(...)
    scheduler.step(val_dice)  # Ajustement selon métrique de validation
```

**Pourquoi la planification du taux d'apprentissage ?**
- Début : Grand taux d'apprentissage pour approcher rapidement l'optimum
- Fin : Petit taux d'apprentissage pour ajustement fin

### 4. Implémentation des Métriques d'Évaluation

#### 4.1 Coefficient de Dice

```python
def dice_coefficient(pred, target):
    """
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Signification physique : Degré de chevauchement entre prédiction et vérité terrain
    - Dice=1 : Chevauchement complet (segmentation parfaite)
    - Dice=0 : Aucun chevauchement (échec de segmentation)
    """
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()

    dice = (2.0 * intersection) / (union + 1e-8)  # Éviter division par zéro
    return dice
```

**Pourquoi Dice plutôt que l'exactitude ?**
```
Exemple : Image 100×100 pixels, organe occupe seulement 100 pixels
- Exactitude : Même en prédisant tout comme fond, exactitude = 99%
- Coefficient Dice : Prédire tout comme fond, Dice=0 (reflète correctement l'échec)
```

#### 4.2 IoU (Indice de Jaccard)

```python
def iou_score(pred, target):
    """
    IoU = |A ∩ B| / |A ∪ B|

    Relation avec Dice :
    IoU = Dice / (2 - Dice)
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    return iou
```

---

## Guide Pratique

### 1. Processus d'Entraînement Détaillé

#### 1.1 Boucle d'Entraînement Complète

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Mode entraînement (BatchNorm et Dropout se comportent différemment)
    epoch_loss = 0

    for batch_idx, (images, masks) in enumerate(loader):
        # 1. Transfert des données vers GPU
        images = images.to(device)  # (B, 1, 256, 256)
        masks = masks.to(device)    # (B, 117, 256, 256)

        # 2. Propagation avant
        optimizer.zero_grad()       # Réinitialiser les gradients
        outputs = model(images)     # (B, 117, 256, 256)

        # 3. Calcul de la perte
        loss = criterion(outputs, masks)

        # 4. Rétropropagation
        loss.backward()             # Calcul des gradients

        # 5. Écrêtage des gradients (optionnel, prévient l'explosion des gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Mise à jour des paramètres
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)
```

**Rôle de chaque étape** :
- **zero_grad()** : PyTorch accumule les gradients par défaut, réinitialisation manuelle nécessaire
- **loss.backward()** : Différenciation automatique, calcul des gradients pour tous les paramètres
- **optimizer.step()** : Mise à jour des paramètres selon les gradients

#### 1.2 Processus de Validation

```python
def validate(model, loader, device):
    model.eval()  # Mode évaluation (BatchNorm utilise stats globales, Dropout désactivé)
    dice_scores = []

    with torch.no_grad():  # Désactive le calcul des gradients, économise mémoire
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Conversion en probabilités

            # Calculer Dice pour chaque structure
            for i in range(117):
                dice = dice_coefficient(outputs[:, i], masks[:, i])
                dice_scores.append(dice.item())

    return np.mean(dice_scores)
```

**Pourquoi torch.no_grad() ?**
- Pas besoin de gradients en validation
- Économise 40-50% de mémoire
- Accélère l'inférence

### 2. Inférence et Prédiction

#### 2.1 Prédiction d'une Seule Image

```python
def predict_single_slice(model, ct_slice, device):
    """
    Entrée : Coupe CT brute (H, W)
    Sortie : 117 masques de segmentation (117, H, W)
    """
    model.eval()

    # 1. Prétraitement
    ct_normalized = normalize_ct(ct_slice)      # Normalisation valeurs HU
    ct_resized = resize(ct_normalized, (256, 256))  # Redimensionnement
    ct_tensor = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

    # 2. Inférence
    with torch.no_grad():
        output = model(ct_tensor.to(device))    # (1, 117, 256, 256)
        probs = torch.sigmoid(output)           # Conversion en probabilités

    # 3. Post-traitement
    masks_binary = (probs > 0.5).cpu().numpy()  # Binarisation
    masks_original_size = resize(masks_binary[0], ct_slice.shape)  # Taille originale

    return masks_original_size
```

#### 2.2 Prédiction par Lots et Reconstruction 3D

```python
def predict_volume(model, ct_volume, device):
    """
    Prédire un volume 3D entier
    """
    num_slices = ct_volume.shape[2]
    predictions = []

    for i in range(num_slices):
        slice_2d = ct_volume[:, :, i]
        pred_masks = predict_single_slice(model, slice_2d, device)
        predictions.append(pred_masks)

    # Empiler en volume 3D
    volume_3d = np.stack(predictions, axis=-1)  # (117, H, W, D)
    return volume_3d
```

### 3. Techniques de Visualisation

#### 3.1 Visualisation du Processus d'Entraînement

```python
import matplotlib.pyplot as plt

def plot_training_progress(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Courbe de perte
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Courbe Dice
    axes[1].plot(history['val_dice'], label='Val Dice', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
```

#### 3.2 Visualisation des Résultats de Segmentation

```python
def visualize_segmentation(ct_slice, pred_mask, gt_mask, structure_name):
    """
    Comparer prédiction et vérité terrain
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # CT original
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title('CT Scan')

    # Vérité terrain
    axes[1].imshow(ct_slice, cmap='gray')
    axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title('Vérité Terrain')

    # Prédiction
    axes[2].imshow(ct_slice, cmap='gray')
    axes[2].imshow(pred_mask, alpha=0.5, cmap='Blues')
    axes[2].set_title('Prédiction')

    # Superposition
    axes[3].imshow(ct_slice, cmap='gray')
    axes[3].imshow(gt_mask, alpha=0.3, cmap='Reds')
    axes[3].imshow(pred_mask, alpha=0.3, cmap='Blues')
    axes[3].set_title('Superposition')

    plt.suptitle(f'Segmentation : {structure_name}')
    plt.tight_layout()
```

---

## Optimisation des Performances

### 1. Optimisation Mémoire

#### 1.1 Entraînement en Précision Mixte

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(EPOCHS):
    for images, masks in train_loader:
        optimizer.zero_grad()

        # Propagation avant en float16
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Mise à l'échelle de la perte et rétropropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Effet** :
- Réduction de 40-50% de l'occupation VRAM
- Accélération de 2-3× (sur GPU avec Tensor Cores)
- Précision quasi-inchangée

#### 1.2 Accumulation de Gradients

```python
# Lorsque la VRAM ne permet pas un grand batch_size, accumuler les gradients
accumulation_steps = 4
optimizer.zero_grad()

for i, (images, masks) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss = loss / accumulation_steps  # Normalisation
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Équivalent à multiplier batch_size par 4**, sans augmenter la VRAM

### 2. Accélération de l'Entraînement

#### 2.1 Optimisation du Chargement des Données

```python
# Utiliser pin_memory pour accélérer le transfert GPU
train_loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,      # Mémoire épinglée, accélère CPU→GPU
    prefetch_factor=2     # Pré-chargement de 2 batches
)
```

#### 2.2 Compilation du Modèle (PyTorch 2.0+)

```python
# Utiliser torch.compile pour accélérer
model = torch.compile(model, mode='reduce-overhead')
```

**Effet** : Accélération de 10-30%

### 3. Optimisation de l'Inférence

#### 3.1 Quantification du Modèle

```python
# Convertir le modèle en précision int8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

**Effet** :
- Réduction de 4× de la taille du modèle
- Accélération de 2-3× de l'inférence
- Légère baisse de précision (1-2%)

#### 3.2 Inférence par Lots

```python
# Traiter plusieurs coupes à la fois
def batch_predict(model, slices, batch_size=16):
    predictions = []
    for i in range(0, len(slices), batch_size):
        batch = slices[i:i+batch_size]
        with torch.no_grad():
            preds = model(batch)
        predictions.append(preds)
    return torch.cat(predictions, dim=0)
```

---

## Résumé

### Philosophie de Conception

1. **Commencer Simple** : U-Net 2D est une excellente base pour le 3D
2. **Guidé par les Données** : Valider chaque décision de conception avec les données
3. **Interprétabilité** : Chaque hyperparamètre a une signification physique claire
4. **Praticité Ingénierie** : Équilibre entre performance et besoins en ressources

### Points Clés

| Décision | Raison | Compromis |
|----------|--------|-----------|
| Architecture U-Net | Meilleure pratique pour segmentation médicale | Relativement simple, mais suffisamment puissant |
| Coupes 2D | Efficace en ressources, facile à déboguer | Sacrifice d'une partie du contexte 3D |
| Perte BCE | Classification multi-label, entraînement stable | N'optimise pas directement Dice |
| Optimiseur Adam | Taux d'apprentissage adaptatif, robuste | Plus de mémoire que SGD |
| Évaluation Dice | Adapté aux données déséquilibrées | Calcul légèrement complexe |

### Directions d'Amélioration Futures

1. **Niveau modèle** : U-Net 3D, mécanismes d'attention, Transformer
2. **Niveau entraînement** : Augmentation de données, apprentissage curriculaire, entraînement adversarial
3. **Fonction de perte** : Combinaison Dice+BCE, Focal Loss
4. **Post-traitement** : Champs aléatoires conditionnels (CRF), optimisation morphologique

---

**Version** : 1.0
**Date de mise à jour** : Novembre 2025
**Auteur** : Claude Code
