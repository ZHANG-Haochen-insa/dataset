# 肌肉分割训练系统 - 使用指南
# Système d'entraînement de segmentation musculaire - Guide d'utilisation

**版本 / Version**: 1.0 (肌肉专用版 / Version spécialisée muscles)
**更新日期 / Date de mise à jour**: 2025-12-15

---

## 目录 / Table des matières

1. [项目概述 / Présentation du projet](#1-项目概述--présentation-du-projet)
2. [目标肌肉结构 / Structures musculaires cibles](#2-目标肌肉结构--structures-musculaires-cibles)
3. [系统要求 / Configuration requise](#3-系统要求--configuration-requise)
4. [快速开始 / Démarrage rapide](#4-快速开始--démarrage-rapide)
5. [配置说明 / Instructions de configuration](#5-配置说明--instructions-de-configuration)
6. [训练监控 / Surveillance de l'entraînement](#6-训练监控--surveillance-de-lentraînement)
7. [输出文件 / Fichiers de sortie](#7-输出文件--fichiers-de-sortie)
8. [故障排查 / Dépannage](#8-故障排查--dépannage)

---

## 1. 项目概述 / Présentation du projet

### 中文说明

本项目实现了基于2D U-Net的**肌肉分割模型**，可从CT扫描图像中精确分割**10种肌肉结构**。

**主要特性：**
- 专注于肌肉组织分割
- 支持实时训练监控（Weights & Biases）
- 早停机制防止过拟合
- 准确率阈值自动停止
- 详细的性能指标记录

### Description en français

Ce projet implémente un **modèle de segmentation musculaire** basé sur U-Net 2D, capable de segmenter avec précision **10 structures musculaires** à partir d'images de scanner CT.

**Caractéristiques principales :**
- Focus sur la segmentation des tissus musculaires
- Surveillance en temps réel de l'entraînement (Weights & Biases)
- Mécanisme d'arrêt précoce pour éviter le surapprentissage
- Arrêt automatique au seuil de précision
- Enregistrement détaillé des métriques de performance

---

## 2. 目标肌肉结构 / Structures musculaires cibles

本系统分割以下10种肌肉结构：
Ce système segmente les 10 structures musculaires suivantes :

| 索引 Index | 英文 English | 中文 | Français |
|:----------:|:-------------|:-----|:---------|
| 0 | Autochthon Left | 左侧背部肌群 | Muscles autochtones gauches |
| 1 | Autochthon Right | 右侧背部肌群 | Muscles autochtones droits |
| 2 | Gluteus Maximus Left | 左臀大肌 | Grand fessier gauche |
| 3 | Gluteus Maximus Right | 右臀大肌 | Grand fessier droit |
| 4 | Gluteus Medius Left | 左臀中肌 | Moyen fessier gauche |
| 5 | Gluteus Medius Right | 右臀中肌 | Moyen fessier droit |
| 6 | Gluteus Minimus Left | 左臀小肌 | Petit fessier gauche |
| 7 | Gluteus Minimus Right | 右臀小肌 | Petit fessier droit |
| 8 | Iliopsoas Left | 左髂腰肌 | Ilio-psoas gauche |
| 9 | Iliopsoas Right | 右髂腰肌 | Ilio-psoas droit |

### 肌肉分组 / Groupes musculaires

#### 背部肌群 / Muscles du dos (Autochthon)
- **功能 / Fonction**: 维持脊柱稳定，支持身体姿势
- **Fonction**: Maintien de la stabilité de la colonne vertébrale, soutien de la posture

#### 臀肌群 / Muscles fessiers (Gluteus)
- **臀大肌 / Grand fessier**: 髋关节伸展，最大的臀部肌肉
- **Grand fessier**: Extension de la hanche, plus grand muscle fessier
- **臀中肌 / Moyen fessier**: 髋关节外展
- **Moyen fessier**: Abduction de la hanche
- **臀小肌 / Petit fessier**: 髋关节稳定
- **Petit fessier**: Stabilisation de la hanche

#### 髂腰肌 / Ilio-psoas
- **功能 / Fonction**: 屈髋，连接脊柱和下肢
- **Fonction**: Flexion de la hanche, connexion entre la colonne vertébrale et les membres inférieurs

---

## 3. 系统要求 / Configuration requise

### 硬件要求 / Matériel requis

| 组件 Composant | 最低要求 Minimum | 推荐配置 Recommandé |
|:--------------|:----------------|:-------------------|
| GPU显存 VRAM | 8 GB | 24 GB+ |
| 系统内存 RAM | 16 GB | 32 GB+ |
| 存储空间 Stockage | 50 GB | 100 GB+ |
| CPU | 4核 4 cores | 8核+ 8 cores+ |

### 软件要求 / Logiciels requis

```
Python >= 3.10
CUDA >= 11.0 (GPU训练 / pour l'entraînement GPU)
PyTorch >= 2.0.0
nibabel >= 3.2.0
scikit-image >= 0.19.0
wandb >= 0.15.0
matplotlib >= 3.5.0
tqdm >= 4.62.0
```

---

## 4. 快速开始 / Démarrage rapide

### 安装依赖 / Installation des dépendances

```bash
cd /local/hzhang02/data/dataset
pip install -r requirements.txt
```

### 数据结构 / Structure des données

确保数据按以下结构组织：
Assurez-vous que les données sont organisées comme suit :

```
/local/hzhang02/data/
├── s0000/
│   ├── ct.nii.gz                    # CT扫描 / Scan CT
│   └── segmentations/
│       ├── autochthon_left.nii.gz   # 左背部肌群
│       ├── autochthon_right.nii.gz  # 右背部肌群
│       ├── gluteus_maximus_left.nii.gz
│       ├── gluteus_maximus_right.nii.gz
│       ├── gluteus_medius_left.nii.gz
│       ├── gluteus_medius_right.nii.gz
│       ├── gluteus_minimus_left.nii.gz
│       ├── gluteus_minimus_right.nii.gz
│       ├── iliopsoas_left.nii.gz
│       └── iliopsoas_right.nii.gz
├── s0001/
│   └── ...
└── ...
```

### 开始训练 / Lancer l'entraînement

```bash
cd /local/hzhang02/data/dataset/scripts
python train_unet_enhanced.py
```

---

## 5. 配置说明 / Instructions de configuration

### 基础配置 / Configuration de base

在 `train_unet_enhanced.py` 中修改以下参数：
Modifier les paramètres suivants dans `train_unet_enhanced.py` :

```python
# 数据路径 / Chemins des données
DATA_ROOT = '/local/hzhang02/data'
OUTPUT_DIR = '/local/hzhang02/data/dataset/outputs'

# 训练参数 / Paramètres d'entraînement
TARGET_SHAPE = (256, 256)    # 图像尺寸 / Taille d'image
BATCH_SIZE = 16              # 批次大小 / Taille de lot
LEARNING_RATE = 1e-3         # 学习率 / Taux d'apprentissage
EPOCHS = 20                  # 最大轮数 / Nombre max d'époques
```

### 早停配置 / Configuration de l'arrêt précoce

```python
USE_EARLY_STOPPING = True    # 启用早停 / Activer l'arrêt précoce
EARLY_STOP_PATIENCE = 5      # 容忍轮数 / Nombre d'époques de patience
EARLY_STOP_MIN_DELTA = 0.001 # 最小改善阈值 / Seuil d'amélioration minimum
```

**工作原理 / Fonctionnement :**
- 监控验证集Dice系数
- Surveillance du coefficient Dice sur l'ensemble de validation
- 如果连续5个epoch无改善，自动停止
- Arrêt automatique si aucune amélioration pendant 5 époques

### 准确率阈值配置 / Configuration du seuil de précision

```python
USE_ACCURACY_THRESHOLD = True    # 启用阈值停止 / Activer l'arrêt au seuil
ACCURACY_THRESHOLD = 0.93        # 目标Dice / Dice cible
ACCURACY_THRESHOLD_PATIENCE = 2  # 确认轮数 / Époques de confirmation
```

**工作原理 / Fonctionnement :**
- 当Dice达到93%时开始计数
- Comptage lorsque le Dice atteint 93%
- 连续2轮保持则停止训练
- Arrêt si maintenu pendant 2 époques consécutives

### 学习率调度器 / Planificateur de taux d'apprentissage

```python
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'  # 'cosine' 或 'plateau'

# 余弦退火 / Recuit cosinus
COSINE_T_MAX = 20
COSINE_ETA_MIN = 1e-6

# 自适应降低 / Réduction adaptative
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 3
PLATEAU_MIN_LR = 1e-6
```

---

## 6. 训练监控 / Surveillance de l'entraînement

### 控制台输出示例 / Exemple de sortie console

```
============================================================
配置信息（肌肉分割专用版）:
  数据根目录: /local/hzhang02/data
  输出目录: /local/hzhang02/data/dataset/outputs
  目标结构: 10种肌肉
  批次大小: 16
  学习率: 0.001
============================================================

目标肌肉列表:
  [0] 左侧背部肌群 / Muscles autochtones gauches
  [1] 右侧背部肌群 / Muscles autochtones droits
  [2] 左臀大肌 / Grand fessier gauche
  [3] 右臀大肌 / Grand fessier droit
  [4] 左臀中肌 / Moyen fessier gauche
  [5] 右臀中肌 / Moyen fessier droit
  [6] 左臀小肌 / Petit fessier gauche
  [7] 右臀小肌 / Petit fessier droit
  [8] 左髂腰肌 / Ilio-psoas gauche
  [9] 右髂腰肌 / Ilio-psoas droit

============================================================
Epoch 10/20
============================================================
Training: 100%|██████████| 1250/1250 [1:30:00<00:00]
Validation: 100%|██████████| 312/312 [0:15:00<00:00]

Epoch 10 结果:
  训练损失: 0.0015 | 训练Dice: 0.9380
  验证损失: 0.0022 | 验证Dice: 0.9250 | 验证IoU: 0.9120
  过拟合差距: 0.0130 (正常)

  表现最好的肌肉:
    gluteus_maximus_right: 0.9645
    gluteus_maximus_left: 0.9623
    iliopsoas_right: 0.9512
```

### Weights & Biases 监控 / Surveillance Weights & Biases

训练开始后，访问 wandb.ai 查看：
Après le début de l'entraînement, consultez wandb.ai pour :

- **实时曲线 / Courbes en temps réel**: 训练/验证损失和Dice
- **肌肉性能 / Performance musculaire**: 每种肌肉的分割效果
- **学习率 / Taux d'apprentissage**: 调度器变化曲线
- **预测可视化 / Visualisation des prédictions**: 样本分割结果

---

## 7. 输出文件 / Fichiers de sortie

训练过程会生成以下文件：
L'entraînement génère les fichiers suivants :

```
outputs/
├── checkpoint_enhanced_epoch{N}.pth  # 每轮检查点 / Point de contrôle
├── best_model.pth                    # 最佳模型 / Meilleur modèle
├── training_history_enhanced.json    # 训练历史 / Historique
├── training_history_enhanced.png     # 训练曲线 / Courbes
├── test_inference_enhanced.png       # 测试结果 / Résultats de test
├── sample_visualization_enhanced.png # 样本可视化 / Visualisation
└── label_map.json                    # 标签映射 / Mapping des labels
```

### 加载训练好的模型 / Charger le modèle entraîné

```python
import torch
from train_unet_enhanced import UNet2D

# 加载模型 / Charger le modèle
checkpoint = torch.load('outputs/best_model.pth')
model = UNet2D(in_ch=1, out_ch=10, features=[32, 64, 128, 256])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"最佳Dice / Meilleur Dice: {checkpoint['val_dice']:.4f}")
print(f"训练轮数 / Époques: {checkpoint['epoch']}")
```

---

## 8. 故障排查 / Dépannage

### 问题1：显存不足 / Problème 1 : Mémoire GPU insuffisante

**症状 / Symptôme**: `RuntimeError: CUDA out of memory`

**解决方案 / Solution**:
```python
BATCH_SIZE = 8           # 减小批次 / Réduire la taille de lot
TARGET_SHAPE = (128, 128) # 减小图像尺寸 / Réduire la taille d'image
```

### 问题2：找不到肌肉数据 / Problème 2 : Données musculaires introuvables

**症状 / Symptôme**: `肌肉结构数量: 0`

**解决方案 / Solution**:
- 检查分割文件是否存在于 `segmentations/` 目录
- Vérifier que les fichiers de segmentation existent dans `segmentations/`
- 确认文件命名正确（如 `gluteus_maximus_left.nii.gz`）
- Confirmer le nommage correct des fichiers

### 问题3：训练过早停止 / Problème 3 : Arrêt prématuré de l'entraînement

**症状 / Symptôme**: 只训练了2-3个epoch

**解决方案 / Solution**:
```python
USE_ACCURACY_THRESHOLD = False  # 暂时禁用 / Désactiver temporairement
# 或 / ou
ACCURACY_THRESHOLD = 0.95       # 提高阈值 / Augmenter le seuil
```

### 问题4：Weights & Biases连接失败 / Problème 4 : Échec de connexion W&B

**解决方案 / Solution**:
```python
USE_WANDB = False  # 禁用wandb / Désactiver wandb
```

---

## 附录：性能基准 / Annexe : Benchmarks de performance

基于肌肉分割的预期性能指标：
Métriques de performance attendues pour la segmentation musculaire :

| 肌肉 Muscle | 预期Dice Dice attendu | 难度 Difficulté |
|:-----------|:---------------------|:---------------|
| 臀大肌 Gluteus Maximus | 0.95+ | 低 Faible |
| 臀中肌 Gluteus Medius | 0.92+ | 中 Moyenne |
| 臀小肌 Gluteus Minimus | 0.88+ | 高 Élevée |
| 髂腰肌 Iliopsoas | 0.90+ | 中 Moyenne |
| 背部肌群 Autochthon | 0.85+ | 高 Élevée |

**注意 / Note**:
- 大块肌肉（如臀大肌）通常更容易分割
- Les muscles volumineux (comme le grand fessier) sont généralement plus faciles à segmenter
- 深层小肌肉（如臀小肌）分割难度较高
- Les petits muscles profonds (comme le petit fessier) sont plus difficiles à segmenter

---

**维护者 / Mainteneur**: hzhang02
**最后更新 / Dernière mise à jour**: 2025-12-15
