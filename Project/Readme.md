
# Aerial Scene Classification Project (COMP9517 - Term 1, 2025)

This project explores various machine learning and deep learning methods to classify aerial scenes. Both classical feature-based models and modern convolutional neural networks (CNNs) are evaluated, including analysis under robust conditions such as noise, blur, and occlusion.

## Environment
- **CPU**: AMD RYZEN AI 9HX
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Platform**: Windows 11 + Python 3.9
- **Frameworks**: PyTorch, scikit-learn, OpenCV, matplotlib

## Requirements
```bash
pip install torch torchvision opencv-python matplotlib scikit-learn tqdm
```

## Models Implemented
- **LBP + KNN**
- **SIFT + SVM**
- **SIFT + SVM + SPM (Spatial Pyramid Matching)**
- **ResNet18** (w/ Grad-CAM)
- **EfficientNet-B0** (w/ Grad-CAM)
- **ResNet18-Beta** (noise, blur, occlusion)
- **EfficientNet-Beta** (noise, blur, occlusion)

## Dataset Structure
```plaintext
split_data/
├── train/
├── val/
└── test/
```
Each category contains 800 images (15 classes), split into:
- Train: 70%
- Val: 10%
- Test: 20%

## How to Run
1. Open `splitdata.ipynb` and run all cells. This decompresses the dataset `Aerial_Landscapes` and splits it into training/validation/testing sets. You can modify parameters to adjust the proportions.
2. Open each of the model-specific notebooks and run all code cells top-down:
   - `LBP+KNN.ipynb`
   - `SIFT+SVM.ipynb`
   - `SIFT+SPM+SVM.ipynb`
   - `ResNet.ipynb`
   - `EfficientNet.ipynb`
   - `ResNet-beta.ipynb` ✅ uses noise-corrupted data
   - `EfficientNet-beta.ipynb` ✅ uses noise-corrupted data
3. For robustness tests (`-beta` models), you **must** run `addnoise.ipynb` to apply noise, occlusion, and blur to the dataset before training.

## External Code / References
- **Grad-CAM** adapted from the paper ["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"](https://arxiv.org/abs/1610.02391)
- **SIFT** used via OpenCV
- **SPM** logic inspired by classical CV papers
- **EfficientNet** from `torchvision.models.efficientnet_b0`

## Performance Comparison (Standard Dataset)
| Model               | Accuracy | Precision | Recall | F1-score | Train Time (s) | Test Time (s) |
|--------------------|----------|-----------|--------|----------|----------------|----------------|
| LBP + KNN          | 50.41%   | 50.89%    | 50.09% | 49.91%   | 300             | 6            |
| SIFT + SVM         | 65.54%   | 65.72%    | 65.54% | 65.31%   | 420             | 7            |
| SIFT + SVM + SPM   | 68.54%   | 68.55%    | 68.54% | 68.31%   | 600             | 7            |
| ResNet18           | 90.92%   | 91.44%    | 90.92% | 90.91%   | 1020            | 9            |
| EfficientNet-B0    | 92.92%   | 93.15%    | 92.92% | 92.85%   | 1320            | 8            |
| ResNet18-Beta      | 90.58%   | 90.90%    | 90.58% | 90.41%   | 900             | 8            |
| EfficientNet-Beta  | 83.47%   | 83.30%    | 83.47% | 83.38%   | 1220            | 8            |

### Robustness Limitation in EfficientNet-B0 Due to Reloading and Resampling

During robustness testing, EfficientNet-B0 showed a notable drop in performance when trained and evaluated on corrupted datasets (`EfficientNet-Beta`). This can be attributed to two main factors:
1. **Sensitivity to Low-Level Perturbations**:  
   EfficientNet’s compound scaling and reliance on high-resolution features make it more vulnerable to Gaussian noise, blur, and occlusion. These distortions degrade the fine-grained patterns critical to its performance.

2. **Dataset Reloading and Resampling Effects**:  
   The corrupted dataset was regenerated via `addnoise.ipynb`, applying noise and then saving new images. This process introduced distribution shifts, compression artifacts, or inconsistencies that disproportionately affected EfficientNet, which relies heavily on pixel-level consistency.

In contrast, ResNet18 maintained more stable performance under these conditions, thanks to its simpler architecture and residual connections that offer better generalization.
This finding suggests that for real-world applications where input quality is unpredictable, EfficientNet may require enhanced robustness techniques such as advanced data augmentation, adversarial training, or regularization strategies.

## Notes
- Models were evaluated using the same train/val/test split.
- **Grad-CAM heatmaps** help us understand where the model "looks" when making decisions. Warm-colored areas in the heatmap indicate regions of higher model attention, helping to interpret whether the model's prediction was based on relevant parts of the image.
- `addnoise.ipynb` modifies the dataset for robustness testing as follows:
  - **Gaussian Noise** added to: `Agriculture`, `Airport`, `Beach`, `City`, `Desert`
  - **Occlusion** added to: `Forest`, `Grassland`, `Highway`, `Lake`, `Mountain`
  - **Gaussian Blur** added to: `Parking`, `Port`, `Railway`, `Residential`, `River`

