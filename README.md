# Deep Neural Network Compression & CV

> **EE Association (EEA), IIT Kanpur** | Dec '25 – Apr '26

Implementation of the **[Deep Compression](https://arxiv.org/pdf/1510.00149)** pipeline from:

> **"Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"**  
> Song Han, Huizi Mao, William J. Dally — ICLR 2016

---

## Results at a Glance

| Metric | Value |
|---|---|
| **Original Model Size** | 37.22 MB |
| **Compressed Model Size** | 1.57 MB |
| **Compression Ratio** | **23.8×** |
| **Baseline Accuracy** | 90.67% |
| **Post-Compression Accuracy** | 89.99% |
| **Accuracy Drop** | **0.68%** |
| **Global Sparsity** | 89.9% |

---

## Pipeline Overview

```
Original Model ──► Stage 1: Pruning ──► Stage 2: Quantization ──► Stage 3: Huffman ──► Compressed Model
   37.22 MB         90% sparse            K-Means clusters          Lossless             1.57 MB
                    (10× from pruning)    (8-bit conv, 5-bit FC)   (+21.3% savings)     (23.8× total)
```

### Stage 1 — Magnitude-Based Network Pruning
- Global unstructured pruning: remove weights below a magnitude threshold (90% sparsity target)
- Fine-tune surviving weights for 25 epochs with cosine-annealing LR to recover accuracy
- Masks re-applied after every gradient step to keep pruned weights at zero

### Stage 2 — Trained Quantization via K-Means Weight Sharing
- **Conv layers:** 256 clusters (8-bit indices) | **FC layers:** 32 clusters (5-bit indices)
- Linear centroid initialisation (evenly spaced between min/max)
- Centroid fine-tuning via per-cluster gradient aggregation for 15 epochs

### Stage 3 — Huffman Coding
- Variable-length encoding of non-uniform cluster indices
- Exploits biased distribution — frequent indices get shorter codes
- Lossless and offline: no retraining needed, applied at the end
- Yielded **21.3% additional savings** on top of pruning + quantization

---

## Project Structure

```
├── README.md
├── Assignments/                              # Learning phase (7 assignments)
│   ├── Assignment_1_FFT_and_Frequency_Filtering.ipynb
│   ├── Assignment_2_Histograms_and_Color_Spaces.ipynb
│   ├── Assignment_3_Convolution_and_Edge_Detection.ipynb
│   ├── Assignment_4_Corner_Detection_and_Hough_Transform.ipynb
│   ├── Assignment_5_Linear_Logistic_Regression_KMeans.ipynb
│   ├── Assignment_6_Group_DataLoader_and_Feature_Extraction.ipynb
│   └── Assignment_7_MLP_from_Scratch.ipynb
│
├── MidEval_Report.pdf                        # Mid-semester evaluation report
├── Research_Paper_PPT.pdf                    # Presentation on the Han et al. paper
├── Research_Paper.pdf                        # Original Deep Compression paper
│
├── Project/                                  # Final project — compression pipeline
│   ├── config.py                             # All hyperparameters & paths
│   ├── main.py                               # End-to-end pipeline orchestrator
│   ├── data/
│   │   └── data_loader.py                    # CIFAR-10 loading + augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   └── vgg_cifar.py                      # VGG-11 adapted for CIFAR-10
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── pruning.py                        # Stage 1: Magnitude-based pruning
│   │   ├── quantization.py                   # Stage 2: K-Means weight sharing
│   │   └── huffman.py                        # Stage 3: Huffman coding
│   └── utils/
│       ├── __init__.py
│       └── metrics.py                        # Accuracy, sparsity, size metrics
│
├── EEA_Report.pdf                            # Final project report
└── Pipeline_Run_Output.txt                   # Complete output log of the compression run
```

---

## Assignments (Learning Phase)

The project had two phases — a **learning phase** with 7 graded assignments covering computer vision and ML fundamentals, followed by the **final project** implementing Deep Compression.

| # | Assignment | Topics Covered |
|---|---|---|
| 1 | FFT & Frequency Filtering | NumPy FFT, low-pass filter masks, phase/magnitude reconstruction |
| 2 | Histograms & Color Spaces | Grayscale histograms, RGB→HSV conversion, white balance |
| 3 | Convolution & Edge Detection | Custom convolution, Sobel edge detection, Laplacian sharpening |
| 4 | Corner Detection & Hough Transform | Harris corners, eigenvalue R-maps, Hough line voting |
| 5 | Linear/Logistic Regression & K-Means | ML from scratch (gradient descent, sigmoid, clustering) |
| 6 | Group: DataLoader & Feature Extraction | PyTorch DataLoader, Fruits-360 dataset, LBP, Canny features |
| 7 | MLP from Scratch | Multilayer Perceptron in PyTorch for non-linear classification |

---

## Model Architecture

VGG-11 adapted for 32×32 CIFAR-10 images (9,756,426 parameters):

| Block | Layers | Output Shape | Parameters |
|---|---|---|---|
| 1 | Conv2d(3→64) + BN + ReLU, MaxPool | 64×16×16 | 1,792 |
| 2 | Conv2d(64→128) + BN + ReLU, MaxPool | 128×8×8 | 73,984 |
| 3 | Conv2d(128→256) + Conv2d(256→256) + BN + ReLU, MaxPool | 256×4×4 | 886,016 |
| 4 | Conv2d(256→512) + Conv2d(512→512) + BN + ReLU, MaxPool | 512×2×2 | 3,540,992 |
| 5 | Conv2d(512→512) + Conv2d(512→512) + BN + ReLU, MaxPool | 512×1×1 | 4,720,640 |
| FC | Linear(512→512)×2 + Linear(512→10), ReLU, Dropout | 10 | 530,442 |

---

## Stage-wise Compression Results

| Stage | Accuracy | Compression |
|---|---|---|
| Baseline (100 epochs) | 90.67% | 1.0× |
| After Pruning (90% sparsity) | 10.00% | 10.0× |
| After Pruning + Fine-tune (25 epochs) | 90.21% | 10.0× |
| After Quantization + Fine-tune (15 epochs) | 89.99% | — |
| **After Huffman Coding (final)** | **89.99%** | **23.8×** |

---

## Requirements

```
torch >= 1.10
torchvision
numpy
scikit-learn
```

```bash
pip install torch torchvision numpy scikit-learn
```

## Usage

### Full pipeline (train baseline + compress):
```bash
cd Project
python main.py
```

### Skip training (load saved baseline):
```bash
python main.py --skip-training
```

### Custom settings:
```bash
python main.py --epochs 50 --prune-sparsity 0.85
```

---

## Key Paper References

| Section | Topic | Implementation |
|---|---|---|
| §2 | Network Pruning | `compression/pruning.py` |
| §3.1 | Weight Sharing via K-Means | `compression/quantization.py` |
| §3.2 | Linear Centroid Initialization | `compression/quantization.py` |
| §3.3 | Centroid Fine-tuning (Eq. 3) | `compression/quantization.py` |
| §4 | Huffman Coding | `compression/huffman.py` |
| §5 | Experiments / Evaluation | `utils/metrics.py` + `main.py` |

---

## References

1. Song Han, Huizi Mao, William J. Dally. *"Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding."* ICLR 2016. [arXiv:1510.00149](https://arxiv.org/pdf/1510.00149)
