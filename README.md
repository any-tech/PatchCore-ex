# PatchCore with Explainability
This is an unofficial implementation of the paper [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/pdf/2106.08265.pdf).
<img src="https://github.com/any-tech/PatchCore-ex/assets/9916906/aeb30f91-1a42-4075-8055-e7982ad63c99" width=400 link="hoge.com">

We ...

This code was implimented with [patchcore-inspection](https://github.com/amazon-science/patchcore-inspection), thanks.

## Prerequisites

- faiss-gpu (easy to install with conda : [ref](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md))
- torch
- torchvision
- numpy
- opencv-python
- scipy
- argparse
- matplotlib
- scikit-learn
- torchinfo
- tqdm


Install prerequisites with:  
```
conda install --file requirements.txt
```

<br/>

Please download [`MVTec AD`](https://www.mvtec.com/company/research/datasets/mvtec-ad/) dataset.

After downloading, place the data as follows:
```
./
├── main.py
└── mvtec_anomaly_detection
    ├── bottle
    ├── cable
    ├── capsule
    ├── carpet
    ├── grid
    ├── hazelnut
    ├── leather
    ├── metal_nut
    ├── pill
    ├── screw
    ├── tile
    ├── toothbrush
    ├── transistor
    ├── wood
    └── zipper
```

<br/>

## Usage

To test **SPADE** on `MVTec AD` dataset:
```
python main.py
```

After running the code above, you can see the ROCAUC results in `result/roc_curve.png`

<br/>

## Results

Below is the implementation result of the test set ROCAUC on the `MVTec AD` dataset.  

### 1. Image-level anomaly detection accuracy (ROCAUC %)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | - | 96.5 | 96.6 | 95.7 |
| cable | - | 84.7 | 84.7 | 80.9 |
| capsule | - | 90.5 | 89.7 | 81.8 |
| carpet | - | 92.4 | 92.5 | 91.8 |
| grid | - | 49.5 | 45.7 | 33.0 |
| hazelnut | - | 89.3 | 88.8 | 85.3 |
| leather | - | 94.9 | 94.6 | 92.8 |
| metal_nut | - | 71.2 | 70.0 | 62.1 |
| pill | - | 79.9 | 79.2 | 78.0 |
| screw | - | 67.1 | 65.3 | 49.8 |
| tile | - | 96.7 | 96.4 | 95.8 |
| toothbrush | - | 86.7 | 86.9 | 75.6 |
| transistor | - | 90.4 | 90.0 | 87.4 |
| wood | - | 97.0 | 96.8 | 96.6 |
| zipper | - | 96.4 | 96.4 | 95.6 |
| Average | 85.5 | 85.5 | 84.9 | 80.1 |

<br/>

### 2. Pixel-level anomaly detection accuracy (ROCAUC %)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | 98.4 | 97.0 | 97.2 | 97.7 |
| cable | 97.2 | 92.7 | 93.4 | 94.5 |
| capsule | 99.0 | 98.2 | 98.3 | 98.6 |
| carpet | 97.5 | 98.9 | 98.9 | 99.0 |
| grid | 93.7 | 96.8 | 98.2 | 98.6 |
| hazelnut | 99.1 | 98.3 | 98.4 | 98.6 |
| leather | 97.6 | 99.2 | 99.2 | 99.2 |
| metal_nut | 98.1 | 96.8 | 97.0 | 97.3 |
| pill | 96.5 | 94.3 | 94.7 | 95.5 |
| screw | 98.9 | 99.0 | 99.1 | 99.3 |
| tile | 87.4 | 92.4 | 92.7 | 93.7 |
| toothbrush | 97.9 | 98.8 | 98.9 | 98.9 |
| transistor | 94.1 | 87.4 | 88.7 | 90.9 |
| wood | 88.5 | 94.8 | 94.9 | 95.2 |
| zipper | 96.5 | 98.3 | 98.5 | 98.8 |
| Average | 96.0 | 96.2 | 96.5 | 97.1 |

<br/>

### 3. Processing time (sec)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | - | 6.6 | 7.2 | 16.4 |
| cable | - | 13.2 | 13.2 | 30.1 |
| capsule | - | 11.6 | 11.9 | 25.3 |
| carpet | - | 12.0 | 11.6 | 26.2 |
| grid | - | 7.4 | 7.5 | 17.7 |
| hazelnut | - | 12.4 | 12.3 | 26.4 |
| leather | - | 10.9 | 10.8 | 26.8 |
| metal_nut | - | 8.3 | 8.6 | 23.1 |
| pill | - | 13.2 | 12.7 | 34.1 |
| screw | - | 11.7 | 11.7 | 28.4 |
| tile | - | 10.3 | 9.9 | 22.2 |
| toothbrush | - | 3.7 | 3.6 | 7.9 |
| transistor | - | 9.6 | 9.7 | 17.7 |
| wood | - | 9.9 | 9.4 | 17.8 |
| zipper | - | 11.4 | 11.4 | 26.6 |
| Average | - | 10.1 | 10.1 | 23.1 |

```
CPU : Intel Xeon Platinum 8360Y
GPU : NVIDIA A100 SXM4
```

<br/>

### ROC Curve 
