# PatchCore with Explainability
This is an unofficial implementation of the paper [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/pdf/2106.08265.pdf).
<img src="https://github.com/any-tech/PatchCore-ex/blob/main/assets/total_recall.jpg" width=400 link="hoge.com">

We measured accuracy and speed for percentage_coreset=0.01, 0.1 and 0.25.

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

$\%$

| | Paper<br/>\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.01 | This Repo<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.25 |
| - | - | - | - | - |
| bottle | 100.0 | xxx | xxx | xxx |
| cable | 99.4 | xxx | xxx | xxx |
| capsule | 97.8 | xxx | xxx | xxx |
| carpet | 98.7 | xxx | xxx | xxx |
| grid | 97.9 | xxx | xxx | xxx |
| hazelnut | 100.0 | xxx | xxx | xxx |
| leather | 100.0 | xxx | xxx | xxx |
| metal_nut | 100.0 | xxx | xxx | xxx |
| pill | 96.0 | xxx | xxx | xxx |
| screw | 97.0 | xxx | xxx | xxx |
| tile | 98.9 | xxx | xxx | xxx |
| toothbrush | 99.7 | xxx | xxx | xxx |
| transistor | 100.0 | xxx | xxx | xxx |
| wood | 99.0 | xxx | xxx | xxx |
| zipper | 99.5 | xxx | xxx | xxx |
| Average | 99.0 | xxx | xxx | xxx |

<br/>

### 2. Pixel-level anomaly detection accuracy (ROCAUC %)

| | Paper<br/>\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.01 | This Repo<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.25 |
| - | - | - | - | - |
| bottle | 98.6 | xxx | xxx | xxx |
| cable | 98.5 | xxx | xxx | xxx |
| capsule | 98.9 | xxx | xxx | xxx |
| carpet | 99.1 | xxx | xxx | xxx |
| grid | 98.7 | xxx | xxx | xxx |
| hazelnut | 98.7 | xxx | xxx | xxx |
| leather | 99.3 | xxx | xxx | xxx |
| metal_nut | 98.4 | xxx | xxx | xxx |
| pill | 97.6 | xxx | xxx | xxx |
| screw | 99.4 | xxx | xxx | xxx |
| tile | 95.9 | xxx | xxx | xxx |
| toothbrush | 98.7 | xxx | xxx | xxx |
| transistor | 98.7 | xxx | xxx | xxx |
| wood | 95.1 | xxx | xxx | xxx |
| zipper | 98.9 | xxx | xxx | xxx |
| Average | 98.1 | xxx | xxx | xxx |

<br/>

### 3. Processing time (sec)

| | Paper<br/>\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.01 | This Repo<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.25 |
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
CPU : Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
GPU : Tesla V100 SXM2
```

<br/>

### ROC Curve 

- percentage_coreset = 0.01
![roc](./assets/roc-curve_p0010_k01.png)

<br/>

- percentage_coreset = 0.1
![roc](./assets/roc-curve_p0100_k01.png)

<br/>

- percentage_coreset = 0.25
![roc](./assets/roc-curve_p0100_k05.png)

<br/>

### Prediction Distribution (percentage_coreset = 0.1)

- bottle
![bottle](./assets/pred-dist_p0100_k01_bottle.png)

- cable
![cable](./assets/pred-dist_p0100_k01_cable.png)

- capsule
![capsule](./assets/pred-dist_p0100_k01_capsule.png)

- carpet
![carpet](./assets/pred-dist_p0100_k01_carpet.png)

- grid
![grid](./assets/pred-dist_p0100_k01_grid.png)

- hazelnut
![hazelnut](./assets/pred-dist_p0100_k01_hazelnut.png)

- leather
![leather](./assets/pred-dist_p0100_k01_leather.png)

- metal_nut
![metal_nut](./assets/pred-dist_p0100_k01_metal_nut.png)

- pill
![pill](./assets/pred-dist_p0100_k01_pill.png)

- screw
![screw](./assets/pred-dist_p0100_k01_screw.png)

- tile
![tile](./assets/pred-dist_p0100_k01_tile.png)

- toothbrush
![toothbrush](./assets/pred-dist_p0100_k01_toothbrush.png)

- transistor
![transistor](./assets/pred-dist_p0100_k01_transistor.png)

- wood
![wood](./assets/pred-dist_p0100_k01_wood.png)

- zipper
![zipper](./assets/pred-dist_p0100_k01_zipper.png)

<br/>

### Localization : percentage_coreset = 0.1

- bottle (test case : broken_large)
![bottle](./assets/localization_p0100_k01_bottle_broken_large_000_s067.png)

- cable (test case : bent_wire)
![cable](./assets/localization_p0100_k01_cable_bent_wire_000_s073.png)

- capsule (test case : crack)
![capsule](./assets/localization_p0100_k01_capsule_crack_000_s020.png)

- carpet (test case : color)
![carpet](./assets/localization_p0100_k01_carpet_color_000_s070.png)

- grid (test case : bent)
![grid](./assets/localization_p0100_k01_grid_bent_000_s048.png)

- hazelnut (test case : crack)
![hazelnut](./assets/localization_p0100_k01_hazelnut_crack_000_s047.png)

- leather (test case : color)
![leather](./assets/localization_p0100_k01_leather_color_000_s058.png)

- metal_nut (test case : bent)
![metal_nut](./assets/localization_p0100_k01_metal_nut_bent_000_s082.png)

- pill (test case : color)
![pill](./assets/localization_p0100_k01_pill_color_000_s036.png)

- screw (test case : manipulated_front)
![screw](./assets/localization_p0100_k01_screw_manipulated_front_000_s047.png)

- tile (test case : crack)
![tile](./assets/localization_p0100_k01_tile_crack_000_s059.png)

- toothbrush (test case : defective)
![toothbrush](./assets/localization_p0100_k01_toothbrush_defective_000_s079.png)

- transistor (test case : bent_lead)
![transistor](./assets/localization_p0100_k01_transistor_bent_lead_000_s043.png)

- wood (test case : color)
![wood](./assets/localization_p0100_k01_wood_color_000_s063.png)

- zipper (test case : broken_teeth)
![zipper](./assets/localization_p0100_k01_zipper_broken_teeth_000_s029.png)


