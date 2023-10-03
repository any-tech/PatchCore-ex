# PatchCore with some "ex"
This is an unofficial implementation of the paper [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/pdf/2106.08265.pdf).
<img src="https://github.com/any-tech/PatchCore-ex/blob/main/assets/total_recall.jpg" width=400 link="hoge.com">

We measured accuracy and speed for percentage_coreset=0.01, 0.1 and 0.25.

This code was implimented with [patchcore-inspection](https://github.com/amazon-science/patchcore-inspection), thanks.

Some "ex" are **ex**plainability, **ex**press delivery, fl**ex**ibility, **ex**tra algolithm and so on.

<br/>

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

To test **PatchCore** on `MVTec AD` dataset:
```
python main.py
```

After running the code above, you can see the ROCAUC results in `result/roc_curve.png`

<br/>

## Results

Below is the implementation result of the test set ROCAUC on the `MVTec AD` dataset.  

### 1. Image-level anomaly detection accuracy (ROCAUC %)

| | Paper<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.01 | This Repo<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.25 |
| - | - | - | - | - |
| bottle | 100.0 | 100.0 | 100.0 | 100.0 |
| cable | 99.4 | 99.6 | 99.9 | 99.6 |
| capsule | 97.8 | 97.6 | 97.8 | 97.8 |
| carpet | 98.7 | 98.3 | 98.7 | 98.5 |
| grid | 97.9 | 97.8 | 98.5 | 98.4 |
| hazelnut | 100.0 | 100.0 | 100.0 | 100.0 |
| leather | 100.0 | 100.0 | 100.0 | 100.0 |
| metal_nut | 100.0 | 100.0 | 100.0 | 99.9 |
| pill | 96.0 | 96.3 | 96.2 | 96.4 |
| screw | 97.0 | 97.9 | 98.5 | 98.1 |
| tile | 98.9 | 99.0 | 99.4 | 99.2 |
| toothbrush | 99.7 | 99.4 | 99.4 | 100.0 |
| transistor | 100.0 | 99.8 | 100.0 | 100.0 |
| wood | 99.0 | 99.1 | 99.0 | 98.9 |
| zipper | 99.5 | 99.7 | 99.6 | 99.5 |
| Average | 99.0 | 99.0 | 99.1 | 99.1 |

<br/>

### 2. Pixel-level anomaly detection accuracy (ROCAUC %)

| | Paper<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.01 | This Repo<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.25 |
| - | - | - | - | - |
| bottle | 98.6 | 98.5 | 98.6 | 98.6 |
| cable | 98.5 | 98.2 | 98.4 | 98.4 |
| capsule | 98.9 | 98.8 | 98.9 | 98.9 |
| carpet | 99.1 | 99.0 | 99.1 | 99.0 |
| grid | 98.7 | 98.2 | 98.7 | 98.7 |
| hazelnut | 98.7 | 98.6 | 98.7 | 98.6 |
| leather | 99.3 | 99.3 | 99.3 | 99.3 |
| metal_nut | 98.4 | 98.5 | 98.7 | 98.7 |
| pill | 97.6 | 97.4 | 97.6 | 97.3 |
| screw | 99.4 | 98.8 | 99.4 | 99.4 |
| tile | 95.9 | 96.2 | 96.1 | 95.9 |
| toothbrush | 98.7 | 98.6 | 98.7 | 98.7 |
| transistor | 96.4 | 94.2 | 96.0 | 95.8 |
| wood | 95.1 | 95.7 | 95.5 | 95.4 |
| zipper | 98.9 | 98.9 | 98.9 | 98.9 |
| Average | 98.1 | 97.9 | 98.2 | 98.1 |

<br/>

### 3. Processing time (sec)

| | This Repo<br/>$\\%_{core}$=0.01 | This Repo<br/>$\\%_{core}$=0.1 | This Repo<br/>$\\%_{core}$=0.25 |
| - | - | - | - |
| bottle | 14.2 | 21.3 | 33.0 |
| cable | 22.9 | 30.3 | 44.6 |
| capsule | 20.4 | 27.3 | 39.8 |
| carpet | 22.4 | 32.6 | 52.7 |
| grid | 16.3 | 26.3 | 44.3 |
| hazelnut | 25.9 | 46.4 | 82.1 |
| leather | 20.8 | 28.5 | 44.0 |
| metal_nut | 16.8 | 23.8 | 36.3 |
| pill | 23.3 | 34.2 | 52.6 |
| screw | 22.4 | 37.7 | 63.9 |
| tile | 18.4 | 26.2 | 40.7 |
| toothbrush | 5.8 | 6.7 | 8.0 |
| transistor | 17.2 | 23.9 | 35.9 |
| wood | 18.0 | 26.5 | 42.1 |
| zipper | 20.5 | 29.0 | 44.2 |

```
CPU : Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
GPU : Tesla V100 SXM2
```

<br/>

### ROC Curve 

- percentage_coreset = 0.01
![roc](./assets/roc-curve_p0010_k01_rim0990_rpm0979.png)

<br/>

- percentage_coreset = 0.1
![roc](./assets/roc-curve_p0100_k01_rim0991_rpm0982.png)

<br/>

- percentage_coreset = 0.25
![roc](./assets/roc-curve_p0250_k01_rim0991_rpm0981.png)

<br/>

### Prediction Distribution (percentage_coreset = 0.1)

- bottle
![bottle](./assets/pred-dist_bottle_p0100_k01_r1000.png)

- cable
![cable](./assets/pred-dist_cable_p0100_k01_r0999.png)

- capsule
![capsule](./assets/pred-dist_capsule_p0100_k01_r0978.png)

- carpet
![carpet](./assets/pred-dist_carpet_p0100_k01_r0987.png)

- grid
![grid](./assets/pred-dist_grid_p0100_k01_r0985.png)

- hazelnut
![hazelnut](./assets/pred-dist_hazelnut_p0100_k01_r1000.png)

- leather
![leather](./assets/pred-dist_leather_p0100_k01_r1000.png)

- metal_nut
![metal_nut](./assets/pred-dist_metal_nut_p0100_k01_r1000.png)

- pill
![pill](./assets/pred-dist_pill_p0100_k01_r0962.png)

- screw
![screw](./assets/pred-dist_screw_p0100_k01_r0985.png)

- tile
![tile](./assets/pred-dist_tile_p0100_k01_r0994.png)

- toothbrush
![toothbrush](./assets/pred-dist_toothbrush_p0100_k01_r0994.png)

- transistor
![transistor](./assets/pred-dist_transistor_p0100_k01_r1000.png)

- wood
![wood](./assets/pred-dist_wood_p0100_k01_r0990.png)

- zipper
![zipper](./assets/pred-dist_zipper_p0100_k01_r0996.png)

<br/>

### Localization : percentage_coreset = 0.1

- bottle (test case : broken_large)
![bottle](./assets/localization_bottle_broken_large_000_p0100_k01_s067.png)

- cable (test case : bent_wire)
![cable](./assets/localization_cable_bent_wire_000_p0100_k01_s073.png)

- capsule (test case : crack)
![capsule](./assets/localization_capsule_crack_000_p0100_k01_s020.png)

- carpet (test case : color)
![carpet](./assets/localization_carpet_color_000_p0100_k01_s070.png)

- grid (test case : bent)
![grid](./assets/localization_grid_bent_000_p0100_k01_s048.png)

- hazelnut (test case : crack)
![hazelnut](./assets/localization_hazelnut_crack_000_p0100_k01_s047.png)

- leather (test case : color)
![leather](./assets/localization_leather_color_000_p0100_k01_s058.png)

- metal_nut (test case : bent)
![metal_nut](./assets/localization_metal_nut_bent_000_p0100_k01_s082.png)

- pill (test case : color)
![pill](./assets/localization_pill_color_000_p0100_k01_s036.png)

- screw (test case : manipulated_front)
![screw](./assets/localization_screw_manipulated_front_000_p0100_k01_s047.png)

- tile (test case : crack)
![tile](./assets/localization_tile_crack_000_p0100_k01_s059.png)

- toothbrush (test case : defective)
![toothbrush](./assets/localization_toothbrush_defective_000_p0100_k01_s079.png)

- transistor (test case : bent_lead)
![transistor](./assets/localization_transistor_bent_lead_000_p0100_k01_s043.png)

- wood (test case : color)
![wood](./assets/localization_wood_color_000_p0100_k01_s063.png)

- zipper (test case : broken_teeth)
![zipper](./assets/localization_zipper_broken_teeth_000_p0100_k01_s029.png)

<br/>

### For your infomation

We also implement a similar algorithm, SPADE.<br/>
https://github.com/any-tech/SPADE-fast/tree/main

There is an explanatory article.<br/>
https://tech.anytech.co.jp/entry/2023/03/24/100000
