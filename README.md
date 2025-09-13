# Scribble-Supervised Learning for Medical Image Segmentation

This repository provides re-implementations of some papers about scribble-supervised  for medical image segmentation:


| #  | Paper                                                                                   | Venue/Year         | Status                          |
|----|-----------------------------------------------------------------------------------------|--------------------|-------2022        | ✅                              |
| 2  | [ShapePU](https://arxiv.org/pdf/2206.02118)                                             | MICCAI 2022        | ⚠️ (Bug)                        |
| 3  | [UAMT](https://www.sciencedirect.com/science/article/pii/S0031320321005215)             | Pattern Recognition 2022 | ✅ (Code xong, chưa chạy)   |--------------------------|
| 1  | [DMPLS](https://arxiv.org/pdf/2203.02106)                                               | MICCAI 
| 4  | [ScribbleVC](https://arxiv.org/pdf/2307.16226)                                          | ACM MM 2023        | ✅                              |
| 5  | [ScribFormer](https://arxiv.org/pdf/2402.02029)                                         | IEEE TMI 2024      | ✅                              |
| 6  | [DMSPS](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001993?dgcid=author) | MedIA 2024        | ✅ (Xog Stage1, còn Stage2)  |
| 7  | [ScribbleVS](https://arxiv.org/pdf/2411.10237)                                          | ArXiv 2024         | ✅                              |
| 8  | [TABNet](https://arxiv.org/pdf/2507.02399)                                              | ArXiv 2025         |  ✅                    |

### Benchmark on ACDC

## 📊 Quantitative Results on ACDC

## 📊 Quantitative Results on ACDC

| Method    | LV Dice ↑ | LV HD95 ↓ | RV Dice ↑ | RV HD95 ↓ | MYO Dice ↑ | MYO HD95 ↓ | **Mean Dice ↑** | **Mean HD95 ↓** |
|-----------|-----------|-----------|-----------|-----------|-------------|-------------|-----------------|-----------------|
| **TABNet** | 0.882     | 1.818     | 0.868     | 1.244     | 0.928       | 2.476       | **0.892**       | **1.846**       |
| **DMSPS**  | 0.880     | 1.503     | 0.851     | 5.899     | 0.923       | 6.548       | **0.885**       | **4.650**       |
| **DMPLS**  | 0.872     | 1.760     | 0.842     | 9.314     | 0.917       | 6.605       | **0.877**       | **5.893**       |




### Tasks
- Fix bug ShapePU
- Run UAMT
- Run DMSPS Stage2
- Run test for all methods

### Acknowledgement
This repo partially uses code from [Hilab-WSL4MIS](https://github.com/HiLab-git/WSL4MIS) and [ShapePU](https://github.com/BWGZK/ShapePU)