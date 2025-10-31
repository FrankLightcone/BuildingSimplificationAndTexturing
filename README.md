# 🏙️ A Raster-Based Method for Building Simplification and Texturing

> **A novel raster-based approach for building simplification and texture preservation in remote sensing imagery, integrating superpixel segmentation, texture matching, and hue adjustment.**

**Authors:** Ruijie Fan, Yilang Shen  
**Affiliation:** School of Geospatial Engineering and Science, Sun Yat-sen University  
**Published in:** *Geo-spatial Information Science* (2025)  
**DOI:** [10.1080/10095020.2025.2573369](https://doi.org/10.1080/10095020.2025.2573369)

---

## 🌍 Overview

This repository provides the implementation and experimental data of the paper  
**“A raster-based method for building simplification considering shape and texture features based on remote sensing images.”**

The proposed **Shape and Texture-Aware Building Simplification (STABS)** method introduces a raster-based approach for simplifying buildings while preserving visual continuity in remote sensing imagery. Unlike traditional vector-based or textureless raster simplification methods, STABS simultaneously considers both **shape** and **texture** information to maintain the perceptual consistency of buildings across multiple scales.

---

## 🚀 Key Contributions

- **Texture-aware building simplification:**  
  Introduces a method that assigns suitable textures from a pre-built texture library by comparing the **GLCM** and **LBP** feature similarities between the original building and library textures.

- **Hue-adjusted texture matching:**  
  Applies hue correction to matched textures to ensure color consistency between simplified and original buildings, enhancing **cartographic continuity**.

- **Superpixel-based segmentation:**  
  Employs the **SEEDS** algorithm for superpixel segmentation, followed by evaluation using **corner ratio (CR)** and **square ratio (SR)** to retain geometrically meaningful building parts.

- **Improved raster map generalization:**  
  The method provides visually stable and structurally faithful simplification results that integrate smoothly back into remote sensing imagery.

---

## 📊 Experimental Data

- **Dataset:**  
  Building imagery from the [Wuhan University Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html), based on aerial data from Land Information New Zealand (LINZ).

- **Texture Library:**  
  Constructed from **50 building textures** collected via [Maps of Switzerland](https://map.geo.admin.ch/), processed with the *Image Quilting for Texture Synthesis* method.

- **Evaluation Metrics:**  
  Quantitative comparison using **MSE**, **PSNR**, **MS-SSIM**, **Gradient Similarity (GS)**, and **Edge Preservation Index (EPI)** demonstrates that STABS achieves higher similarity to the original imagery than Gaussian filtering and traditional methods.

---

## 🧩 Repository Contents

```bash
├── data/ # Sample dataset and building labels
├── textures/ # Building texture library
├── src/ # Source code for STABS implementation
│ ├── segmentation/ # Superpixel extraction (SEEDS)
│ ├── simplification/ # CR/SR-based building simplification
│ ├── texture_matching/ # GLCM + LBP texture comparison
│ └── hue_adjustment/ # Hue correction and compositing
├── results/ # Experimental outputs and comparisons
└── README.md # Project documentation
```

---

## 🧠 Citation

If you use this work, please cite:

> **Fan, R., & Shen, Y.** (2025). *A raster-based method for building simplification considering shape and texture features based on remote sensing images.*  
> *Geo-spatial Information Science.* DOI: [10.1080/10095020.2025.2573369](https://doi.org/10.1080/10095020.2025.2573369)

---

## 🔮 Future Work

- Incorporate **generative models (e.g., GANs)** to fill in missing textures and improve realism.  
- Extend to **3D-aware simplification**, integrating structural line extraction and rooftop geometry.  
- Explore **deep learning-based end-to-end map generalization** to automatically learn simplification strategies.
