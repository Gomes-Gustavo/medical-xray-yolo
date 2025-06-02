# YOLOv8 for Medical Finding Detection in Chest X-Rays

This repository contains a **medical image analysis project** that adapts and evaluates the **YOLOv8 object detection model** to identify **14 common thoracic findings** in chest X-rays from the **VinBigData dataset**. The project includes data processing, model adaptation, training on a subset, and clinical validation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Installation](#installation)
- [Model Performance](#model-performance)
- [Dataset Subset and Impact](#dataset-subset-and-impact)
- [References](#references)
- [Author](#author)

## Project Overview

This project builds an end-to-end pipeline to detect thoracic abnormalities in chest X-rays using YOLOv8. It focuses on adapting a general-purpose object detector for medical imaging, emphasizing practical implementation and evaluation with a zero-cost approach.

## Dataset

The **VinBigData Chest X-ray Abnormalities Detection dataset** is used. It contains **18,000 chest X-ray images** with annotations for **14 common thoracic findings** (plus a "No finding" class).

-   **Source:** [VinBigData Chest X-ray Abnormalities Detection on Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)
-   **Project Usage:** A **stratified subset of 500 images** was used for training and initial validation due to computational limitations (zero-cost GPU). 
## Project Structure

```
medical-xray-yolo/
├── .configs
│   ├── medical_augmentation.yaml
│   ├── medical_hyper.yaml
│   ├── medical_xray_model.yaml
│   └── train_config.yaml
├── data
│   ├── processed_yolo_dataset
│   │   ├── test
│   │   │   ├── images
│   │   │   └── labels
│   │   ├── train
│   │   │   ├── images
│   │   │   └── labels
│   │   └── val
│   │       ├── images
│   │       └── labels
│   ├── data.yaml
│   ├── raw
│   │   ├── test_images
│   │   └── train_images
│   └── train.csv
├── models
│   └── medical_xray_best_model.pt
├── notebooks
│   ├── 01_medical_data_processing.ipynb
│   ├── 02_model_adaptation.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_clinical_validation.ipynb
└── results
    ├── clinical_cases_examples.png
    ├── clinical_performance_analysis.png
    ├── confidence_analysis.png
    ├── confusion_matrix.png  
    └── roc_curves.png
├── venv
├── .gitignore
├── LICENSE
└── requirements.txt   
```

## Jupyter Notebooks

| Step                             | Notebook                                                                 | Description                                                                                                   |
| :------------------------------- | :----------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| **1. Data Processing** | [01_medical_data_processing.ipynb](notebooks/01_medical_data_processing.ipynb) | Loads, explores, preprocesses data (DICOM to PNG, YOLO format), creates subset & `data.yaml`.                |
| **2. Model Adaptation** | [02_model_adaptation.ipynb](notebooks/02_model_adaptation.ipynb)        | Configures YOLOv8, class weights, augmentation, and training parameters for the medical task.             |
| **3. Model Training** | [03_model_training.ipynb](notebooks/03_model_training.ipynb)           | Trains the YOLOv8 model (typically on Google Colab), uses metrics such as mAP. Saves `medical_xray_best_model`.                                       |
| **4. Clinical Validation** | [04_clinical_validation.ipynb](notebooks/04_clinical_validation.ipynb)      | Evaluates model on test set using Precision/Recall, ROC AUC analysis, etc.                             |

## Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Gomes-Gustavo/medical-xray-yolo.git
cd medical-xray-yolo
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```
## Model Performance
---

**A Note on Model Performance:**

The performance metrics detailed below reflect the model's capabilities when trained on a limited, albeit strategically sampled, subset of 500 images from the VinBigData dataset. This constraint was a deliberate choice to align with the project's zero-cost objective, utilizing free-tier GPU resources for training.

While this approach facilitates a complete demonstration of the development pipeline—from data preprocessing and model adaptation to rigorous clinical validation—it is important to recognize that the scale of the training data inherently influences the quantitative outcomes. Specifically, metrics such as recall (sensitivity) and the model's overall generalization capacity are typically correlated with dataset size.

Consequently, the presented results should be interpreted as a baseline performance achieved under these resource constraints. The primary focus of this project iteration is to showcase a robust methodology, the successful adaptation of the YOLOv8 architecture to the complex domain of medical image analysis, and a comprehensive evaluation framework, rather than to achieve peak performance metrics that would necessitate more extensive computational resources for training on the full dataset.

---

Key performance indicators are derived from `Notebook 04` using the 500-image training subset:

| Metric                       | Value          | Notes                                             |
| :--------------------------- | :------------- | :------------------------------------------------ |
| **Average ROC AUC** | **0.679** | Across 14 findings, moderate discrimination.  |
| **Precision (weighted avg.)**| ~52.7%         | For positive detections.                          |
| **Recall (weighted avg.)** | ~21.9%         | For positive detections.                          |

**Performance Highlights (AUC):**
* **Strongest:** Cardiomegaly (0.844), Pleural effusion (0.830), Aortic enlargement (0.812).
* **Most Challenging:** Consolidation (0.591), Atelectasis (0.584), Calcification (0.527).

- More metrics and analysis in [Notebook 04](notebooks/04_clinical_validation.ipynb) 

## Dataset Subset and Impact

This project was developed using a **stratified subset of 500 images** from the VinBigData dataset for training and initial validation. 

While this subset allows for rapid prototyping and demonstration of the end-to-end pipeline:
-   The model's performance, particularly **recall (sensitivity)**, is significantly impacted by the limited data.
-   The generalization capability to unseen data from the full dataset might be constrained.

Training on a larger portion of the VinBigData dataset is the most critical next step for substantial performance improvements.

## References

-   **Dataset:** [VinBigData Chest X-ray Abnormalities Detection on Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)
-   **YOLOv8:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Author

Developed by Gustavo Gomes

- [LinkedIn](https://www.linkedin.com/in/gustavo-gomes-581975333/)
