# KI-67 Cell Segmentation and Classification System

A GUI-based medical image analysis system that performs semantic segmentation on KI-67 histopathology images to classify positive and negative cell nuclei.

The system integrates a pre-trained deep learning segmentation model with post-processing techniques, providing an interactive interface for refining segmentation results.

---

## Project Overview

This project builds an end-to-end medical image segmentation system that allows users to:

1. Load KI-67 histopathology images  
2. Perform semantic segmentation using a trained deep learning model  
3. Classify nuclei into positive and negative cells  
4. Refine segmentation using watershed algorithm  
5. Visualize and interact with results via GUI  

---

## System Pipeline

- Input Image Folder  
- Image Loading Module  
- Semantic Segmentation (U-Net / ResNet backbone)  
- Nuclei Classification (Positive / Negative)  
- Post-processing (Watershed Algorithm)  
- Contour Visualization  
- Final Output Display  

---

## Models Used

### Segmentation Model (U-Net via segmentation_models.pytorch)
- Library: https://github.com/qubvel-org/segmentation_models.pytorch  
- Version: segmentation_models.pytorch==0.3.3  
- Architecture: U-Net  
- Backbone: ResNet50  
- Purpose: Perform pixel-wise segmentation of cell nuclei  

---

## Model Training and Integration

- The segmentation model was trained separately on KI-67 dataset  
- After training, the model weights were exported and integrated into the system  
- The GUI loads trained weights for inference only  
- This design separates:
  - Training pipeline  
  - Inference + visualization system  

---

## Post-processing Method

### Watershed Algorithm
- Purpose: Separate overlapping or clustered nuclei  
- Feature: Adjustable threshold via GUI slider  
- Benefit: Improves segmentation accuracy in dense cell regions  

---

## Integration Strategy

This project integrates deep learning and classical image processing:

- Deep Learning: Semantic segmentation using U-Net  
- Image Processing: Watershed algorithm for refinement  

A unified pipeline manages inference and post-processing steps.

---

## UI Framework

The graphical user interface is built using PyQt5.

UI features include:

- Source image folder selection  
- Destination folder selection  
- Model selection (U-Net)  
- Backbone selection (ResNet50)  
- Weight loading  
- Image navigation (Previous / Next)  
- Segmentation visualization  
- Contour display (positive / negative)  
- Watershed parameter adjustment (interactive slider)  

---

## System Architecture

- UI Layer: PyQt5 interface  
- Inference Layer: PyTorch model (segmentation_models.pytorch)  
- Processing Layer: Watershed + contour extraction  
- Image Layer: OpenCV / NumPy  

---

## Key Contributions

- Designed medical image segmentation pipeline for KI-67 analysis  
- Trained and deployed a deep learning segmentation model  
- Integrated segmentation model into an interactive GUI system  
- Combined deep learning with watershed-based refinement  
- Enabled real-time parameter tuning for segmentation improvement  

---

## Technical Challenges

- Handling overlapping and clustered nuclei in dense regions  
- Ensuring stable inference performance within GUI environment  
- Designing smooth interaction between model inference and UI updates  
- Integrating externally trained model into deployment pipeline  

---

## Requirements

### Core Environment
- Python == 3.8  

### Deep Learning Framework
- PyTorch == 2.1.0  
- torchvision == 0.16.0  
- torchaudio == 2.1.0  
- segmentation_models.pytorch == 0.3.3  

### UI Framework
- PyQt == 5.15.10  

### Image Processing Libraries
- NumPy  
- OpenCV  
- SciPy  

---

## Usage

### 1. Project Structure
```
DATA/
|-- GUI4ImageSeg
|   |-- HEDseg_ui.py
|   |-- HEDseg_contain.py
|   |-- HEDseg_start.py
|   |-- source folder
|   |   |-- src_1.png
|   |   |-- src_2.png
|   |-- model weight folder
|   |   |-- weight_1.pth
|   |   |-- weight_2.pth
```
---

### 2. Run the application
```
cd code
python HEDseg_start.py
```
---

### 3. Workflow

1. Select source image folder  
2. Select destination folder  
3. Load trained model weights  
4. Choose model and backbone  
5. Run segmentation  
6. Adjust watershed threshold using slider  
7. Visualize positive/negative contours  
8. Navigate images and export results  

![image](https://github.com/vGuNfnxXLJ/KI-67-Cell-Segmentation-and-Classification-System/blob/main/MedSeg_UI.PNG)
---

## External Acknowledgements

This project uses the following open-source library:

- segmentation_models.pytorch: https://github.com/qubvel-org/segmentation_models.pytorch  

All rights and credits belong to their respective authors.

---

## Notes

- This project is for research and educational purposes only  
- The model must be trained separately before use  
- Performance depends on training data and model quality  
