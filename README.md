# Face Mask Detection using CNN

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies images as **with mask** or **without mask**. Trained on the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) from Kaggle.

## Overview

This project loads a labeled dataset of face images, preprocesses them into fixed-size RGB arrays, and trains a CNN classifier to distinguish between masked and unmasked faces. It also includes a simple predictive system for classifying a new, single input image.

## Dataset

- **Source:** [omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) on Kaggle
- **Classes:**
  - `with_mask` — 3,725 images (label `1`)
  - `without_mask` — 3,828 images (label `0`)
- Images are resized to **128x128** and converted to RGB before being fed into the model.

## Model Architecture

A sequential CNN with the following structure:

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3x3 kernel, ReLU |
| MaxPooling2D | 2x2 pool size |
| Conv2D | 64 filters, 3x3 kernel, ReLU |
| MaxPooling2D | 2x2 pool size |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense | 64 units, ReLU |
| Dense | 2 units (output layer) |

**Compilation settings:**
- Optimizer: `adam`
- Loss: `sparse_categorical_crossentropy`
- Metric: `accuracy`

## Results

Trained for 5 epochs with a 90/10 train/validation split:

| Metric | Value |
|---|---|
| Test Accuracy | **~92.2%** |
| Test Loss | ~0.21 |

Training and validation loss/accuracy curves are plotted in the notebook to visualize learning progress.

## Project Structure

```
Face-Mask-Detection-using-CNN/
├── Face_Mask_Detection_using_CNN.ipynb   # Main notebook (data pipeline, model, training, inference)
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.x
- A [Kaggle API token](https://www.kaggle.com/docs/api) (`kaggle.json`) to download the dataset

### Installation

```bash
pip install kaggle numpy matplotlib opencv-python pillow scikit-learn tensorflow
```

### Usage

1. Place your `kaggle.json` API credentials in the working directory.
2. Run the notebook (locally in Jupyter, or in [Google Colab](https://colab.research.google.com/github/ektakum13/Face-mask-detection-using-CNN/blob/main/Face_Mask_Detection_using_CNN.ipynb)):
   - Downloads and extracts the dataset from Kaggle
   - Loads and labels images (`with_mask` = 1, `without_mask` = 0)
   - Resizes images to 128x128 and converts them to numpy arrays
   - Splits data into training (80%) and test (20%) sets, then scales pixel values to [0, 1]
   - Builds, compiles, and trains the CNN
   - Evaluates the model on the test set and plots training curves
3. Run the **Predictive System** cell at the end of the notebook and provide a path to an image — the model will predict whether the person is wearing a mask.

## Tech Stack

- **TensorFlow / Keras** — model building and training
- **OpenCV & Pillow** — image loading and preprocessing
- **NumPy** — array operations
- **Matplotlib** — visualization
- **scikit-learn** — train/test splitting

## Future Improvements

- Add data augmentation to improve generalization
- Train for more epochs with early stopping / learning rate scheduling
- Integrate with a face detector (e.g., Haar Cascade or MTCNN) for real-time webcam mask detection
- Evaluate with a confusion matrix, precision, and recall in addition to accuracy

## License

This project is open source and available for educational use.
