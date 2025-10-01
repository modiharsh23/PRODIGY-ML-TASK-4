# ✍️ Hand Gesture Recognition using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) to recognize and classify 10 different hand gestures from the "Leap Motion Gesture Recognition" dataset.

This notebook demonstrates an end-to-end deep learning workflow using TensorFlow and Keras, including data preprocessing, model architecture design, training, and evaluation.

---

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [How to Run](#-how-to-run)

---

## 🖼️ Dataset

This project uses the **"Hand Gesture Recognition Database (leapGestRecog)"**. You can download it from [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).

The dataset is organized into folders by subject (person) and then by gesture type. It contains images for 10 distinct gestures:

1.  Palm (Horizontal)
2.  "L"
3.  Fist (Horizontal)
4.  Fist (Vertical)
5.  Thumb
6.  Index
7.  "OK"
8.  Palm (Vertical)
9.  "C"
10. Down

The required directory structure should be:


```
.
├── Task-4.ipynb
└── DATA/
    └── leapGestRecog/
    ├── 00/
    │   ├── 01_palm/
    │   ├── 02_l/
    │   └── ...
    ├── 01/
    └── ...
```
---

## ⚙️ Methodology

The project follows a standard deep learning pipeline:

1.  **Image Preprocessing**:
    * Images are loaded from their respective directories.
    * Each image is converted to **grayscale**.
    * All images are resized to a uniform dimension of 128x128 pixels.
    * Pixel values are normalized to a range of [0, 1].

2.  **Data Preparation**:
    * Labels are extracted from folder names and adjusted to be zero-indexed (0-9).
    * Labels are one-hot encoded to be compatible with the categorical crossentropy loss function.
    * The dataset is split into an 80% training set and a 20% testing set.

3.  **CNN Model Architecture**:
    * A sequential model is built with three convolutional blocks.
    * Each block consists of a `Conv2D` layer followed by a `MaxPooling2D` layer. This helps the network learn hierarchical features from simple edges to more complex shapes.
    * The output from the convolutional blocks is flattened.
    * A `Dense` layer with 512 units and a `Dropout` layer (to prevent overfitting) are used for classification.
    * The final `Dense` output layer has 10 units (one for each gesture) and a `softmax` activation function.

4.  **Training & Evaluation**:
    * The model is compiled with the `adam` optimizer and `categorical_crossentropy` loss function.
    * It is trained for 10 epochs.
    * Performance is evaluated on the unseen test set, and training/validation history is plotted.

---

## 📈 Results

The trained CNN model performs exceptionally well on this dataset, achieving **99.8% accuracy** on the test set. The training and validation accuracy curves show that the model learns quickly and effectively without significant overfitting.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/modiharsh23/PRODIGY-ML-TASK-4.git](https://github.com/modiharsh23/PRODIGY-ML-TASK-4.git)
    cd YOUR-REPOSITORY-NAME
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install tensorflow opencv-python scikit-learn matplotlib
    ```

4.  **Download the data:**
    * Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
    * Unzip the files and place the `leapGestRecog` folder inside a `DATA` directory as shown in the [Dataset section](#-dataset).

5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter lab Task-4.ipynb
    ```
    Execute the cells in order to preprocess the data, train the model, and see the evaluation results.
