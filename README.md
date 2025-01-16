# Cat vs Dog Image Classification

## Overview

This project involves building and training a **Convolutional Neural Network (CNN)** from scratch to classify images of cats and dogs. The goal is to demonstrate advanced skills in data preprocessing, model architecture design, evaluation, and deployment through an interactive **Streamlit interface**. 

This project highlights key deep learning and deployment skills, offering insights into handling real-world machine learning challenges.

---

## Project Structure

The project is organized as follows:

```
Cat_Dog_Classification/
├── Dataset/
│   ├── Cat/
│   └── Dog/
├── app.py
├── model_training.py
├── README.md
└── requirements.txt
```

- **Dataset/**: Contains the images used for training and validation, divided into `Cat` and `Dog` subdirectories.
- **app.py**: Streamlit application for real-time image classification.
- **model_training.py**: Python script for training the CNN model.
- **README.md**: Detailed documentation for the project.
- **requirements.txt**: List of dependencies required to run the project.

---

## Key Features

- Data preprocessing pipeline, including **data cleaning** and **resizing**.
- Custom **CNN architecture** with:
  - **Batch Normalization**: To stabilize training.
  - **Dropout**: To prevent overfitting.
  - **L2 Regularization**: To improve generalization.
- Model evaluation using **precision**, **recall**, and **accuracy** metrics.
- Deployment of the trained model through a **Streamlit interface**, allowing real-time predictions on user-uploaded images.

---

## Dataset

The dataset contains approximately **20,000 images**, split evenly between cats and dogs. Each image is stored in the respective class folder:

- `Dataset/Cat/`: Contains images of cats.
- `Dataset/Dog/`: Contains images of dogs.

The dataset is sourced from the **[Microsoft Cats vs Dogs Dataset on Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data)**.

---

## Approach and Methodology

### 1. Preprocessing

1. **Data Cleaning**:
   - Removed corrupted images using **OpenCV** and **PIL**.
   - Ensured all images had valid extensions (`jpg`, `jpeg`, `png`).

2. **Image Resizing**:
   - Resized all images to **128x128 pixels** for uniformity and faster processing.

3. **Normalization**:
   - Scaled pixel values to the range [0, 1] for stable training.

4. **Dataset Splitting**:
   - Divided into:
     - 70% for training.
     - 20% for validation.
     - 10% for testing.

---

### 2. Model Architecture

The **CNN architecture** was designed with the following considerations:

- **Convolutional Layers**: Extract meaningful features from the images.
- **Pooling Layers**: Reduce dimensionality and computational cost.
- **Batch Normalization**: Accelerate training and stabilize convergence.
- **Dropout**: Mitigate overfitting.
- **L2 Regularization**: Penalize large weights for better generalization.

#### Architecture Overview

| Layer               | Type              | Filters/Units | Kernel Size | Stride | Activation | Regularization |
| -------------------- | ----------------- | ------------- | ----------- | ------ | ---------- | -------------- |
| Input               | Conv2D            | 16            | (3, 3)      | 1      | ReLU       | L2(0.001)      |
| Batch Normalization | -                 | -             | -           | -      | -          | -              |
| MaxPooling2D        | Pooling           | -             | (2, 2)      | -      | -          | -              |
| Dropout             | Regularization    | -             | -           | -      | -          | 0.3            |
| Conv2D              | Convolutional     | 32            | (3, 3)      | 1      | ReLU       | L2(0.001)      |
| Batch Normalization | -                 | -             | -           | -      | -          | -              |
| MaxPooling2D        | Pooling           | -             | (2, 2)      | -      | -          | -              |
| Dropout             | Regularization    | -             | -           | -      | -          | 0.4            |
| Conv2D              | Convolutional     | 64            | (3, 3)      | 1      | ReLU       | L2(0.001)      |
| Batch Normalization | -                 | -             | -           | -      | -          | -              |
| MaxPooling2D        | Pooling           | -             | (2, 2)      | -      | -          | -              |
| Dropout             | Regularization    | -             | -           | -      | -          | 0.5            |
| Flatten             | -                 | -             | -           | -      | -          | -              |
| Dense               | Fully Connected   | 128           | -           | -      | ReLU       | L2(0.001)      |
| Dropout             | Regularization    | -             | -           | -      | -          | 0.5            |
| Dense               | Output            | 1             | -           | -      | Sigmoid    | -              |

---

### 3. Training and Evaluation

- **Optimizer**: Adam for adaptive learning rate optimization.
- **Loss Function**: Binary Crossentropy to handle binary classification.
- **Metrics**: Accuracy, Precision, Recall.
- **Training**: Conducted over 20 epochs with early stopping to prevent overfitting.

---

### 4. Deployment

The trained model was deployed using **Streamlit**, enabling users to interact with the model by uploading images. Key steps include:

- Saving the trained model as an `.h5` file.
- Creating a user-friendly interface in `app.py`.
- Preprocessing uploaded images to match the model's input requirements.
- Displaying predictions in real-time.

---

## Results

- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~87%
- **Precision**: 0.81
- **Recall**: 0.79

---

## Skills Developed

This project demonstrates expertise in:

- **Data Cleaning**: Handling corrupted images and ensuring consistent preprocessing.
- **Deep Learning**: Designing and training a CNN with modern techniques (Batch Normalization, Dropout, L2 Regularization).
- **Evaluation**: Using advanced metrics (Precision, Recall, Accuracy) to validate model performance.
- **Deployment**: Deploying a machine learning model with Streamlit for real-world usability.

---

## Acknowledgments

Special thanks to **[Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data)** for providing the Microsoft Cats vs Dogs Dataset.
