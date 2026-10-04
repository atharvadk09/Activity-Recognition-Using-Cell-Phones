# Activity-Recognition-Using-Cell-Phones
# Activity Recognition Using Cell Phones

## 1. Project Overview

Human Activity Recognition (HAR) is the task of identifying a person's physical activity using sensor data.

This project uses smartphone accelerometer and gyroscope sensor data to classify six different human activities using machine learning algorithms.

The project compares three classification algorithms:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

The model with the best performance is selected as the final activity recognition model.

---

## 2. Problem Statement

To develop a machine learning-based system that can recognize human physical activities from smartphone sensor data and compare different classification algorithms based on their performance.

The system classifies sensor data into six activities:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

---

## 3. Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset contains sensor measurements collected from smartphones carried by participants while performing different physical activities.

### Dataset characteristics

- 30 volunteers
- 6 activity classes
- Smartphone accelerometer and gyroscope sensors
- 50 Hz sampling frequency
- 561 extracted features
- Separate training and testing datasets

The dataset is kept locally and is **not uploaded to GitHub** because of its size.

---

## 4. Methodology

The overall workflow is:

```text
UCI HAR Dataset
       |
       v
Data Loading
       |
       v
Exploratory Data Analysis
       |
       v
Train Machine Learning Models
       |
       +-------------------+
       |         |         |
       v         v         v
      SVM       KNN    Random Forest
       |         |         |
       +---------+---------+
                 |
                 v
        Model Comparison
                 |
                 v
          Best Model: SVM
                 |
                 v
       Confusion Matrix