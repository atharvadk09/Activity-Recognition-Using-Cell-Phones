# Activity Recognition Using Cell Phones
## 1. Project Overview
Human Activity Recognition (HAR) is the task of identifying a person's physical activity using sensor data collected from a device such as a smartphone.
This project uses the **UCI Human Activity Recognition Using Smartphones Dataset** to classify six human activities using machine learning algorithms.
The project compares three supervised machine learning models:
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
The models are evaluated using accuracy, precision, recall, F1-score, training time, and prediction time.
Based on the experimental results, **SVM achieved the best overall performance with 96.20% test accuracy** and was selected as the final model.
---
## 2. Problem Statement
Smartphones contain accelerometer and gyroscope sensors that continuously capture information about human movement.
The problem addressed in this project is:
> **To develop a machine learning system that can recognize and classify human physical activities from smartphone sensor data.**
The system classifies sensor observations into six activities:
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying
The project also compares different machine learning algorithms to determine which model performs best for activity recognition.
---
## 3. Objectives
The main objectives of this project are:
- To understand smartphone-based human activity recognition.
- To use the UCI HAR dataset for supervised classification.
- To preprocess and load the sensor feature data.
- To train multiple machine learning classification models.
- To compare SVM, KNN, and Random Forest.
- To evaluate models using standard classification metrics.
- To perform SVM hyperparameter tuning using GridSearchCV.
- To visualize model performance and the confusion matrix.
- To select the best-performing model for final activity recognition.
---
## 4. Dataset
### UCI Human Activity Recognition Using Smartphones
The project uses the **Human Activity Recognition Using Smartphones Dataset**.
The dataset was collected from **30 volunteers** performing six different activities while carrying a smartphone.
The smartphone's:
- Accelerometer
- Gyroscop
were used to capture movement-related signals.
The sensor signals were sampled at **50 Hz**.
The dataset contains **561 extracted features** for each observation.
### Activity Classes
| ID | Activity |
|---:|---|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

### Dataset Split
The dataset already provides separate training and testing sets.
| Dataset | Samples | Features |
|---|---:|---:|
| Training | 7,352 | 561 |
| Testing | 2,947 | 561 |
The raw dataset is stored locally in the `data/` directory and is excluded from GitHub using `.gitignore`.
---
## 5. Project Workflow
The overall workflow of the project is:
```text
UCI HAR Dataset
       |
       v
Load Sensor Features
       |
       v
Preprocess / Organize Data
       |
       v
Training Dataset
       |
       +------------------+
       |                  |
       v                  v
      SVM                KNN
       |                  |
       +--------+---------+
                |
                v
        Random Forest
                |
                v
       Model Evaluation
                |
                v
   Accuracy / Precision /
   Recall / F1-score /
   Training & Prediction Time
                |
                v
        Model Comparison
                |
                v
        Best Model: SVM
```
--
## 6. Methodology
### Step 1: Data Loading
The `load_data.py` file reads:
- `features.txt`
- `X_train.txt`
- `X_test.txt`
- `y_train.txt`
- `y_test.txt`
- `activity_labels.txt`

The feature names are assigned to the training and testing data.
The activity IDs are used as the target labels.
### Step 2: Model Training
Three supervised machine learning algorithms are trained:
1. SVM
2. KNN
3. Random Forest

### Step 3: Prediction
Each trained model predicts activity labels for the test dataset
### Step 4: Evaluation
The predictions are compared with the actual test labels.
The following metrics are calculated:
- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Prediction time

### Step 5: Model Comparison
The performance of all three algorithms is stored in:
```text
results/model_comparison_results.csv
```
The results are also visualized using:
```text
results/model_accuracy_comparison.png
```
### Step 6: Hyperparameter Tuning
GridSearchCV is used to search for better SVM hyperparameters.
The parameters explored were:

- `C`: 1, 10, 100
- `gamma`: scale, 0.01, 0.001
- `kernel`: RBF

Five-fold cross-validation was used.
The best tuning result was:
```text
C = 10
gamma = 0.01
kernel = rbf
Cross-validation accuracy = 94.61%
Test accuracy = 96.20%
```

The tuning experiment did not improve the final test accuracy compared with the original SVM configuration, so the original SVM configuration remains the selected final model.
---

## 7. Machine Learning Models
### 7.1 Support Vector Machine (SVM)
SVM is a supervised machine learning algorithm that finds a decision boundary that separates different classes.
For this project, an RBF kernel is used.
Configuration:
```text
Kernel = RBF
C = 10
Gamma = scale
```
SVM achieved the highest accuracy among the three tested models.
---

### 7.2 K-Nearest Neighbors (KNN)
KNN classifies a new observation based on the classes of its nearest training observations.
Configuration:
```text
Number of neighbors = 5
```
KNN is simple to understand, but prediction is slower because distances need to be calculated against training samples
---

### 7.3 Random Forest
Random Forest is an ensemble learning algorithm that combines multiple decision trees.
Configuration:
```text
Number of estimators = 100
Random state = 42
```
Random Forest provided good performance and relatively fast prediction.
---
## 8. Model Comparison
The experimental results are:

| Model | Accuracy | Precision | Recall | F1 Score | Training Time (s) | Prediction Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 96.20% | 96.31% | 96.20% | 96.19% | 0.91 | 0.80 |
| KNN | 90.16% | 90.59% | 90.16% | 90.07% | 0.04 | 1.85 |
| Random Forest | 92.57% | 92.70% | 92.57% | 92.55% | 1.40 | 0.05 |

### Best Model
SVM achieved the highest performance:
- **Accuracy:** 96.20%
- **Precision:** 96.31%
- **Recall:** 96.20%
- **F1-score:** 96.19%
Therefore, **SVM was selected as the final activity recognition model.**

---
## 9. Activity-wise SVM Performance
The classification report for the final SVM shows the following performance:

| Activity | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Walking | 96% | 98% | 97% |
| Walking Upstairs | 94% | 96% | 95% |
| Walking Downstairs | 99% | 94% | 96% |
| Sitting | 98% | 90% | 94% |
| Standing | 91% | 98% | 95% |
| Laying | 100% | 100% | 100% |
The model performs particularly well for **Laying**.
The relatively lower recall for **Sitting** indicates that some sitting observations are confused with other activities, especially activities with similar stationary sensor patterns.
---

## 10. Hyperparameter Tuning
SVM hyperparameters were optimized using `GridSearchCV`.

### Search Space
```text
C:
1
10
100
Gamma:
scale
0.01
0.001
Kernel:
rbf
```
A total of 9 parameter combinations were evaluated using **5-fold cross-validation**.

### Best Parameters
```text
C = 10
Gamma = 0.01
Kernel = RBF
```

### Results
```text
Best Cross-Validation Accuracy = 94.61%
Test Accuracy = 96.20%
```
The tuning experiment confirms that the SVM configuration performs strongly on the dataset. However, because the tuned configuration did not increase the final test accuracy, the original SVM configuration is retained as the final model.

---
## 11. Evaluation Metrics

### Accuracy
Accuracy measures the proportion of correctly classified samples.
```text
Accuracy = Correct Predictions / Total Predictions
```
### Precision
Precision measures how many samples predicted as a particular activity actually belong to that activity.
```text
Precision = True Positives / (True Positives + False Positives)
```
### Recall
Recall measures how many samples belonging to an activity were correctly detected.
```text
Recall = True Positives / (True Positives + False Negatives)
```
### F1-score
F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
These metrics provide a more complete evaluation than accuracy alone.
---
## 12. Confusion Matrix
A confusion matrix is used to understand how the model classifies each activity.
It shows:
- Correct predictions on the diagonal.
- Incorrect predictions outside the diagonal.
- Which activities are commonly confused with each other.
The generated SVM confusion matrix is stored in:
```text
results/svm_confusion_matrix.png
```
The final confusion matrix is also generated by:
```text
src/final_results.py
```
and stored in:
```text
results/final_confusion_matrix.png
```
---
## 13. Project Visualizations
The project generates visual outputs to help analyze model performance.
### Model Accuracy Comparison
```text
results/model_accuracy_comparison.png
```
This visualization compares the accuracy of SVM, KNN, and Random Forest.
### SVM Confusion Matrix
```text
results/svm_confusion_matrix.png
```
This visualization shows the classification performance of SVM for each activity.
### Final Confusion Matrix
```text
results/final_confusion_matrix.png
```
This provides the confusion matrix for the selected final SVM model.
---
## 14. Project Structure

```text
Activity-Recognition-Using-Cell-Phones/
│
├── data/
│   └── UCI HAR Dataset/
│       └── UCI HAR Dataset/
│           ├── train/
│           ├── test/
│           ├── activity_labels.txt
│           ├── features.txt
│           ├── features_info.txt
│           └── README.txt
│
├── results/
│   ├── model_comparison_results.csv
│   ├── model_accuracy_comparison.png
│   ├── svm_confusion_matrix.png
│   └── final_confusion_matrix.png
│
├── src/
│   ├── load_data.py
│   ├── eda.py
│   ├── train_svm.py
│   ├── train_knn.py
│   ├── train_random_forest.py
│   ├── model_comparison.py
│   ├── visualize_results.py
│   ├── confusion_matrix.py
│   ├── final_results.py
│   └── tune_svm.py
│
├── .gitignore
├── README.md
└── requirements.txt
```
> Note: The actual raw dataset is intentionally not uploaded to GitHub. The `data/` directory is ignored using `.gitignore`.

---
## 15. Technologies Used
### Programming Language
- Python 3
### Libraries
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
### Machine Learning Algorithms
- Support Vector Machine
- K-Nearest Neighbors
- Random Forest
### Development Tools
- Visual Studio Code
- Git
- GitHub
- Python Virtual Environment

---
## 16. Installation and Setup
### 1. Clone the Repository
```bash
git clone https://github.com/atharvadk09/Activity-Recognition-Using-Cell-Phones.git
cd Activity-Recognition-Using-Cell-Phones
```
### 2. Create a Virtual Environment
Windows:
```powershell
python -m venv .venv
```
Activate it:
```powershell
.venv\Scripts\Activate.ps1
```
### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```
If `requirements.txt` is not available, install:
```powershell
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```
### 4. Add the Dataset
Download the UCI HAR Dataset and place it inside:
```text
data/UCI HAR Dataset/UCI HAR Dataset/
```
The folder should contain:
```text
train/
test/
activity_labels.txt
features.txt
features_info.txt
README.txt
```
---
## 17. How to Run the Project
Run the data loader:
```powershell
python src/load_data.py
```
Run SVM:
```powershell
python src/train_svm.py
```
Run KNN:
```powershell
python src/train_knn.py
```
Run Random Forest:
```powershell
python src/train_random_forest.py
```
Run model comparison:
```powershell
python src/model_comparison.py
```
Generate accuracy visualization:
```powershell
python src/visualize_results.py
```
Generate confusion matrix:
```powershell
python src/confusion_matrix.py
```
Run SVM hyperparameter tuning:
```powershell
python src/tune_svm.py
```
Run the final model evaluation:
```powershell
python src/final_results.py
```
---

## 18. Results and Discussion
The experiments show that SVM performs better than KNN and Random Forest for this dataset.
SVM achieved an accuracy of **96.20%**, while Random Forest achieved **92.57%** and KNN achieved **90.16%**.
The high SVM performance indicates that the extracted smartphone sensor features provide enough information to distinguish the six activities effectively.
Some activities are easier to distinguish than others.
For example, Laying achieved 100% precision, recall, and F1-score in the reported classification results. Sitting has comparatively lower recall, which suggests that some stationary activity observations can be difficult to distinguish.
The comparison also shows that training time and prediction time are important when selecting a model. KNN has a very low training time but a comparatively high prediction time, whereas Random Forest has a higher training time but very fast prediction.

---
## 19. Limitations
The current project has several limitations:
1. The system is evaluated on the UCI HAR benchmark dataset rather than live smartphone sensor data.
2. The project uses pre-extracted features provided by the dataset.
3. The current implementation does not perform real-time activity recognition on a smartphone.
4. The model is trained on a fixed set of six activities.
5. Performance may vary for people or sensor conditions that are different from those represented in the dataset.
6. The hyperparameter search was limited to a small parameter grid to keep experimentation practical.

---
## 20. Future Scope
The project can be extended in several ways:

- Implement real-time activity recognition using smartphone sensors.
- Develop a mobile application for activity prediction.
- Add more physical activities.
- Test the model on data collected from different users.
- Compare additional algorithms such as Logistic Regression, XGBoost, or neural networks.
- Perform more extensive hyperparameter optimization.
- Explore dimensionality reduction techniques such as PCA.
- Use deep learning models such as CNNs or LSTMs on raw sensor signals.
- Deploy the trained model as an API or mobile inference system.

---
## 21. Conclusion
This project demonstrates the use of machine learning for human activity recognition using smartphone sensor data.
Three classification algorithms were implemented and compared:
- SVM
- KNN
- Random Forest

Among the tested models, **SVM achieved the best overall performance with 96.20% accuracy, 96.31% precision, 96.20% recall, and 96.19% F1-score**.
Therefore, SVM was selected as the final model for the activity recognition task.
The project also demonstrates the importance of comparing multiple machine learning algorithms, evaluating them using multiple metrics, and tuning model hyperparameters before selecting a final model.
---
## 22. Reproducibility
The project includes separate scripts for data loading, model training, comparison, visualization, confusion matrix generation, and hyperparameter tuning.
The dataset is not committed to GitHub because of its size and is expected to be placed locally in the `data/` directory
The `.gitignore` file excludes:
```text
data/
.venv/
__pycache__/
*.pyc
```
This keeps the repository focused on source code, documentation, and generated results.
---
## 23. Authors
This project was developed as a two-member machine learning project.
- Repository Owner: `atharvadk09`
- Project Contributor: `Anusha`
---
## 24. Repository
GitHub Repository:
https://github.com/atharvadk09/Activity-Recognition-Using-Cell-Phones

