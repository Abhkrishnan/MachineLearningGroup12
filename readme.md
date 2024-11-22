# 1. GROUP_12

# 2. Heart disease Prediction & ECG image classification of Cardiac Patients

# 3. Group Members
    1. Abhay Krishnan
    2. Aghil
    3. Akul
    4. Edwin
    5 .Hilal

# 4. Objective
* Predicting heart disease using machine learning techniques.
    - The Code explores various classifiers like Logistic Regression, Decision Trees,Random Forest to analyze heart disease factors, such as cholesterol levels, chest pain types, and blood pressure. 
    - The notebook aims to build predictive models using structured health data.

* ECG image classification
    - ECG (Electrocardiogram) images contain distinct patterns that reflect various heart conditions. 
    *By applying deep learning techniques, such as convolutional neural networks (CNNs), these visual patterns can be automatically analyzed to classify different types of heart abnormalities, such as arrhythmias, ischemia, or myocardial infarction. 

# 5. Project Milestone
Data Loading and Inspection -> Data Preprocessing -> Data Splitting: -> Model Development -> Accuracy Improvement: -> Clustering

## 6. a. Source
* The source of Dataset 1 and Dataset 2 is from [Kaggle](https://www.kaggle.com/)

## b. License
### Dataset 1 - Predicting Heart Disese
* Dataset Link https://www.kaggle.com/datasets/mexwell/heart-disease-dataset
* License Link  https://creativecommons.org/licenses/by/4.0/

### Dataset 2 - ECG Image Classification
* Dataset Link https://www.kaggle.com/datasets/evilspirit05/ecg-analysis
* License Link https://www.mit.edu/~amini/LICENSE.md

## c. Preview of Dataset

### Dataset 1
* This image shows the overview of the tabular data, which showcases the attributes that we are using for the model
    - Age
    - Sex
    - Chest Pain Type
    - Resting Blood Pressure 
    - Serum Cholesterol 
    - Fasting Blood Sugar
    - Resting Electrocardiogram Results 
    - Maximum Heart Rate Achieved 
    - Exercise Induced Angina
    - Oldpeak (ST Depression)
    - The Slope of Peak Exercise ST Segment
    - Class (Target)​
![Dataset1Description](https://github.com/user-attachments/assets/1e119828-043f-4f21-af0e-f475a31b4550)
### Dataset 2 

*This image represents the ECG of a MI patient

![MI - IMAGE](https://github.com/user-attachments/assets/f8f64f06-6716-400a-9fcc-bead43c8da47)

# 7. Data Prepration Pipeline
## Dataset 1
*   The data is loaded using pandas library, and it is proceeded  with initial inspection and visualization  (using seaborn)
*   Then the data is preprocessed to check the missing values 
*   The data is checked for outliers using box plot
*   The data is feature scaled using Standard scalers to handle the outliers
*   The data is then split into training set and testing set

## Dataset 2
*   All the required libraries ie, pandas, os, tensorflow are imported
*   data is being imported and classified accordingly using the os library
*   the data is then fed to a ImageDataGenerator to augment the data and artificially creating new datas from the  existing samples

## Dataset 2

# 8
## Dataset 1
[R2] https://github.com/Abhkrishnan/MachineLearningGroup12/blob/c6e03fa86bc937a157dcb2546afb69643cec761f/PorjectOne/code.ipynb#L41-L937
*   Essential libraries required to run the model such as  pandas, numpy, matplotlib, seabord and sklearn are imported
*   The data is loaded using pandas library, and it is proceeded with initial inspection by describing the dataset and visualization of correlation (using seaborn)
*   Then the data is preprocessed to check the missing values 
*   The data is check for outliers using box plot
*   The data is feature scaled using Standard scalers as there are outliers found
*   The data is then split into training set and testing set

*   From sklearn library Standard Scalar is import to Normalise the data


*   Then the data is passed through a decision tree estimation model to find out the suitable depth and sample split using the GridSearcCV method
*   Various Machine Learning ie, DecisionTree, RandomForest, Logistic Regression, KMeans, KNN, GradientBoosting, NaiveBayes, SVM that are required to train the dataset is also imported
*   The data is then fed into a loop of model when it predicts the accuracy
*   To improve the accuracy added a Kfold Cross Validation with split= 5 and ran the loop with it
[R3] https://github.com/Abhkrishnan/MachineLearningGroup12/blob/c6e03fa86bc937a157dcb2546afb69643cec761f/PorjectOne/code.ipynb#L938-L1151
*   KMeans algorithm is ran though the data, by finding the suitable k number using the Elbow method

[R5]
Normal Loop https://github.com/Abhkrishnan/MachineLearningGroup12/blob/c6e03fa86bc937a157dcb2546afb69643cec761f/PorjectOne/code.ipynb#L637-L825
![Normal Loop](https://github.com/user-attachments/assets/ca05926c-2648-4077-bff7-aad5075f3ffc)
K Fold Loop https://github.com/Abhkrishnan/MachineLearningGroup12/blob/c6e03fa86bc937a157dcb2546afb69643cec761f/PorjectOne/code.ipynb#L892-L929
![K Fold Loop](https://github.com/user-attachments/assets/89021041-b207-4399-a079-8ec062d75790)
K Means https://github.com/Abhkrishnan/MachineLearningGroup12/blob/c6e03fa86bc937a157dcb2546afb69643cec761f/PorjectOne/code.ipynb#L1079-L1151
![K Means](https://github.com/user-attachments/assets/446b7a5e-e73c-4d52-899e-a51579ad5110)
## Dataset 2
[R2]
*   data is being imported and classified accordingly using the os library
*   the data is then fed to a ImageDataGenerator to augment the data and artificatilly create new datas from the already existing ones
[R4]
*   A deep learning model is created and compiled for training
*   the data is fitted into the model with an epochs of 25 prediting its accuracy score

# 8 a. Short Description of the requirement
## Dataset 1 
*   The model aims to predict heart disease of a patients with relevant attributes using different machine learning algorith Such as: DecisionTree, RandomForest, Logistic Regression, KMeans, KNN, GradientBoosting, NaiveBayes, SVM and compare the accuracy between the model for better prediction.

## Dataset 2 
*   The model aims to classify ECG of different Cardiac condition patients, such as Myocardial Infarction Patients​, Patients with Abnormal Heartbeat​, History of Myocardial Infarction​,Normal Person ECG Images​ USING Neural Network

# b. Model (input)
## Dataset 1 
*   The model is trying to predict Heart Disease of a patient using multiple given attributes

## Dataset 2
*   The model is trying to classify distinct patterns that reflect various heart conditions in a patient 

# c. Output
## Dataset 1 
* The best model with the best accuracy is Random Forest with a Accuracy of 0.93
![Output](https://github.com/user-attachments/assets/36607227-c1ae-45d4-b931-62b56bcc7a9)
* After performing K - Fold algorithm on the model there was a significant improvement in the accuracy, further validating the model
![Output](https://github.com/user-attachments/assets/5836ff1a-7a2a-414a-af2a-b2db7893c232)

## Dataset 2
* The MLP Model ran with a accuracy of 0.45
* The CNN Model ran with a accuracy of 0.50


# 9. Files and Folders
*   The repository contains 2 folder ('Dataset 1 HeartDiseaseClassification' and 'Dataset 2 ECGImageClassification') and a readme file
*   'Dataset 1 HeartDiseaseClassification' is the Heart Disease Prediction and contains the 'HeartDiseaseClassification.ipynb' which contains the code for the analysis, 'heart_statlog_cleveland_hungary_final.csv' contains the data for the analysis
*   'Dataset 2 ECGImageClassification' is the ECG Image classificaiton and contains the 'ECGImageClassification.ipynb' which contains the code for the analysis and ECG_DATA folder containing the dataset used for the analysis
