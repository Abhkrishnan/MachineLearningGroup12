# 1. GROUP_12

# 2. Heart disease Prediction & ECG image classification of Cardiac Patients

# 3. Group Members
Abhay Krishnan ,Aghil, Akul, Edwin, Hilal

# 4. Objective
* Predicting heart disease using machine learning techniques.
    *It explores various classifiers like Logistic Regression, Decision Trees,Random Forest to analyze heart disease factors, such as cholesterol levels, chest pain types, and blood pressure. 
    *The notebook aims to build predictive models using structured health data.

* ECG image classification
    *ECG (Electrocardiogram) images contain distinct patterns that reflect various heart conditions. 
    *By applying deep learning techniques, such as convolutional neural networks (CNNs), these visual patterns can be automatically analyzed to classify different types of heart abnormalities, such as arrhythmias, ischemia, or myocardial infarction. 

# 5. Project Milestone
*   Week 4 -Project Pitch
*   Week 5 - 11 Project workflow and Execution
*   Week 12 - Project Submission

## 6. a. Source
* The source of Dataset 1 and Dataset 2 is from [Kaggle](https://www.kaggle.com/)

## b. Lisence
### Dataset 1 - Predicting Heart Disese
* Link of the Dataset https://www.kaggle.com/datasets/mexwell/heart-disease-dataset
* Link of Lisense https://creativecommons.org/licenses/by/4.0/

### Dataset 2 - ECG Image Classification
* Link of the dataset https://www.kaggle.com/datasets/evilspirit05/ecg-analysis
* Link of Lisense https://www.mit.edu/~amini/LICENSE.md

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
*   the data is loaded using pandas library, and it is procceded with initial inspection and visualisation (using seaborn)
*   Then the data is preprocesed to check the missing values 
*   The data is check for outliers using box plot
*   The data is feature scaled using Standard scalers as there are outliers found
*   The data is then split into training set and testing set

## Dataset 2
*   All the required libraries ie, pandas, os, tensorflow are imported
*   data is being imported and classified accordingly using the os library
*   the data is then fed to a ImageDataGenerator to augment the data and artificatilly create new datas from the already existing ones

## Dataset 2

# 8 a. Short Description of the requirement
## Dataset 1 
*   The model aims to predict heart disease of a patients with relevent attributes using different machine learning algorith Such as: DecisionTree, RandomForest, Logistic Regression, KMeans, KNN, GradientBoosting, NaiveBayes, SVM and compare the accuracy between the model for better prediction.

## Dataset 2 
*   The model aims to classify ECG of different Cardiac condition patients, such as Myocardial Infarction Patients​, Patients with Abnormal Heartbeat​, History of Myocardial Infarction​,Normal Person ECG Images​ USING Neural Network

# b. Model (input)
## Dataset 1 
*   The model is trying to predict Heart Disease of a patient using multiple given attributes

## Dataset 2
*   The model is trying to classify distinct patterns that reflect various heart conditions in a patient 


## Dataset 1
*   Essential libraries required to run the model such as  pandas, numpy, matplotlib, seabord and sklearn are imported
*   Models ie, DecisionTree, RandomForest, Logistic Regression, KMeans, KNN, GradientBoosting, NaiveBayes, SVM that are required to train the dataset is also imported
*   From sklearn library Standard Scalar is import to Normalise the data


*   Then the data is passed through a decision tree estimation model to find out the suitable deptj and sample split using the GridSearcCV method
*   The data is then fed into a loop of model when it predicts the accuracy
*   To improve the accuracy added a Kfold algorith with split 5 and ran the loop with it
*   KMeans algorithm is ran though the data, by finding the suitable k number using the Elbow method

## Dataset 2

*   deep learning model is created and complied to feed the data
*   the data is fitted into the model with an epochs of 25 prediting its accuracy score

#   8.

# 9. FilesandFolders
*   The repository contains 2 folder (ProjectOne and ProjectTwo) and a readme file
*   ProjectOne is the Heart Disease Prediction and contains the 'PorjectOnecode.ipynb' which contains the code for the analysis, 'heart_statlog_cleveland_hungary_final.csv' contains the data for the analysis
*   ProjectTwo is the ECG Image classificaiton and contains the 'ProjectTwoCode.ipynb' which contains the code for the analysis and ECG_DATA folder containing the dataset used for the analysis
