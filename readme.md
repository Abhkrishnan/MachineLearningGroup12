# GROUP_12
<br></br>
## Heart disease Prediction & ECG image classification of Cardiac Patients

## Group Members
Abhay Krishnan ,Aghil, Akul, Edwin, Hilal

# Objective
* Predicting heart disease using machine learning techniques.
    *It explores various classifiers like Logistic Regression, Decision Trees,Random Forest to analyze heart disease factors, such as cholesterol levels, chest pain types, and blood pressure. 
    *The notebook aims to build predictive models using structured health data.

* ECG image classification
    *ECG (Electrocardiogram) images contain distinct patterns that reflect various heart conditions. 
    *By applying deep learning techniques, such as convolutional neural networks (CNNs), these visual patterns can be automatically analyzed to classify different types of heart abnormalities, such as arrhythmias, ischemia, or myocardial infarction. 

## Source
* The source of Dataset 1 and Dataset 2 is from Kaggle

## Lisence
### Dataset 1 - Predicting Heart Disese
* Link of the Dataset https://www.kaggle.com/datasets/mexwell/heart-disease-dataset
* Link of Lisense https://creativecommons.org/licenses/by/4.0/

### Dataset 2 - ECG Image Classification
* Link of the dataset https://www.mit.edu/~amini/LICENSE.md
* Link of Lisense https://www.mit.edu/~amini/LICENSE.md

## Preview of Dataset

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

# How  to Run the Model
*   Essential libraries required to run the model such as  pandas, numpy, matplotlib, seabord and sklearn are imported
*   Models ie, DecisionTree, RandomForest, Logistic Regression, KMeans, KNN, GradientBoosting, NaiveBayes, SVM that are required to train the dataset is also imported
*   From sklearn library Standard Scalar is import to Normalise the data
*   the data is loaded using pandas library, and it is procceded with initial inspection and visualisation (using seaborn)
*   Then the data is preprocesed to check the missing values and the outliers
    The data is feature scaled using Standard scalers as there are outliers found
