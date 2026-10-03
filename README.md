# **House Price Prediction Using Multiple Linear Regression**
## **1. Project Purpose**
### 
This project develops a machine learning model to predict the market price of houses in Rwanda.

The model uses different characteristics of a house, such as its area, number of bedrooms, number of bathrooms, age, distance from the city centre, parking spaces, and neighborhood, to estimate the house price in million Rwandan Francs (RWF).

The project uses Multiple Linear Regression and provides a Streamlit web application where a user can enter house details and receive an estimated house price. ###
## **2. Dataset** ##
###
The dataset used in this project is:

house_price_prediction_dataset.csv

The dataset contains information about houses and their selling prices.
 ###
 ## **Dataset Variables**
 ###  
  <table>
    <thead>
      <tr>
        <th>Variable</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
        <tr>
            <td>House_ID</td>
            <td>Unique identifier for each house. It was not used as a predictor.</td> 
        </tr>
         <tr>
            <td>Area_m2</td>
            <td>Floor area of the house in square metres</td> 
        </tr>
        <tr>
            <td>Bedrooms</td>
            <td>Number of bedrooms</td> 
        </tr>
        <tr>
            <td>Bathrooms</td>
            <td>Number of bathrooms</td> 
        </tr>
        <tr>
            <td>House_Age_Years</td>
            <td>Age of the house in years</td> 
        </tr>
        <tr>
            <td>Distance_to_City_km</td>
            <td>Distance from the city centre in kilometres</td> 
        </tr>
        <tr>
            <td>Parking_Spaces</td>
            <td>Number of parking spaces</td> 
        </tr>
         <tr>
            <td>Neighborhood</td>
            <td>Location of the house</td> 
        </tr>
      
  </table> 


 ### The target variable for prediction is:

## 
House_Price_Million_RWF 
###
## 3. Data Cleaning and Preparation ##
### 
The following steps were performed:

1. Missing values were checked.
2. Duplicate records were detected and removed.
3. Impossible values, such as negative distances or house ages, were checked.
4. Outliers were detected using the Interquartile Range (IQR) method.
5. Identified numerical outlier records were removed.
6. Missing numerical predictor values are handled using median imputation.
7. Missing categorical values are handled using the most frequent category.
8. House_ID was excluded because it is only an identifier and is not a useful predictor.
9. Neighborhood was converted into numerical variables using one-hot encoding.

After cleaning, the dataset contained 91 observations.
##
## How the Model Was Built ##
 ### The machine learning model was developed using scikit-learn.

###
The predictors used were: 
* Area
* Bedrooms
* Bathrooms
* House Age
* Distance to City Centre
* Parking Spaces
* Neighborhood 
##
The target variable was:
* House Price in Million RWF
###
### 
The dataset was divided into:
 * 80% training data
 * 20% testing data
 ### 
 The split used:

random_state = 42

A scikit-learn Pipeline was used to combine data preprocessing and the regression model.

The categorical variable Neighborhood was encoded using one-hot encoding, with one category dropped to avoid the dummy-variable trap.

The final model is:

## Multiple Linear Regression

### Model Performance

The model was evaluated using:

* R²
* Adjusted R²
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Performance Results
<table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>R²</th>
        <th>Adjusted R²</th>
        <th>MAE</th>
        <th>RMSE</th>
      </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training</td>
            <td>0.7666</td>
            <td>0.7238</td>
            <td>15.84</td>
            <td>19.61</td>   
        </tr>
        <tr>
            <td>Testing</td>
            <td>0.7759</td>
            <td>0.4237</td>
            <td>16.33</td>
            <td>18.87</td>   
        </tr>
      
  </table> 

  The test R² is 0.7759, meaning that the model explains approximately 77.59% of the variation in house prices in the test data.

The test RMSE is 18.87 million RWF, meaning the typical size of prediction error, measured using RMSE, is approximately 18.87 million RWF.

All price values are expressed in million RWF.
##
### Model File 
###
##
The trained model is saved as:

house_price_model.sav

The model was saved using joblib.

The saved file contains the complete machine learning pipeline, including:

* Missing-value preprocessing
* One-hot encoding
* Multiple Linear Regression model

This allows the saved model to be loaded and used directly for predictions.

###  Streamlit Web Application
##

A Streamlit web application was created to allow users to enter house information and obtain a predicted price.

The application provides input fields for:

* Area
* Bedrooms
* Bathrooms
* House Age
* Distance to City Centre
* Parking Spaces
* Neighborhood

By entering or adjusting  the Predict any of the se input field , the application displays the estimated house price in million RWF.

## How to Run the App Locally


### Step 1: Install Python

Make sure Python 3.10 or newer is installed.

### Step 2: Open the project folder

Open a terminal or command prompt inside the project folder.

### Step 3: Install the required libraries

Run:

pip install -r requirements.txt

The required libraries include:

pandas

numpy

matplotlib

scikit-learn

joblib

streamlit
### Step 4: Run the Streamlit application

Run:

streamlit run app.py
Step 5: Open the application

Streamlit will provide a local address, normally similar to:

http://localhost:8501

Open this address in a web browser.

## RUN APP REMOTELY(ONLINE)
visit the link below 

https://01a0fc20-d767-8a23-da82-b8de5be90964.share.connect.posit.cloud/
## Project Files structure

The project contains the following important files:

house-price-predictor/

│

├── app.py

├── house_price_model.sav

├── house_price_prediction_assignment.ipynb

├── house_price_prediction_dataset.csv

├── cleaned_house_price_dataset.csv

├── requirements.txt

├── README.md

└── house_price_prediction_report.pdf


### File descriptions
* app.py — Streamlit web application
* house_price_model.sav — trained machine learning model
* index.ipynb — data analysis and machine learning code

* house_price_prediction_dataset.csv — original dataset
* cleaned_house_price_dataset.csv — cleaned dataset
* requirements.txt — Python dependencies
* README.md — project documentation
* house_price_prediction_report*.pdf — project report

## Example Prediction

For example, a house with:

* Area: 150 m²
* Bedrooms: 3
* Bathrooms: 2
* House Age: 5 years
* Distance to City Centre: 4 km
* Parking Spaces: 1
* Neighborhood: Gasabo

produces an estimated price of approximately:

175.40 million RWF

## Conclusion 
 
This project demonstrates how Multiple Linear Regression can be used to predict house prices using several characteristics of houses in Rwanda.

The trained model achieved a test R² of 0.7759 and a test RMSE of 18.87 million RWF.

The model has been saved as house_price_model.sav and integrated into a Streamlit application so that users can enter house characteristics and obtain a price prediction.