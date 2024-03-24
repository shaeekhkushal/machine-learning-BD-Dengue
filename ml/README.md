# Dengue Prediction and Analysis

This project uses Python, pandas, and scikit-learn to predict the occurrence of Dengue based on various factors such as Gender, Area, AreaType, HouseType, and District. It also analyzes the rates of Dengue by gender and area.

## Description

The script reads data from a CSV file, encodes categorical features using scikit-learn's LabelEncoder, and splits the data into training and test sets.

A Logistic Regression model is then trained on the training data and used to make predictions on the test data. The accuracy of the model and a classification report are printed to the console.

A confusion matrix of the classifier's predictions is generated and displayed as a heatmap using seaborn.

The script also calculates the rates of Dengue by gender and area, and prints these rates to the console. The rates are calculated as the mean 'Outcome' for each group of 'Gender' or 'Area'.

Finally, the Dengue rates by gender and area are saved to CSV files for further analysis.

## Usage

To run the script, you need to have Python and the required libraries (numpy, pandas, scikit-learn, matplotlib, seaborn) installed. You also need to have the data file 'data_files/dataset.csv' in the same directory as the script.

To run the script, use the following command:

```
python script_name.py
```

Replace `script_name.py` with the name of the script file.

After running the script, you will find two new CSV files in the same directory: 'gender_dengue.csv' and 'area_dengue.csv'. These files contain the Dengue rates by gender and area, respectively.