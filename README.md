# ML_Guild_2020
terrific tartan submission


# config 
1. data prep 
    1.1. clean data 
        1.1.1 impute nulls/values
            1.1.1.1 BusinessType -> Create a new category 'Not Specified' - SK
            1.1.1.2 Convert NonProfit to Binary indicator - SK
        1.1.2 remove dupes/columns
            1.1.2.1 Unnamed: 0, City, Industry and Zip (skip for now) - SK
        1.1.3 eda - SG
        1.1.4 descriptive analysis - 
        1.1.5 outlier analysis - 
    1.3. feature engineering
        1.3.1 CD - State and DC. Replace Null DC by -1 - SK 
        1.3.2 LoanRange - Mean, Min, Max -  SK
        1.3.3 Weighted_Loan_Amt - Loan_Amount * Wi*Fi (W - weight, F- feature)  - SK
    1.2. data transformation
        1.2.1 label encoding - CD, City, Lender, NAICSCode, State, Zip - SK/ SG
        1.2.2 target/mean encoding  - CD, City, Lender, NAICSCode, State, Zip - SG
        1.2.3 one hot encoding - Veteran, RaceEthnicity, Gender, BusinessType - SG
    1.4. train test split (80/20) - SG
    1.5. models
        1.5.1 XGBoost - SG
        1.5.2 GBM - SG
        1.5.3 RF - SG
        1.5.4 NN - TBD
        1.5.5 LR - SK
        1.5.6 
        
#notebook
data_prep_sg.ipynb
feature_engineering.ipynb
feature_encoding.ipynb
model.ipynb

#file structure
data
    - raw
    - master
code
    - data prep
    - model

