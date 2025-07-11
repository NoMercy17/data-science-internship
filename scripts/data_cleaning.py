import pandas as pd
import numpy as np
from IPython.display import display
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
#import matplotlib.pyplot as plt
#import seaborn as sns
#from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

data= pd.read_csv("/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/raw/hotel_booking_cancellation_prediction.csv")

hotel_data = data.copy()

def handle_missing_values(data):
    """Process each column based on data type and missing values"""
    
    for column in data.columns:
        print(f"\n Processing column: {column}")
        dtype = data[column].dtype
      
        print(f"Data type: {dtype}")
        print(f"Missing values: {data[column].isnull().sum()} ({data[column].isnull().mean():.1%})")
        
        # For numeric columns
        if np.issubdtype(dtype, np.number):
            print(f"Zeros count: {(data[column] == 0).sum()}")
            print(f"Stats: mean={data[column].mean():.2f}, median={data[column].median():.2f}")
        if data[column].isnull().mean() > 0.5:  # If over 50% missing
            print(f"DROPPING COLUMN '{column}' (over 50% missing values)")
            data.drop(column, axis=1, inplace=True)
            
        elif np.issubdtype(dtype, np.number):  # Numeric columns
            # Replace invalid zeros with median
            median_val = data[column].median()
            zero_mask = (data[column] == 0) & (data[column].notnull())
            data.loc[zero_mask, column] = median_val
            
            # Fill remaining NaNs with median
            data[column].fillna(median_val, inplace=True)
            
        else:  # Categorical columns
            mode_val = data[column].mode()[0]
            data[column].fillna(mode_val, inplace=True)
            print(f"Filled missing values with mode: {mode_val}")

    print("\n Processing complete. Remaining missing values per column:")
    print(data.isnull().sum())

    return data


handle_missing_values(hotel_data)
dtype_df = hotel_data.dtypes.to_frame(name='dtype').reset_index()
dtype_df.columns = ['column', 'dtype']
display(dtype_df)
print(hotel_data["market_segment"], hotel_data["distribution_channel"])


def handle_duplicates(data):
    nr_duplicates = hotel_data.duplicated().sum()

    if nr_duplicates: 
        data = data.drop_duplicates()
        print(f"Duplicates removed. New shape: {hotel_data.shape}")
    else:
        print("No duplicates found.")
    return data


def handle_statistical_outliers(data):
    pass


def handle_data_entry_errors(data):
    pass

def handle_logical_errors(data):
    pass

def handle_context_dependent_outliers(data):
    pass

def handle_infrequent_categories(data):
    pass

def handle_target_leakage_outliers(data):
    pass

def handle_outliers(data):
    print(" Starting Comprehensive Outlier Handling")
    print("=" * 50)
    original_shape = data.shape
    print(f"Original data shape: {original_shape}")

    # extreme but valid values
    print("\n" + "="*60)
    data = handle_statistical_outliers(data)

    # clear data entry errors
    print("\n" + "="*60)
    data = handle_data_entry_errors(data)

    # handle logical errors
    print("\n" + "="*60)
    data = handle_logical_errors(data)

    # handle context-dependent outliers
    print("\n" + "="*60)
    data = handle_context_dependent_outliers(data)

    # handle infrequent categories
    print("\n" + "="*60)
    data = handle_infrequent_categories(data)

    # handle target leakage outliers
    print("\n" + "="*60)
    data = handle_target_leakage_outliers(data)

def clean_data(data):
    data = handle_duplicates(data)
    data = handle_missing_values(data)
    data = handle_outliers(data)
    #data = standardize_data_types(data)
    return data

