import pandas as pd
import openai
import os
import difflib
import re
import openai
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
openai.api_key = ''
client = OpenAI(api_key="")

class AI:
    def __init__(self):
        pass    
        
    def data_summary(self,df,dataset_name, target_variable):
            # Load the dataset
            # Get basic information about the dataset
            num_records = len(df)
            num_features = len(df.columns)
            feature_names = [str(name) for name in df.columns.tolist()]  # Convert feature names to strings
            data_types = df.dtypes.tolist()
            data_types_str = [str(dtype) for dtype in data_types]
            missing_values = df.isnull().sum().tolist()
            dataset_shape = df.shape
            if target_variable is None:
                 target_variable = "not provided"        
            
             # Generate a brief introduction using GPT API
            prompt = f"this is a dataset of {dataset_name} and it has a target variable {target_variable} overall shape of dataset is {dataset_shape}.The dataset contains {num_records} records and {num_features} features. The features include: {', '.join(feature_names)}. The data types of features are: {', '.join(map(str, data_types_str))}."
            prompt += f" Total missing values in each feature are: {', '.join(map(str, missing_values))}. Here are the summary statistics:\n{df.describe(include='all')}"
            prompt += f" Please provide a Summary of overall dataset in simple words"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            # Print the generated introduction
            print(response.choices[0].message.content)

    def object_correction(self,dataset_name, target_variable, objects):
        obj = objects
        if target_variable is None:
            target_variable = "not provided"
        prompt = f"This dataset is named {dataset_name} and it includes target variables {target_variable}. The following dictionary {obj} contains column names of Object data types along with their unique data values. Please analyze the correct data type for each column and return a dictionary with the column names(original as given in input set) and their corresponding pandas data types (e.g., category, float, datetime, etc.)."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Parse the generated response into dictionary format
        obj_columns = eval(response.choices[0].message.content)
        
        return obj_columns
def poss_corr(df,dataset_name,target_variable):
        dataset_columns = df.columns.tolist()
        data_types = df.dtypes.tolist()
        max_unique_values=10
        object_columns={}
        for col in df.columns:
            unique_values = df[col].unique().tolist()
            if len(unique_values) > max_unique_values:
                unique_values = random.sample(unique_values, max_unique_values)
                object_columns[col] = unique_values
        prompt = f"This dataset is named {dataset_name} and it includes target variable {target_variable}. here are the data columns {dataset_columns} in dataframe with their {data_types} datatypes i want a list of datacolumns at which graph plotting (for univariate) is possibe means ignore name,id,passengerid,and another unusual data column names and don't forget to include target variable and key important variable for the datasets.Please ensure and return only a list of case sesnsitive df columns."
        response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Parse the generated response into dictionary format
        uni_columns = eval(response.choices[0].message.content)
        print(uni_columns)
        return uni_columns
def bi_poss_corr(df,dataset_name,target_variable):
        dataset_columns = df.columns.tolist()
        data_types = df.dtypes.tolist()
        max_unique_values=10
        object_columns={}
        for col in df.columns:
            unique_values = df[col].unique().tolist()
            if len(unique_values) > max_unique_values:
                unique_values = random.sample(unique_values, max_unique_values)
                object_columns[col] = unique_values
        prompt = f"This dataset is named {dataset_name} and it includes target variable {target_variable}. here are the data columns {dataset_columns} in dataframe with their {data_types} datatypes i want a list of datacolumns at which graph plotting (for univariate) is possibe means ignore name,id,passengerid,and another unusual data column names and don't forget to include target variable and key important variable for the datasets.Please return only a list of df columns."
        response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Parse the generated response into dictionary format
        bi_columns = eval(response.choices[0].message.content)
        return bi_columns