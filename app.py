from flask import Flask, render_template, request
import subprocess
import os
import warnings
import openai
import pandas as pd
from openai import OpenAI
import text_generation
from data_cleaning import DatasetCleaning
from univariate import uni_analyze_and_visualize

warnings.filterwarnings("ignore")
openai.api_key = ''
client = OpenAI(api_key="")

app = Flask(__name__, static_url_path='/static')

# Set the template directory to the same directory as app.py
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app.template_folder = template_dir

@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    global processing_complete 
    # Get form data
    csv_file = request.files['file']
    url_link = request.form['url_link']
    dataset_name = request.files['file'].filename  # Using the filename as the dataset name

    # Perform Python tasks based on form input
    if csv_file:
        # Load data from the uploaded CSV file
        data = pd.read_csv(csv_file)
    else:
        data = None

    # Start processing data if dataset_name is provided
    if dataset_name and data is not None:
        target_variable = get_target_variable(data , dataset_name)
        if target_variable:
            cleaner = DatasetCleaning(data)
            GPT = text_generation.AI()
            df = pd.DataFrame(data)
            print("AI Describing Dataset Summary: ")
            print("....................................................")
            summary = GPT.data_summary(df, dataset_name, target_variable)
            df_1 = cleaner.remove_duplicates(df)
            objects = cleaner.object_columns(df)
            obj_cols = GPT.object_correction(dataset_name, target_variable, objects)
            df_2 = cleaner.convert_dtypes(df_1, obj_cols)
            df_3 = cleaner.onehot_encode(df_2)
            df_4 = cleaner.remove_missing(df_3)
            print(df_4.head())
            df_5, outliers_count = cleaner.remove_outliers(df_4)
            # features_draw = cleaner.plot_feature_importance(df_5, target_variable)
            uni_analyze_and_visualize(df_5, dataset_name, target_variable)
            print("Before Cleaning:", df.shape, "After Preprocessing Operations:", df_5.shape)
            processing_complete = True
            return "Data processing complete."
        else:
            print("Error: Unable to determine the target variable.")
            return "Error: Unable to determine the target variable."

    else:
        return "Error: No dataset provided."

def get_target_variable(data , dataset_name):
    # Load the dataset
    # Get basic information about the dataset
    df = pd.DataFrame(data)
    num_records = len(df)
    num_features = len(df.columns)
    feature_names = [str(name) for name in df.columns.tolist()]  # Convert feature names to strings
    data_types = df.dtypes.tolist()
    data_types_str = [str(dtype) for dtype in data_types]
    missing_values = df.isnull().sum().tolist()
    dataset_shape = df.shape

    # Generate a brief introduction using GPT API
    prompt = f"this is a dataset of {dataset_name} and overall shape of dataset is {dataset_shape}.The dataset contains {num_records} records and {num_features} features. The features include: {', '.join(feature_names)}. The data types of features are: {', '.join(map(str, data_types_str))}."
    prompt += f"Here are the summary statistics:\n{df.describe(include='all')}"
    prompt += f" Please provide most possible target varaible (only one) of dataset return in a only a single string."
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
    target_variable = (response.choices[0].message.content)
    # If the response is not a string, convert it to a string
    if not isinstance(target_variable, str):
        target_variable = str(target_variable)
    if target_variable:
        print("Target variables found:", target_variable)
    else:
        print("No target variables found.")
        target_variable = None
    return target_variable

if __name__ == '__main__':
    app.run(debug=True)
