# Now, put all your Streamlit code here

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from openai import OpenAI
import os
from scipy.stats import percentileofscore
from utils import create_gauge_chart, create_model_probability_chart, create_percentile_chart



if 'GROQ_API_KEY' in os.environ:
  api_key = os.environ.get('GROQ_API_KEY')
else:
  api_key = st.secrets['GROQ_API_KEY']



# Initialize OpenAI client

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"))

# Function to load a machine learning model from a file
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load the trained machine learning models
xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_clade_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost_featureEngineered.pkl')


# Prepares the input data for the machine learning models.

# This function takes individual input features, converts categorical features
# into numerical representations using one-hot encoding, and creates a
# dictionary that can be used as input for the models.

# Args:
#     credit_score (int): The credit score of the customer.
#     location (str): The geographical location of the customer (France, Germany, Spain).
#     gender (str): The gender of the customer (Male, Female).
#     age (int): The age of the customer.
#     tenure (int): The number of years the customer has been with the bank.
#     balance (float): The account balance of the customer.
#     num_products (int): The number of products the customer has with the bank.
#     has_credit_card (int): Whether the customer has a credit card (1 for yes, 0 for no).
#     is_active_member (int): Whether the customer is an active member (1 for yes, 0 for no).
#     estimated_salary (float): The estimated salary of the customer.

# Returns:
#     dict: A dictionary containing the preprocessed input features.
def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
    # Create a dictionary to store the input features
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_credit_card,
        'IsActivemember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

print(' ')


# Makes churn prediction using multiple models and visualizes the results.

# This function takes the preprocessed input data, calculates churn probabilities
# using three models (XGBoost, Random Forest, K-Nearest Neighbors), calculates
# the average probability, and visualizes the results using Streamlit and Plotly.

# Args:
#     input_df (pd.DataFrame): DataFrame containing the preprocessed input features.
#     input_dict (dict): A dictionary containing the preprocessed input features.



def make_prediction(input_df, input_dict):
    # Calculate churn probabilities for each model
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random_Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest_Neighbors': knn_model.predict_proba(input_df)[0][1],
    }
     # Calculate average churn probability
    avg_probability = np.mean(list(probabilities.values()))
    # Visualize results using Streamlit and Plotly
    col1, col2 =st.columns(2) # Create two columns for layout
    with col1:
        fig =create_gauge_chart(avg_probability) # Create gauge chart using custom function
        st.plotly_chart(fig, use_container_width=True)
        st.write(f'The customer has a { avg_probability:.2%} probability of churning.')
    with col2:
        fig_probs = create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
    

    metrics = ['CreditScore', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    
    percentiles = {}
    for i in metrics:
        percentiles[i] = percentileofscore(df[i], selected_customer[i]) / 100
    
    fig_percentiles = create_percentile_chart(percentiles, metrics)
    st.plotly_chart(fig_percentiles, use_container_width=True)
    
    return avg_probability




def explain_prediction(probability, input_dict, surname):

    prompt = f'''You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

    Here is the customer's information:
    {input_dict}

    Here are the machine learning model's top 10 most important features for predicting churn:

    Feature | Importance
    ----------------------
    NumOfProducts | 0.323888
    IsActiveMember | 0.164156
    Age | 0.158188
    Geography_Germany | 0.091373
    Balance | 0.051816
    Geography_France | 0.046463
    Gender_Female | 0.045823
    Geography_Spain | 0.036585
    CreditScore | 0.036555
    EstimatedSalary | 0.035625
    HasCrCard | 0.031949
    Tenure | 0.030504
    Gender_Male | 0.000000
    {pd.set_option('display.max_colwidth', None)}
    Here are summary statistics for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are summary statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

- If the customer has over 40% risk of churning , (generate a 3-sentence explanation of why they are at the risk of churning)
else
- If the customer has less than 40% risk of churning ,( generate a 3-sentence explanation of why they may not be at risk of churning)
-your explanation sh based on the customer's profile without referencing specific numbers or comparisons to data sets.
**Important:** Do not mention the probability of churning, the machine learning model, any specific model outputs, or do not provide any summary statistics.




'''
    print("EXPLANATION PROMPT:", prompt)

    raw_response = client.chat.completions.create(
        model="llama3-groq-8b-8192-tool-use-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
    You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.

    You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

    Here is the customer's information:
    {input_dict}

    Here is some explanation as to why the customer might be at risk of churning:
    {explanation}

    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

    Make sure to list out a set of incentives to stay based on their information, in bullet point format must in each new lines . Don't ever mention the probability of churning,your name or the machine learning model to the customer.
    """

    raw_response = client.chat.completions.create(
        model="llama3-groq-8b-8192-tool-use-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    print("\n\nEMAIL PROMPT:", prompt)
    return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")
df = pd.read_csv('churn.csv')
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox('Select a customer', customers)
if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(' - ')[0])
    selected_surname = selected_customer_option.split(' - ')[1]
    selected_customer = df.loc[df['CustomerId'] ==
                               selected_customer_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input('Credit Score',
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))
        location = st.selectbox('Location', ['Spain', 'France', 'Germany'],
                                index=['Spain', 'France', 'Germany'
                                       ].index(selected_customer['Geography']))

        gender = st.radio(
            'Gender', ['Male', 'Female'],
            index=0 if selected_customer['Gender'] == 'Male' else 1)

        age = st.number_input('Age',
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer['Age']))

        tenure = st.number_input('Tenure',
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input('Balance',
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))
        num_products = st.number_input('Number of Products',
                                       min_value=1,
                                       max_value=10,
                                       value=int(
                                           selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox('Has Credit Card',
                                      value=bool(
                                          selected_customer['HasCrCard']))
        is_active_member = st.checkbox(
            'Is Active Member',
            value=bool(selected_customer['IsActiveMember']))
        estimated_salary = st.number_input(
            'Estimated Salary',
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    # Predict button
    if st.button('Predict'):
        # Preparing input for prediction
        input_df, input_dict = prepare_input(credit_score, location, gender,
                                             age, tenure, balance,
                                             num_products, has_credit_card,
                                             is_active_member,
                                             estimated_salary)

        # Making the prediction and generating explanation
        avg_probability = make_prediction(input_df, input_dict)
        explanation = explain_prediction(avg_probability, input_dict,
                                         selected_customer['Surname'])

        # Displaying the prediction explanation
        st.markdown('----')
        st.subheader('Explanation of Prediction')
        st.markdown(explanation)

        # Generating and displaying the email to the customer
        email = generate_email(avg_probability, input_dict, explanation,
                               selected_customer['Surname'])
        st.markdown('----')
        st.subheader('Email to Customer')
        st.markdown(email)
