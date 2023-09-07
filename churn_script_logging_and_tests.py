'''
Author: Sakthivel T
Date: September 2023
'''


import os
import logging
import pandas as pd
import numpy as np
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        raw_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert raw_df.shape[0] > 0
        assert raw_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            f"Testing import_data: The file doesn't appear to have rows and columns - {str(err)}")
        raise err


def test_eda(perform_eda, import_data):
    '''
    test perform eda function
    '''
    image_directory = "images/eda"

    try:
        raw_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        perform_eda(raw_df)

        assert os.path.exists(
            os.path.join(
                image_directory,
                'churn_distribution.png'))
        assert os.path.exists(
            os.path.join(
                image_directory,
                'customer_age_distribution.png'))
        assert os.path.exists(os.path.join(image_directory, 'heatmap.png'))
        assert os.path.exists(
            os.path.join(
                image_directory,
                'marital_status_distribution.png'))
        assert os.path.exists(
            os.path.join(
                image_directory,
                'total_transaction_distribution.png'))
        logging.info(
            "SUCCESS: EDA Images are generated and exists in images directory")

    except AssertionError as err:
        logging.error(
            f"ERROR: The image files from EDA does not seem to exist in images directory - {str(err)}")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    Creating a sample DataFrame and testing whether the encoder_helper works
    '''
    # Create a sample DataFrame
    sample_data = {
        'Category1': ['A', 'B', 'A', 'B'],
        'Category2': ['X', 'Y', 'X', 'Z'],
        'Response': [1, 0, 1, 0]
    }
    sample_df = pd.DataFrame(sample_data)

    # Define the list of categorical columns and response variable
    category_lst = ['Category1', 'Category2']
    response = 'Response'

    try:
        encoded_df = encoder_helper(sample_df, category_lst, response)

        # Check if the generated DataFrame has the expected columns
        expected_columns = ['Category1_Churn', 'Category2_Churn']
        assert all(col in encoded_df.columns for col in expected_columns)
        logging.info("SUCCESS: Tested encoder_helper")
    except AssertionError as err:
        logging.error(f"Testing encoder_helper: Error - {str(err)}")
        raise err


def generate_sample_data():
    '''
    A sample data which will be used in test_perform_feature_engineering
    '''
    np.random.seed(42)
    data = {
        'Customer_Age': np.random.randint(18, 70, size=1000),
        'Dependent_count': np.random.randint(0, 6, size=1000),
        'Total_Relationship_Count': np.random.randint(1, 7, size=1000),
        'Months_Inactive_12_mon': np.random.randint(0, 7, size=1000),
        'Credit_Limit': np.random.uniform(1000, 50000, size=1000),
        'Total_Revolving_Bal': np.random.uniform(0, 3000, size=1000),
        'Total_Trans_Amt': np.random.uniform(0, 10000, size=1000),
        'Churn': np.random.choice([0, 1], size=1000),
        'Months_on_book': np.random.randint(12, 48, size=1000),
        'Contacts_Count_12_mon': np.random.randint(0, 6, size=1000),
        'Avg_Open_To_Buy': np.random.uniform(0, 50000, size=1000),
        'Total_Amt_Chng_Q4_Q1': np.random.uniform(0, 1, size=1000),
        'Total_Trans_Ct': np.random.randint(0, 150, size=1000),
        'Total_Ct_Chng_Q4_Q1': np.random.uniform(0, 1, size=1000),
        'Avg_Utilization_Ratio': np.random.uniform(0, 1, size=1000),
        'Gender_Churn': np.random.choice([0, 1], size=1000),
        'Education_Level_Churn': np.random.choice([0, 1], size=1000),
        'Marital_Status_Churn': np.random.choice([0, 1], size=1000),
        'Income_Category_Churn': np.random.choice([0, 1], size=1000),
        'Card_Category_Churn': np.random.choice([0, 1], size=1000),
    }
    raw_df = pd.DataFrame(data)
    return raw_df


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        raw_df = generate_sample_data()
        response_variable = 'Churn'
        # Call the perform_feature_engineering function with your DataFrame
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            raw_df, response_variable)

    # Check if the returned training and testing data have the expected shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        logging.info(
            "SUCCESS: Tested perform_feature_engineering with sample data")

    except AssertionError as err:
        logging.error(
            f"ERROR: Error during perform_feature_engineering with sample data - {str(err)}")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        # check if rf_results.png, logistic_results.png, roc_curve_result.png
        # has been created
        assert os.path.exists('images/results/rf_results.png')
        assert os.path.exists('images/results/logistic_results.png')
        assert os.path.exists('images/results/roc_curve_result.png')

        # Check if the feature importance images have been created
        assert os.path.exists('images/results/rfc_feature_importance.png')
        assert os.path.exists('images/results/lrc_feature_importance.png')

        # Check if the model files have been created
        assert os.path.exists('models/rfc_model.pkl')
        assert os.path.exists('models/logistic_model.pkl')

        logging.info(
            "SUCCESS: classification images, feature importance image, and the model file exists.")

    except AssertionError as err:
        logging.error(
            f"ERROR: Any of the classification image, feature importance image or the model file does not exist - {str(err)}")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda, cls.import_data)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models()
