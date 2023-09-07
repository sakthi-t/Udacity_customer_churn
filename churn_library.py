'''
Author: Sakthivel T
Date: September 2023
'''


# import libraries
import os
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
# import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# plot_roc_curve is depreciated so using RocCurveDisplay

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    image_directory = "images/eda"

    # Creating the churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Creating a histogram for churn column and storing it in images/eda
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Histogram of Churn')
    plt.xlabel('Churn')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(image_directory, 'churn_distribution.png'))
    plt.close()

    # Creating a histogram for age distribution column and storing it in
    # images/eda
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Histogram of Customer Age')
    plt.xlabel('Customer Age')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(image_directory, 'customer_age_distribution.png'))
    plt.close()

    # Creating a bar plot for marital status and storing it in images/eda
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Bar Plot of Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion/Percentage')
    plt.savefig(
        os.path.join(
            image_directory,
            'marital_status_distribution.png'))
    plt.close()

    # Creating a kde plot for total transaction and storing it in images/eda
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('KDE Plot of Total Transaction')
    plt.xlabel('Total Transaction')
    plt.ylabel('Density')
    plt.savefig(
        os.path.join(
            image_directory,
            'total_transaction_distribution.png'))
    plt.close()

    # Creating a heatmap and storing it in images/eda
    # using df.corr() results in ValueError. ValueError: could not convert string to float: 'Existing Customer'
    # Selecting only numeric columns to avoid ValueError
    numeric_columns = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        numeric_columns.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Heatmap')
    plt.savefig(os.path.join(image_directory, 'heatmap.png'))
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        if category != response:
            try:
                # Create a dictionary mapping each category to its
                # corresponding churn rate
                category_churn_dict = df.groupby(
                    category)[response].mean().to_dict()
                encoded_column = f'{category}_Churn'
                df[encoded_column] = df[category].map(category_churn_dict)
            except Exception as e:
                print(f"Error in column '{category}': {e}")

    # Debug print to check the DataFrame columns after encoding
    print("Columns after encoding:")
    print(df.columns)

    return df


def perform_feature_engineering(df, response):
    '''
    input:
            df: pandas dataframe
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    # Creating the churn column
    # df['Churn'] = df[response].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Features to work with
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = df[keep_cols]
    y = df[response]

    # Split the data into train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                cv_rfc,
                                lrc):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            cv_rfc: GridSearchCV object for Random Forest
            lrc_plot: RocCurveDisplay object for Logistic Regression
            lrc: LogisticRegression model
    output:
            None
    '''
    # Classification reports
    train_report_lr = classification_report(y_train, y_train_preds_lr)
    test_report_lr = classification_report(y_test, y_test_preds_lr)
    train_report_rf = classification_report(y_train, y_train_preds_rf)
    test_report_rf = classification_report(y_test, y_test_preds_rf)

    # Save classification reports as images in 'images/results'
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(train_report_rf), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(test_report_rf), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/rf_results.png')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(train_report_lr), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(test_report_lr), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/logistic_results.png')
    plt.close()

    # Plot ROC curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp.plot(ax=ax, alpha=0.8)
    plt.savefig('images/results/roc_curve_result.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''
    if isinstance(model, RandomForestClassifier):
        # Calculate feature importances
        importances = model.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        # Save the plot to the specified output path
        plt.savefig(output_pth)
        plt.close()

    elif isinstance(model, LogisticRegression):
        # Get the absolute coefficients of the logistic regression model
        coefficients = np.abs(model.coef_[0])

        # Sort coefficients in descending order
        indices = np.argsort(coefficients)[::-1]

        # Rearrange feature names so they match the sorted coefficients
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance (Logistic Regression)")
        plt.ylabel('Absolute Coefficient Value')

        # Add bars
        plt.bar(range(X_data.shape[1]), coefficients[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        # Save the plot to the specified output path
        plt.savefig(output_pth)
        plt.close()
    else:
        print("Unsupported model type for feature importance visualization")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    output:
            None
    '''
    # Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)

    # added few more max_features in param_grid to avoid following error
    # sklearn.utils._param_validation.InvalidParameterError: The
    # 'max_features' parameter of RandomForestClassifier must be an int in the
    # range [1, inf), a float in the range (0.0, 1.0], a str among {'log2',
    # 'sqrt'} or None. Got 'auto' instead.
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2', 0.5, 0.7],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Logistic Regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Classification report images and model storage
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        cv_rfc,
        lrc)

    # Call feature_importance_plot function for Random Forest
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        'images/results/rfc_feature_importance.png')

    # Call feature_importance_plot function for Logistic Regression
    feature_importance_plot(
        lrc, X_train, 'images/results/lrc_feature_importance.png')

    # Save the best models
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')


if __name__ == "__main__":
    CSV_PATH = "./data/bank_data.csv"
    raw_df = import_data(CSV_PATH)
    # Printing sum of null
    print("Sum of null:")
    print(f"{raw_df.isnull().sum()}")
    # Printing the head of the dataframe
    print(raw_df.head())
    # Printing the shape of the data
    print()
    print(f"Shape of the data {raw_df.shape}")
    print()
    print("Basic Statistics: ")
    print(f"{raw_df.describe()}")

    # Perform EDA and save images to images/eda
    perform_eda(raw_df)

    # List of categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df_encoded = encoder_helper(raw_df, cat_columns, 'Churn')
    print("Encoded DF Head")
    print(df_encoded.head())

    # Perform feature engineering to get the training and testing data
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_encoded, 'Churn')

    # Call the train_models function
    train_models(X_train, X_test, y_train, y_test)
