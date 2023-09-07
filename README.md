# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
* The purpose of this project is to predict customers who are likely to churn.
* The entire project is described in the notebook titled 'churn_notebook.ipynb'
* The code in the notebook has to be refactored and written in churn_library.py
* Test cases has to be written in churn_scripts_logging_and_tests.py
* The images from exploratory data analysis is stored in images/eda
* The results (images) are stored in images/results
* The logs is stored in logs directory
* The models are stored in models directory
* Overall the code is refactored and images, logs, and models are stored in different directories
* The data is first imported using import_data method
* Data analysis is performed using perform_eda method
* Encoding is done using encoder_helper method
* Feature engineering is performed using perform_feature_engineering method
* In the classification_report_image method rf_results.png, logistic_results.png, roc_curve_result.png are generated
* The feature_importance_plot method contains code to generate logistic and random forest feature importance plots
* The train_moels method contains the code to train the models. The classification_report_image and feature_importance_plot are called inside the train models. The model files logistic_model.pkl and rfc_model.pkl files are also generated. 
* The main method contains some useful print statements like df.head() and the methods are called inside the main method. 
* The churn_script_logging_and_tests.py contains the corresponding tests for churn_library.py

## Files and data description
* The data is stored in data directory
* The eda images are stored in images/eda directory
* The results images are stored in images/results directory
* The logs are stored in logs directory
* The models are stored in models directory
* In images/eda customer_distribution.png, customer_age_distribution.png, heatmap.png, marital_status_distribution.png, total_transaction_distribution.png are generated
* In images/results logistic_results.png, lrc_feature_importance.png (logistic regression feature importance), rf_results.png (random forest results), rfc_feature_importance.png (random forest feature importance), roc_curve_result.png are generated
* The models directory contain logistic_model.pkl and rfc_model.pkl files
* The logs directory contains churn_library.log

## Running Files
* Before running the churn_library pip install the dependencies.
* The dependencies are in requirements.txt
* Install the dependencies from requirements.txt individually or collectively using pip install
* Activate the environment (env) env\Scripts\activate in Windows
* Run python churn_library.py
* The churn_library.py would take about 30 minutes to run because of Grid Search CV
* The eda images would be generated and stored in images/eda
* The results images would be generated and stored in images/results
* The models would be generated and stored in models directory
* Run python churn_script_logging_and_tests.py
* This would test whether the images are there in eda and results directory, model files are in models directory and test preprocessing and encoder_helper
* After the tests are run a log file would be generated in logs directory. 


## Notes
* Pylint score of 10 can't be achieved as that would mean changing even variable names like df, X_train, X_test, y_train, y_test
* Renaming df, X_train, X_test, y_train, y_test at this point would break the code
* The pylint score for churn_library.py is 8.15/10
* The pylint score for churn_script_logging_and_tests.py is 8.98/10
* Changed df to raw_df in churn_script_logging_and_tests.py to avoid df is not following snake case convention warning while using pylint
* Did not change df to raw_df in churn_library.py as it would lead to changes in so many places and the code might break
* It important to note that plot_roc_curve is depreceated so using RocCurveDisplay in churn_library.py
* autopep8 is applied to the churn_library.py and churn_script_logging_and_tests.py
