import logging
import pyodbc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import azure.functions as func

# Set logging level
logging.basicConfig(level=logging.INFO)

app = func.FunctionApp()

@app.timer_trigger(schedule="0 0 0 * * *", arg_name="myTimer", run_on_startup=False, use_monitor=False)
def moli_dev_ai_es_fa(myTimer: func.TimerRequest) -> None:
    logging.info('Python Timer trigger function started.')

    if myTimer.past_due:
        logging.info('The timer is past due!')

        logging.info('Python Timer trigger function started.')

    # SQL connection string
    connection_string = (
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=moli-dev-es-sqlserver.database.windows.net,1433;"
        "Database=moli-dev-es-sqldb;"
        "Uid=moli-dev-es-sql-admin;"
        "Pwd=HyIa0MInqwdpD3N;"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    try:
        # Establish the connection
        conn = pyodbc.connect(connection_string)
        logging.info('Connected to SQL Server.')

        # Define the query for price prediction
        query = """
        SELECT
            j.language,
            j.modality_id,
            j.certification,
            e.state_id,
            COALESCE(p.min_minutes_pay, j.rate) AS rate
        FROM
            tbl_jobs j
        JOIN
            tbl_languagemaster l ON j.language = l.lang_id
        JOIN
            tbl_jobapplications ja ON j.job_id = ja.job_id
        JOIN
            tbl_interpreters i ON ja.interpreter_id = i.interpreter_id
        JOIN
            tbl_entities e ON j.entity_id = e.entity_id
        JOIN
            tbl_certificatemaster c ON c.cert_id = j.certification
        LEFT JOIN
            tbl_paymentrecords p ON p.job_id = j.job_id AND p.interpreter_id = ja.interpreter_id
        WHERE
            j.entity_id NOT IN (12)
            AND ja.application_status = 1
            AND ja.is_deleted = 0
            AND j.is_deleted = 0
        ORDER BY j.job_id
        """

        # Read the data from SQL Server into a DataFrame
        df = pd.read_sql(query, conn)

        # Feature selection
        X = df[['state_id', 'language', 'certification', 'modality_id']]
        y = df['rate']

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the RandomForestRegressor model
        model = RandomForestRegressor(random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Predict the rate for each unique combination
        unique_combinations = df[['state_id', 'language', 'certification', 'modality_id']].drop_duplicates()
        unique_combinations['predicted_rate'] = model.predict(unique_combinations)
        unique_combinations['predicted_rate'] = unique_combinations['predicted_rate'].apply(lambda x: round(x / 5) * 5)

        # Insert predicted values into the SQL table
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE tbl_moli_ai_pricePrediction")
        
        unique_combinations = unique_combinations.astype({
            'state_id': int,
            'language': int,
            'certification': int,
            'modality_id': int,
            'predicted_rate': int
        })
        
        sql = """
        INSERT INTO tbl_moli_ai_pricePrediction (state_id, language, certification, modality_id, predicted_rate)
        VALUES (?, ?, ?, ?, ?)
        """
        for index, row in unique_combinations.iterrows():
            cursor.execute(sql, (
                int(row['state_id']),
                int(row['language']),
                int(row['certification']),
                int(row['modality_id']),
                float(row['predicted_rate'])
            ))

        conn.commit()
        logging.info("Data inserted into tbl_moli_ai_pricePrediction table successfully.")
        
        query = """
        SELECT
        j.entity_id,
        j.modality_id,
        j.language,
        DATEDIFF(MINUTE,
                    CASE
                        WHEN j.job_id < 432 THEN j.date_of_job_post
                        ELSE jal.time_stamp
                    END,
                    ja.application_date) AS time_difference_minutes,

        j.certification,
            e.state_id
        FROM
        tbl_jobs j
        JOIN
        tbl_jobapplications ja ON j.job_id = ja.job_id
        JOIN
        tbl_interpreters i ON ja.interpreter_id = i.interpreter_id
        JOIN
        tbl_entities e ON j.entity_id = e.entity_id
        JOIN
        tbl_certificatemaster c ON c.cert_id = j.certification
        LEFT JOIN
        tbl_jobactivitylog jal ON j.job_id = jal.job_id AND jal.activity_type = 9
        WHERE
        ((j.job_id < 432 AND DATEDIFF(MINUTE, j.date_of_job_post, ja.application_date) >= 0) OR
            (j.job_id > 432 AND DATEDIFF(MINUTE, jal.time_stamp, ja.application_date) >= 0))
        AND j.entity_id NOT IN (12)
        AND ja.application_status = 1
        AND ja.is_deleted = 0
        AND j.is_deleted = 0
            ORDER by j.job_id

        """
        data = pd.read_sql(query, conn)

        # Filter the data to limit time_difference_minutes to only 2880 minutes
        data = data[data['time_difference_minutes'] <= 2880]

        # Identify and remove outliers using the IQR method
        Q1 = data['time_difference_minutes'].quantile(0.25)
        Q3 = data['time_difference_minutes'].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers from the target variable
        data = data[(data['time_difference_minutes'] >= lower_bound) & (data['time_difference_minutes'] <= upper_bound)]

        # Select the features and target variable
        X = data[['state_id','entity_id', 'modality_id', 'certification', 'language']]
        y = data['time_difference_minutes']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # One-hot encode categorical features
        categorical_features = ['state_id','entity_id', 'modality_id', 'certification', 'language']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        pipeline_gb = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])

        param_grid_gb = {
            'regressor__n_estimators': [100, 150],
            'regressor__learning_rate': [0.05, 0.1],
        }

        grid_search_gb = GridSearchCV(estimator=pipeline_gb, param_grid=param_grid_gb, cv=3, scoring='r2')
        grid_search_gb.fit(X_train, y_train)
        best_gb = grid_search_gb.best_estimator_

        unique_combinations = data[['state_id', 'entity_id', 'modality_id', 'certification', 'language']].drop_duplicates()
        predicted_time_differences = best_gb.predict(unique_combinations)

        # Subtract 40 from the predicted time differences and round to the nearest multiple of 15
        adjusted_time_differences = [
            abs(round((time_diff - 40) / 5) * 15) for time_diff in predicted_time_differences
        ]

        # Add the adjusted time differences to the unique combinations DataFrame
        unique_combinations['adjusted_time_difference_in_minutes'] = adjusted_time_differences

        cursor.execute("TRUNCATE TABLE tbl_moli_ai_timePrediction")

        # Convert DataFrame columns to native Python types
        unique_combinations = unique_combinations.astype({
            'state_id': int,
            'entity_id': int,
            'modality_id': int,
            'certification': int,
            'language': int,
            'adjusted_time_difference_in_minutes': int
        })


        sql = """
        INSERT INTO tbl_moli_ai_timePrediction (state_id, entity_id, modality_id, certification, language, adjusted_time_difference_in_minutes)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        for index, row in unique_combinations.iterrows():
            cursor.execute(sql, (
                int(row['state_id']),
                int(row['entity_id']),
                int(row['modality_id']),
                int(row['certification']),
                int(row['language']),
                int(row['adjusted_time_difference_in_minutes'])
            ))

        conn.commit()

        logging.debug("Data inserted into tbl_moli_ai_timePrediction table successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    
    finally:
        if conn:
            conn.close()
            logging.info('SQL connection closed.')