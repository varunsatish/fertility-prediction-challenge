"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    ## This script contains a bare minimum working example
    random.seed(1) # not useful here because logistic regression deterministic

    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")   # onehot encoding categorical vars
    numerical_preprocessor = StandardScaler()       # standardizing numerical vars 

    categorical_columns = ["cf20m128", "oplzon_2020"]
    numerical_columns = ["nettohh_f_2020", "age"]

    # preprocessor function that transforms columns of a dataframe 
    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_columns),
        ('standard_scaler', numerical_preprocessor, numerical_columns)
        ])
    
    # Logistic regression model
    model = make_pipeline(preprocessor, LogisticRegression())

    # Fit the model
    features_to_select = [
        "age",
        "cf20m128", 
        "oplzon_2020", 
        "nettohh_f_2020", 
        "nomem_encr"
        ]

    #model.fit(model_df[features_to_select], model_df["new_child"])

    model = LogisticRegression()
    model.fit(model_df[features_to_select], model_df['new_child'])


    # Save the model
    joblib.dump(model, "model.joblib")
