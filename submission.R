# This is an example script to generate the outcome variable given the input dataset.
# 
# This script should be modified to prepare your own submission that predicts 
# the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.
# 
# The predict_outcomes function takes a data frame. The return value must
# be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
# should contain the nomem_encr column from the input data frame. The outcome
# column should contain the predicted outcome for each nomem_encr. The outcome
# should be 0 (no child) or 1 (having a child).
# 
# clean_df should be used to clean (preprocess) the data.
# 
# run.R can be used to test your submission.

# List your packages here. Don't forget to update packages.R!
library(dplyr) # as an example, not used here
library(caret)

clean_df <- function(df, background_df = NULL){
  # Preprocess the input dataframe to feed the model.
  ### If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

  # Parameters:
  # df (dataframe): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
  # background (dataframe): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

  # Returns:
  # data frame: The cleaned dataframe with only the necessary columns and processed variables.

  ## This script contains a bare minimum working example
  # Create new age variable
  #df$age <- 2024 - df$birthyear_bg

  # Selecting variables for modelling

  #keepcols = c('nomem_encr', # ID variable required for predictions,
  #             'age')        # newly created variable
  
  ## Keeping data with variables selected
  #df <- df[ , keepcols ]
  
  
  # copied from Flora's code 21st June
  
  # Variable: gender
  
  df <- df %>%
    mutate(gender = recode(gender_bg, 
                           "1" = "Male",
                           "2" = "Female"))
  
  # Variable: migration background
  df <- df %>% 
    mutate(ethnic = recode(migration_background_bg,
                           "0" = "Dutch background",
                           "101" = "First generation foreign, Western background",
                           "102" = "First generation foreign, non-western background",
                           "201" = "Second generation foreign, Western background",
                           "202" = "Second generation foreign, non-western background"))
  
  # Variable: age & age^2
  
  df <- df %>% 
    mutate(age_2020 = age_bg,
           age_2020_2 = age_bg**2)
  
  
  # Variable: highedu_2020
  # whether respondents had a high education (hbo / wo) or not in 2020
  
  df <- df %>% 
    mutate(highedu_2020 = case_when(
      oplmet_2020 == 5 | oplmet_2020 == 6 ~ "1.high edu",
      (oplmet_2020 >= 1 & oplmet_2020 <=4) | (oplmet_2020 >= 7 & oplmet_2020 <= 9) ~ "0.no high edu"))
  
  # Variable: woonvorm_2020
  
  df <- df %>% 
    mutate(living_arr_2020 = recode(woonvorm_2020,
                                    "1" = "0.single",
                                    "2" = "1.(un)married co-habitation, without child(ren)",
                                    "3" = "2.(un)married co-habitation, with child(ren)",
                                    "4" = "3.single, with child(ren)",
                                    "5" = "4.other"))
  
  # Variable: housing situation
  
  df <- df %>% 
    mutate(housing_2020 = recode(woning_2020,
                                 "1" = "0.self-own dwelling",
                                 "2" = "1.rental dwelling",
                                 "4" = "missing"))
  
  df <- df %>% 
    mutate(housing_2020 = na_if(housing_2020, "missing"))
  
  # Variable: net household income in 2020
  df <- df %>% 
    mutate(log_houseincome_2020 = log(nettohh_f_2020 + 1))
  
  
  # Variable: whether has a partner in 2020 
  
  df <- df %>% 
    mutate(partner_2020 = case_when(
      cf20m024 == 1 ~ "1. have a partner",
      cf20m024 == 2 ~ "0. no partner"))
  
  
  # Variable: the number of children you had in 2020 
  
  df <- df %>% 
    mutate(childnum_2020 = case_when(
      cf20m454 == 2 ~ "no child",
      cf20m455 == 1 ~ "one child",
      cf20m455 >= 2 & cf20m455 <=5 ~ "two children or more"))
  
  # Variable: People that want to have children should get married. Higher scores indicate higher degrees of agreement
  
  df <- df %>% 
    mutate(child_value = cv20l125)
  
  # Variable: Do you think you will have [more] children in the future?
  
  df <- df %>%
    mutate(childintention_2020 = case_when(
      cf20m128 == 1 ~ "Yes",
      cf20m128 == 2 ~ "No",
      cf20m128 == 3 ~ "I don't know"))
  
  # Variable: Do you live together with this partner? Flora: this variable is not significant, so I didn't add it in the model
  
  df <- df %>%
    mutate(livetogether = case_when(
      cf20m025 == 1 ~ "1. live together",
      cf20m025 == 2|cf20m024 == 2 ~ "0. not live together"
    ))
  
  # turning to -99 for now. In the future we will need to create dummies 
  df <- df %>%
    mutate_all(~ ifelse(is.na(.), -99, .))
  
  return(df)
}

predict_outcomes <- function(df, background_df = NULL, model_path = "./model.rds"){
  # Generate predictions using the saved model and the input dataframe.
    
  # The predict_outcomes function accepts a dataframe as an argument
  # and returns a new dataframe with two columns: nomem_encr and
  # prediction. The nomem_encr column in the new dataframe replicates the
  # corresponding column from the input dataframe The prediction
  # column contains predictions for each corresponding nomem_encr. Each
  # prediction is represented as a binary value: '0' indicates that the
  # individual did not have a child during 2021-2023, while '1' implies that
  # they did.
  
  # Parameters:
  # df (dataframe): The data dataframe for which predictions are to be made.
  # background_df (dataframe): The background data dataframe for which predictions are to be made.
  # model_path (str): The path to the saved model file (which is the output of training.R).

  # Returns:
  # dataframe: A dataframe containing the identifiers and their corresponding predictions.
  
  ## This script contains a bare minimum working example
  if( !("nomem_encr" %in% colnames(df)) ) {
    warning("The identifier variable 'nomem_encr' should be in the dataset")
  }
  
  # Load the model
  model <- readRDS(model_path)
    
  # Preprocess the fake / holdout data
  df <- clean_df(df, background_df)

  # Exclude the variable nomem_encr if this variable is NOT in your model
  vars_without_id <- colnames(df)[colnames(df) != "nomem_encr"]
  
  # Generate predictions from model
  #predictions <- predict(model, 
  #                       subset(df, select = vars_without_id), 
  #                       type = "response") 
  
  predictions <- predict(model, 
                         subset(df, select = vars_without_id)) 
  
  # Create predictions that should be 0s and 1s rather than, e.g., probabilities
  #predictions <- ifelse(predictions > 0.5, 1, 0)  
  
  # Output file should be data.frame with two columns, nomem_encr and predictions
  df_predict <- data.frame("nomem_encr" = df[ , "nomem_encr" ], "prediction" = predictions)
  # Force columnnames (overrides names that may be given by `predict`)
  names(df_predict) <- c("nomem_encr", "prediction") 
  
  # Return only dataset with predictions and identifier
  return(df_predict )
}
