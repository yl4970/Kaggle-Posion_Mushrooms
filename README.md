# Scripts for Binary Classification of Poisonous Mushrooms
## Playground Series - Season 4, Episode 8

=============== What this repo contains ===============
#### EDA.py contains funcs that were helpful during the EDA; 
#### Some were stepping stones to the final data-cleaning strategy
  - graph funcs to view the skewness and distribution of data
  - tried IQR for outlier handling but dropped for restricting the model's ability to train for outliers properly
  - tried to reference UCI (original dataset) rigidly (i.e. label categorical variables that are not listed in the reference dictionary as "others") but the result was not as expected since test dataset includes these noises

#### config.py contains flags to maneuver the script easily or constants to keep the script consistent

#### train_prep.py contains essential/final optimal steps to clean the dataset for training

#### optuna_xgb.py trains XGBoost and hyperparameter tuning with Optuna to search for best params
