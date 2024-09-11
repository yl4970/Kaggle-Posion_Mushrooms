import numpy as np
import pandas as pd
import time
import warnings
warnings.simplefilter(action='ignore')

import category_encoders as ce # encoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from config import *
from EDA import *

train_data = pd.read_csv('/Users/liuyerong/Desktop/Quant_Trading/Kaggle/08-2024/poison_mushroom/train.csv')
test_data = pd.read_csv('/Users/liuyerong/Desktop/Quant_Trading/Kaggle/08-2024/poison_mushroom/test.csv')

def iden_missing_val(df, del_criteria = DEL_CRITERIA):
    
    float_col = list(df.select_dtypes(include=['float','float32','float64']).columns)
    object_col = list(df.select_dtypes(include=['object']).columns)

    col_del_1 = [i for i in df.columns if df[i].isnull().sum()/len(df)>=del_criteria]
    col_complete = [i for i in df.columns if i not in col_del_1]
    
    print(f"Variables with missing value more than {del_criteria*100}% is \n {col_del_1}" )

    float_col_1 = [i for i in float_col if i in col_complete]
    object_col_1 = [i for i in object_col if i in col_complete]

    return float_col_1, object_col_1


def clean_cat_by_step(df, col_name, replace_dict=REPLACE_DICT_FLAG):

    ## step 0: find all invalid entries per uci reference
    uniques = list(df[col_name].unique())
    var_val = var_top_val (df, col_name, top_rk=len(df[col_name].unique()))

    ## step 1: clean np.nan into unknown to make processing easier
    nan_replace = {np.nan: 'unknown'}
    nan_replacer = nan_replace.get
    uniques = [nan_replacer(i,i) for i in uniques]

    ## step 2: replace numeric values in the categorical variable
    def find_numerics(list):    
        for i in list:
            try:
                yield float(i)
            except ValueError:
                pass
    numerics_float = list(find_numerics(uniques))
    numerics_str =list(str(i) for i in numerics_float)
    numerics_str += [str(int(i)) for i in numerics_float if str(i) not in uniques]
    numeric_replace = dict(zip(numerics_str, ['error']*len(numerics_str)))

    ## step 3: identify single alphabet entries that are not included in uci reference
    singles = [i for i in uniques if i not in numerics_str and len(i)==1]
    
    ## step 4: identify entries that were recorded incorrectly potentially due to formatting
    formatted = []
    corrected = []
    for i in uniques:
        for j in singles:
            if ' '+j in i:
                formatted += [i]
                corrected += [j]
    format_replace = dict(zip(formatted, corrected))

    ## step 5: all other undefined long strings
    replaced = ['unknown', 'error']
    undefined_str = [i for i in uniques if i not in numerics_str+singles+formatted+replaced 
                     and isinstance(i, str)]
    undefined_str_replace = dict(zip(undefined_str, ['error']*len(undefined_str)))

    final_replace_dict = {**nan_replace, **numeric_replace, **format_replace, **undefined_str_replace}
    df[col_name] = df[col_name].replace(final_replace_dict, inplace=False)

    if replace_dict==True:
        return df[col_name], final_replace_dict
    else: 
        return df[col_name]


def iden_outliers_IQR(df):
    # Calculate quantiles and IQR
    float64_col = list(df.select_dtypes(include='float64'))
    df[float64_col] = df[float64_col].astype('float32')
    float_col = list(df.select_dtypes(include=['float','float32','float64']).columns)
    
    q1 = df[float_col].quantile(0.25)
    q3 = df[float_col].quantile(0.75)
    iqr = q3 - q1

    # Calculate bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    output = df.copy()
    for i in float_col:
        output[i] = df[i].apply(lambda x: np.where((x >= lower_bound[i]) & (x <= upper_bound[i]), x, np.nan))
    output = output.dropna()
    return output


def normalize(df):
  qt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
  return qt.fit_transform(df)


def main(df):
    print("<<loading data preprocessing>>")
    start_time = time.time()

    ## 1. define valid numeric and categorical variables
    num_col, cat_col = iden_missing_val(train_data)
    df = train_data[num_col + cat_col]
    cat_col.pop(cat_col.index('class')) # remove target variable from categorical features

    ## 2. clean categorical data for unwanted string inputs, numeric values, and np.nan
    for ind, i in enumerate(cat_col):
        df[i] = clean_cat_by_step(train_data, i)

    ## 3. fill np.nan in numercial variables with median
    df[num_col] = df[num_col].fillna(df[num_col].median())
    
    ## train_data[num_col].skew() # check the skewness of numerical variables - data are very skewed
    ## 4. normalize numerical variables
    df[num_col] = normalize(df[num_col])

    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))
    return df


if __name__ == '__main__':
    main()
 
train_df = main(train_data)
## prepare feature data and target data
X = train_df.drop(columns=['class']) # prepare features and target for target encoding
y = train_df['class']
y = y.replace({'e':0, 'p':1}) # transform target per submission format

## 5. encode categorical variables
encoder = ce.CatBoostEncoder()
encoder.fit(X, y)
X_enc = encoder.transform(X)

## prepare data for training
# X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=RANDOM_STATE)

## prepare data for testing
# test_df = main(test_data)