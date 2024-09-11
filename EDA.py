import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from train_prep import *

def var_top_val (df, col_name, top_rk=10):
    object_col = list(df[[col_name]].select_dtypes(include=['object']).columns)
    output = pd.DataFrame()
    for i in object_col:
        col = df[i].value_counts().to_frame().sort_values(by=['count'],ascending=False).head(top_rk)
        col = col.reset_index()
        col['count'] = round(col['count'],2)
    output = pd.concat([output, col], ignore_index=True, axis=1)
    output.columns = ['val', 'count']
    return output


def var_unique_val (df):
    object_col = list(df.select_dtypes(include=['object']).columns)
    output = pd.DataFrame([df[i].unique() for i in object_col]).T
    output.columns = object_col
    return output


def arrays_to_df (arrays, col_names, fill_with=np.nan):
    max_length = max(len(i) for i in arrays)
    output = pd.DataFrame([i + [fill_with]*(max_length-len(i)) for i in arrays]).T
    output.columns = col_names
    return output


def get_replaced_dict(replaces_by_object, col_name, object_cols):
    print(f'{col_name}:\n {replaces_by_object[object_cols.index(col_name)]}')


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


## Visualize the categorical variable distribution
def cat_barplot(df):

    cat_col = list(df.select_dtypes(include=['object']).columns)
    nrows = int(np.sqrt(len(df[cat_col].columns)))
    nelements = len(cat_col)
    ncols = int(nelements/nrows)+1

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, tight_layout=True, figsize=(20, nrows*20/ncols), dpi=100)
    axs = axs.flatten()

    for ind, i in enumerate(cat_col):
        sns.barplot(x=df[i].value_counts().index,
                    y=df[i].value_counts().values,
                    ax=axs[ind],
                    palette='Set2')

    for i in range(nrows*ncols-len(cat_col)):
        fig.delaxes(axs[len(cat_col)-i])

    plt.show()


## Visualize the float variable distribution in boxplots
def num_boxplot (df):

    num_col = list(df.select_dtypes(include=['float','float32','float64']).columns)
    nrows = int(np.sqrt(len(df[num_col].columns)))
    nelements = len(num_col)
    ncols = int(nelements/nrows)+1

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, tight_layout=True, figsize=(24, nrows*24/ncols), dpi=100)
    axs = axs.flatten()

    for ind, i in enumerate(num_col):
        sns.boxplot(data=df, x=df[i], y='class', ax=axs[ind])

    for i in range(nrows*ncols-nelements):
        fig.delaxes(axs[nelements-i])

    plt.show()


## Understand the distribution of numeric variables
def num_pdf(df):

    num_col = list(df.select_dtypes(include=['float','float32','float64']).columns)
    nrows = int(np.sqrt(len(df[num_col].columns)))
    nelements = len(num_col)
    ncols = int(nelements/nrows)+1

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, tight_layout=True, figsize=(20, nrows*20/ncols), dpi=100)
    axs = axs.flatten()

    for ind, i in enumerate(num_col):
        sns.histplot(df[i], kde=True, ax=axs[ind], element='step', stat='density')

    for i in range(nrows*ncols-nelements):
        fig.delaxes(axs[nelements-i])

    plt.show()


def main():

    print("generating barplot for categorical variables >>")
    object_barplot(train_df)

    print("generating boxplot for numeric variables >>")
    num_boxplot(train_df)

    print("generating probability density for numeric variables (post-encoding) >>")
    num_pdf(X)



if __name__ == '__main__':
    main()
