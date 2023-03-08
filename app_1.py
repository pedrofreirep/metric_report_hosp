import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as cv
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
# from skopt.space import Real, Categorical, Integer

from sklearn.metrics import (roc_curve, precision_recall_curve,
                             average_precision_score, classification_report,
                             roc_auc_score, recall_score, precision_score,
                             accuracy_score, f1_score, make_scorer)

from sklearn.model_selection import cross_validate

import joblib

# from skopt import BayesSearchCV

import datetime

import io
import os

st.title('🏥 Binary Prediction for Hospitalizations - Metric Report')

st.markdown("Welcome to the **Binary Prediction for Hospitalizations - Metric Report**. First, **upload your data** at the sidebar on your left, as many datasets you want. Sencond, **build your analysis** by selecting the files you want to compare and the metrics you want to analyze. Right after that, the aggregate metrics should be presented automaticaly. We use F1, Precision, and Recall to evaluate our predictions for a specific class. The margin of error is usually used for regressions.")
st.warning("**Important to have in mind!** \n\n Those are only the aggregate metrics, just one of the steps from the first level of our AI development sprint. In order to deploy the model that best fits our product, we should go further and analyze metrics per class aligned with the problem, document error patterns, search for local and global explanations, run what-if analysis, and perform other tasks to help our understanding of the model. \n\n You can find our complete AI development process [here](https://www.notion.so/blueai/aecd7f9ce3fb4aa38f24ee199ac1cacb?v=fc291161a84b42ef8f0084bbd9ef3991&pvs=4).", icon="⚠️")

st.sidebar.write('# 1. Choose your data')
st.sidebar.info('Your csv must cointain the following columns: **id_pessoa, pred_year_target and prediction.**', icon="ℹ️")
st.sidebar.markdown('\n\n')
data_source = st.sidebar.radio(
    "Choose the source of your data:",
    ('Upload', 'Pre-saved'))
if data_source == 'Upload':
    st.sidebar.markdown('\n\n')
    path = st.sidebar.file_uploader("Choose your files:", accept_multiple_files=True)
else:
    st.success('Pre-saved files is coming in future versions. For now, keep using the upload feature to analyze metrics.', icon="🤖")
    '\n\n'
    '\n\n'
    '\n\n'
    '\n\n'
    '\n\n'
    '\n\n'
    # path = "/files"
    # path = "/Users/pedro/Documents/Blue/apps/metric_report/files"

    # files = os.listdir(path)
    # files_list = [os.path.splitext(file)[0].lower() for file in files]

    # filtered_files = [file for file in files_list if files_list]
    # sel_filtered_file = st.selectbox("Choose file:", filtered_files)

df_list = {}
y_train_list = {}
predict_prob_list = {}

file_list = []

for i in range(len(path)):
    df_list[path[i].name] = pd.read_csv(path[i])
        # df_list.append(pd.read_csv(path[i]))
    
    df_list[path[i].name]['id_pessoa'] = df_list[path[i].name]['id_pessoa'].astype(str)
    df_list[path[i].name]['pred_year_target'] = df_list[path[i].name]['pred_year_target'].astype(int)
    df_list[path[i].name]['prediction'] = df_list[path[i].name]['prediction'].astype(float)

    y_train_list[path[i].name] = df_list[path[i].name][['id_pessoa','pred_year_target']]
    predict_prob_list[path[i].name] = df_list[path[i].name][['id_pessoa','prediction']]

    file_list.append(path[i].name)
    # st.write(path[i].name)


'\n'
'\n'
if len(path) > 0:
    st.write('#### 2. Build your analysis')
    file_options = st.multiselect(
        'Select the files you want to analyze:',
        file_list,
        file_list[0])

    metrics_options = st.multiselect(
        'Select the metrics you want to analyze:',
        ['Precision & Recall', 'ROC Curve', 'P/R Curve'],
        'Precision & Recall')
    
    def roc_func(true, predict_prob, model_name=''):
        # ROC CURVE
        fpr, tpr, _ = roc_curve(true['pred_year_target'], predict_prob['prediction'])
        roc_auc     = roc_auc_score(true['pred_year_target'], predict_prob['prediction'])
        
        fig, ax = plt.subplots()

        ax.plot(fpr, tpr, color='darkorange', lw=0.5,
                label='ROC Curve Train (area = %0.3f)' % roc_auc)
        ax.plot([0, 1], [0, 1], lw=2, linestyle='--',
                label='Random Guess (area = 0.5)')   
            
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_title(model_name + 'Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    def p_r_curve(true, predict_prob, model_name=''):
        # P/R
        precision, recall, _ = precision_recall_curve(true['pred_year_target'], predict_prob['prediction'])

        average_precision = average_precision_score(true['pred_year_target'], predict_prob['prediction'])
        
        fig_2, ax = plt.subplots()

        ax.step(recall, precision, color='C0', alpha=0.8, where='post',
                label='Precision-Recal Curve')
        ax.fill_between(recall, precision, alpha=0.2, color='C0', step='post')
        ax.axhline(average_precision, color='r', linestyle='--',
                label=f'Avg Precision = {average_precision:.2f}')        

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([-0.02, 1.0])
        ax.set_title(model_name + 'Precision-Recall Curve')
        ax.legend(loc="lower left")
        st.pyplot(fig_2)

    if 'Precision & Recall' in metrics_options:
        '\n'
        '\n'
        st.write('##### Aggregate Metrics')

        '\n'

        cutoff_perc = st.slider(
            "Select your model's cutoff value:",
            0.0, 1.0, (0.5))
        
        def aggregate_metrics(true, predict_prob, model_name='', cutoff=cutoff_perc):
            st.text('Model Report:\n    '+classification_report(true['pred_year_target'], np.where(predict_prob['prediction'] > cutoff, 1, 0)))

        for i in range(len(file_options)):
            '\n'
            st.write("###### _File name: "  + path[i].name + "_")
            y_train = y_train_list[path[i].name]
            predict_prob = predict_prob_list[path[i].name]
            aggregate_metrics(y_train, predict_prob)

    if 'ROC Curve' in metrics_options:
        '\n'
        '\n'
        st.write('##### ROC Curve')
        for i in range(len(file_options)):
            '\n'
            st.write("###### _File name: "  + path[i].name + "_")
            y_train = y_train_list[path[i].name]
            predict_prob = predict_prob_list[path[i].name]
            roc_func(y_train, predict_prob)

    if 'P/R Curve' in metrics_options:
        '\n'
        '\n'
        st.write('##### Precision/Recall Cruve')
        for i in range(len(file_options)):
            '\n'
            st.write("###### _File name: "  + path[i].name + "_")
            y_train = y_train_list[path[i].name]
            predict_prob = predict_prob_list[path[i].name]
            p_r_curve(y_train, predict_prob)

else:
    pass
