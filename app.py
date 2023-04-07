import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Title: Binary Classification Web App")
    st.sidebar.title("Sidebar: Binary Classification Web App")
    st.markdown("Are your mushroom edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushroom edible or poisonous? 🍄")

    @st.cache(persist=True)
    def load_data(file_path):
        data = pd.read_csv(file_path)
        # preprocess data here
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        X = df.drop(columns=['type'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7042023)

        return X_train, X_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'confusion matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'roc curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, X_test, y_test)

        if 'precision-recall curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()

    file_path = "mushrooms.csv"
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose classifier:")
    classifier = st.sidebar.selectbox("Classifier", ("SVM", "Logistic Regression", "Random Forest"))

    if classifier == 'SVM':
        st.sidebar.subheader("Model Hyperparameters")
        # C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, steps=0.01, key='C')
        C = st.sidebar.number_input("C (Regularization Parameter)")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('confusion matrix', 'roc curve', 'precision-recall curve'))

    if st.sidebar.button("Classify", key='classify'):
        st.subheader("SVM Results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)    
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)  # display the data in the Streamlit app
        st.subheader("Training and test sets:")
        st.write(len(X_train), len(X_test), X_train, y_train)
        
        

if __name__ == '__main__':
    main()


