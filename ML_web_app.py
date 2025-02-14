import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Heading of the app 
st.write('''
 # **Machine Learning Model Comparison App**''')

# Sidebar selection for dataset
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# Sidebar selection for classifier
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('SVM', 'KNN', 'Random Forest')
)

# Function to load datasets
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y

# Load dataset
x, y = get_dataset(dataset_name)

# Display dataset info
st.write('Shape of dataset:', x.shape)
st.write('Number of classes:', len(np.unique(y)))

# Function to set classifier parameters
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('Max Depth', 2, 15)
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

# Get classifier parameters
params = add_parameter_ui(classifier_name)

# Function to create classifier
def get_classifier(classifier_name, params):
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'], random_state=1234)
    return clf

# Get classifier
clf = get_classifier(classifier_name, params)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Train the model
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc:.2f}')
# Interactive plot using PCA
pca = PCA(2)
x_projected = pca.fit_transform(x)
fig, ax = plt.subplots()
scatter = ax.scatter(x_projected[:, 0], x_projected[:, 1], c=y, alpha=0.8, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
st.pyplot(fig)