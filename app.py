import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from os import system
from graphviz import Source
from sklearn import tree
import pandas as pd
import numpy as np



def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array
# Function to load data from CSV file
def load_data(file):
    data = pd.read_csv(file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Streamlit app
st.sidebar.markdown("# Decision Tree Classifier")

# Allow user to upload a CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
st.sidebar.markdown("Note : Upload CSV with 3 colum only (2 for inputs and 1 for output)")

# If a file is uploaded, load the data
if uploaded_file is not None:
    X, y = load_data(uploaded_file)
else:
    # Default data if no file is uploaded
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.style.use('fivethirtyeight')
# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.xlabel("Col1")
plt.ylabel("Col2")
orig = st.pyplot(fig)
st.sidebar.markdown("# Decision Tree Classifier")

criterion = st.sidebar.selectbox(
    'Criterion',
    ('gini', 'entropy')
)

splitter = st.sidebar.selectbox(
    'Splitter',
    ('best', 'random')
)

max_depth = int(st.sidebar.number_input('Max Depth'))

min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2, key=1234)

min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1, key=1235)

max_features = st.sidebar.slider('Max Features', 1, 2, 2, key=1236)

max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

# If "Run Algorithm" button is clicked
if st.sidebar.button('Run Algorithm'):
    orig.empty()

    # Rest of the code remains the same...

    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None
    # Attempt to fit the DecisionTreeClassifier
   
try:
    clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                 random_state=42, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    XX, YY, input_array = draw_meshgrid(X)
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Decision Tree  " + str(round(accuracy_score(y_test, y_pred), 2)))

    tree = export_graphviz(clf, feature_names=["Col1", "Col2"])

    st.graphviz_chart(tree)
except ValueError as e:
    # Handle the ValueError by printing an error message
    print(f"Error: {e} , errror")
  

   
    
# Data view section
st.sidebar.markdown("## Data View")
if uploaded_file is not None:
    data_view = pd.DataFrame(X)
    data_view['Label'] = y
    st.sidebar.dataframe(data_view.head())
else:
    st.sidebar.text("No uploaded file. Using default data.")

# Info section
st.sidebar.markdown("## Info")
st.sidebar.markdown("This app allows you to upload a CSV file with two feature columns (Col1 and Col2) and a label column (Label).")
st.sidebar.markdown("The Decision Tree Classifier is then trained on the data, and the results are displayed.")

# Footer


st.markdown("Developed by Om Bade")
