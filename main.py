import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Title
st.title("🌸 Iris Classification")
st.markdown("<h3 style='text-align: center; background-color: green; color: white; padding: 10px;'>Iris Classification</h3>", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("iris.csv")
    return df

df = load_data()

# Encode target
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# Split dataset
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar
st.sidebar.markdown("### Model Used")
st.sidebar.info("Linear Regression")

# Input boxes
sepal_length = st.slider("Select Sepal Length", 0.0, 10.0, 0.0)
sepal_width = st.slider("Select Sepal Width", 0.0, 10.0, 0.0)
petal_length = st.slider("Select Petal Length", 0.0, 10.0, 0.0)
petal_width = st.slider("Select Petal Width", 0.0, 10.0, 0.0)

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred = model.predict(input_data)
pred_rounded = round(pred[0])

# Clamp prediction within valid index
pred_rounded = max(0, min(pred_rounded, 2))

# Predict and display
if st.button("Classify"):
    predicted_species = le.inverse_transform([pred_rounded])[0]
    st.success(predicted_species)