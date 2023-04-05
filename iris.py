import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
import warnings
warnings.filterwarnings('ignore')




#Load the iris dataset using the "datasets.load_iris()" function and assign the data and target variables to "X" and "Y", respectively.

data = pd.read_csv(r'C:\Users\USER\Documents\Datascience\newproject\churners\iris.data.csv')


data.head()

data.isnull().sum().sort_values(ascending = False)


numerical = data.select_dtypes(exclude  = 'object') # ............. identify the numerical variables
categorical = data.select_dtypes(include = 'object') # ............ Identify the categorical variables

for i in numerical:
  data[i].fillna(data[i].mean(), inplace = True)

for i in categorical:
  data[i].fillna(data[i].mode()[0], inplace = True)
  
data.isnull().sum().sort_values(ascending = False).head(3)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in data.columns:
  data[i] = encoder.fit_transform(data[i])

data.head()

# Define x and y
# Define x and y
x = data.drop(['Iris-setosa'], axis = 1)
y = data['Iris-setosa']


# Split data into train and test
# Notice that i pass in 'y', and not y_trans.
# coz y is still in words, so my prediction will be straight forward
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


rf_model = RandomForestClassifier()
# Model Creation
rf_model.fit(x_train, y_train)

# Predict the test data for checking accuracy
prediction = rf_model.predict(x_test)

# Save Model
import joblib
joblib.dump(rf_model, 'iris_model.pkl')

# FROM HERE WE BEGIN THE IMPLEMENTATION FOR STREAMLIT.
st.header('IRIS MODEL DEPLOYMENT')
user_name = st.text_input('Register User')

if(st.button('SUBMIT')):
    st.text(f"You are welcome {user_name}. Enjoy your usage")

st.write(data)

from PIL import Image
# image = Image.open(r'images\use.png')
# st.sidebar.image(image)


st.sidebar.subheader(f"Hey {user_name}")
metric = st.sidebar.radio('How do you want your feature input?\n \n \n', ('slider', 'direct input'))


if metric == 'slider':
   sepal_length = st.sidebar.slider('SEPAL LENGTH', 0.0, 9.0, (5.0))

   sepal_width = st.sidebar.slider('SEPAL WIDTH', 0.0, 4.5, (2.5))

   petal_length = st.sidebar.slider('PETAL LENGTH', 0.0, 8.0, (4.5))

   petal_width = st.sidebar.slider('PETAL WIDTH', 0.0, 3.0, (1.5))
else:
    sepal_length = st.sidebar.number_input('SEPAL LENGTH')
    sepal_width = st.sidebar.number_input('SEPAL WIDTH')
    petal_length = st.sidebar.number_input('PETAL LENGTH')
    petal_width = st.sidebar.number_input('PETAL WIDTH')


input_values = [[sepal_length, sepal_width, petal_length, petal_width]]


# Modelling
# import the model
model = joblib.load(open('iris_model.pkl', 'rb'))
pred = model.predict(input_values)


# fig, ax = plt.subplots()
# ax.scatter(y_pred, y_test)
# st.pyplot(fig)


if pred == 0:
    st.success('The Flower is an Iris-setosa')
    # setosa = Image.open('images\Irissetosa1.JPG')
    # st.image(setosa, caption = 'Iris-setosa', width = 400)
elif pred == 1:
    st.success('The Flower is an Iris-versicolor ')
    # versicolor = Image.open('images\irisversicolor.JPG')
    # st.image(versicolor, caption = 'Iris-versicolor', width = 400)
else:
    st.success('The Flower is an Iris-virginica ')
    # virginica = Image.open('images\Iris-virginica.JPG')
    # st.image(virginica, caption = 'Iris-virginica', width = 400 )