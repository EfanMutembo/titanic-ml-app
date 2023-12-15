from re import X
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd


with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)





#preprocess the inputs ready for the model
def preprocess(pclass, age, sibsp, parch, fare, sex, embarked):

    """
    This function processes the data to what the model is expecting
    """

    # Convert data types
    pclass = int(pclass)
    age = int(age)
    sibsp = int(sibsp)
    parch = int(parch)
    fare = int(fare)

    # Create solo travel feature 
    solo_travel = 1 if sibsp == 0 and parch == 0 else 0

    #Create Data Frame
    row = np.array([pclass, age, sibsp, parch, fare, solo_travel, sex, embarked])
    X = pd.DataFrame([row], columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'solo_travel', 'Sex', 'Embarked'])

    # Create a custom mapping
    sex_mapping = {'Male': 1, 'Female': 0}

    # Apply the custom mapping to the 'Sex' column
    X['Sex'] = X['Sex'].map(sex_mapping)

    # Create a custom mapping
    embarked_mapping = {'Southampton': 1, 'Cherbourg': 2, 'Queenstown': 3}

    # Apply the custom mapping to the 'Embarked' column
    X['Embarked'] = X['Embarked'].map(embarked_mapping)
    
    return X
    
# Create function fo predict survived
def predict(pclass, age, sibsp, parch, fare, sex, embarked,name):

    # preprocess the data before running model
    processesd_data = preprocess(pclass, age, sibsp, parch, fare, sex, embarked)
    
    prediction = model.predict(processesd_data)[0]

    #get probability of survival or death

    proba = model.predict_proba(processesd_data)[0]

    return prediction,proba

#Web app ui and main page
def main_page():

    st.title('Would you survive the Titanic? :ship:')
    # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    pclass = st.selectbox("Choose class", [1,2,3])
    name  = st.text_input("Input Name", 'John Smith')
    sex = st.radio("Choose Sex", ('Male','Female'))
    age = st.number_input("Choose age",0,100)
    sibsp = st.slider("Choose Siblings Onboard",0,10)
    parch = st.slider("Choose Parents Onboard",0,2)
    fare = st.slider("Input Fare Price", 0,1000)
    embarked = st.radio("Select Embarking Port",('Southampton','Cherbourg','Queenstown'))

    #column order fo the model
    column = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'solo_travel', 'sex','embarked']

    if st.button('Predict'):

        #on click of predict button run predict function and initalize second page
        prediction,proba = predict(pclass, age, sibsp, parch, fare, sex, embarked,name)
        second_page(prediction,proba,name)


def second_page(prediction,proba,name):


    st.title('Your Prediction')

    proba_positive = 100*round(proba[1],2)
    if prediction == 1:
        st.subheader(f"{name} You :sparkles:Survived:sparkles: with {proba_positive}% chance of survival")
        #add fun animation
        st.image("https://media.giphy.com/media/LY1DH1AMbG0tq/giphy.gif", caption='Survived', use_column_width=True)

    else:
        st.subheader(f"{name} You Died :skull: with {proba_positive}% chance of survival")
        #add fun animation
        st.image("https://media.giphy.com/media/ZiHgApcM5ij1S/giphy.gif", caption='Died', use_column_width=True)

        # Explanation and Kaggle link
    st.markdown("## How it Works")
    st.write("I created a machine learning model to predict your survival on the Titanic.")
    st.write("The model takes into account various features such as class, age, and more.")
    st.write("To explore my model and analysis, visit the following link!:")
    st.write("[Titanic: Classification Model](https://www.kaggle.com/code/efaniorimutembo/titanic-classification-model)")




# Display the appropriate page based on the value of session_state.page
if st.session_state.get("page", "main") == "main":
    main_page()
elif st.session_state.get("page") == "second":
    second_page()
