import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model from disk
def load_model():
    with open('naive_bayes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Define the feature input function
def user_input_features():
    N = st.number_input('Nitrogen', min_value=0.0, format="%.2f")
    P = st.number_input('Phosphorus', min_value=0.0, format="%.2f")
    K = st.number_input('Potassium', min_value=0.0, format="%.2f")
    temperature = st.number_input('Temperature', min_value=0.0, format="%.2f")
    humidity = st.number_input('Humidity', min_value=0.0, format="%.2f")
    ph = st.number_input('pH', min_value=0.0, format="%.2f")
    rainfall = st.number_input('Rainfall', min_value=0.0, format="%.2f")
    
    data = {'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall}
    features = pd.DataFrame(data, index=[0])
    return features

# Main function to display the Streamlit app
def main():
    st.title('Crop Recommendation System')

    st.write("This app predicts the best crop to plant based on the input conditions.")
    df = user_input_features()

    if st.button('Predict'):
        prediction = model.predict(df)
        st.write(f'The recommended crop is {prediction[0]}')

if __name__ == '__main__':
    main()
