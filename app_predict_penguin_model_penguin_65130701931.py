

import pickle
import pandas as pd
import streamlit as st

# Load the model and encoders from the pickle file
with open('model_penguin_65130701931.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app title
st.title('Penguin Species Prediction App')

# User input for prediction
island = st.selectbox('Island', island_encoder.classes_)
culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0, step=0.1)
culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0, step=0.1)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, step=0.1)
body_mass_g = st.number_input('Body Mass (g)', min_value=0.0, step=0.1)
sex = st.selectbox('Sex', sex_encoder.classes_)

# Create new data for prediction
if st.button('Predict'):
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })

    # Encode categorical features
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])

    # Make predictions
    y_pred_new = model.predict(x_new)

    # Inverse transform the prediction to get the original label
    result = species_encoder.inverse_transform(y_pred_new)

    # Display the result
    st.success(f'Predicted Species: {result[0]}')


