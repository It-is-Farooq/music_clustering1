import streamlit as st
import pandas as pd
import joblib

# Load the model
loaded_model = joblib.load('music_genre_classifier.pkl')

class_names = ['Acoustic/Folk', 'Alt_Music', 'Blues', 'Bollywood', 'Country', 'HipHop', 'Indie', 'Instrumental','Metal', 'Pop', 'Rock']

# Title of the app
st.title("Music Genre Classification Using ML")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file for input data", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)

        # Dropdown to select a row
        selected_index = st.selectbox("Select a row for prediction:", data.index)

        # Pre-fill inputs based on the selected row or manual entry
        input_data = {}
        columns = ['Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                   'valence', 'tempo', 'duration_in min/ms', 'time_signature']

        st.write("Enter or edit the input data for prediction:")
        for col in columns:
            if selected_index is not None:
                input_data[col] = st.number_input(f"{col}:", 
                                                  value=float(data.loc[selected_index, col]), 
                                                  step=0.00001,
                                                  format="%.5f", 
                                                  key=col)
            else:
                input_data[col] = st.number_input(f"{col}:", value=0.0, step=0.00001,format="%.5f",  key=col)

        # Convert input data to DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Predict button
        if st.button("Predict"):
            prediction = loaded_model.predict(input_df)

            st.write(f"Predicted Class: {class_names[prediction[0]]}")
            
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.write("Please upload a CSV file or manually enter data for prediction.")

