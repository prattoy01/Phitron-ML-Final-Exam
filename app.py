import gradio as gr
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained pipeline
pipeline = joblib.load('insurance_rf_pipeline.pkl')

# 2. Define the prediction function
def predict_insurance_charges(age, sex, bmi, children, smoker, region):
    
    # Re-apply the custom logic used in training
    def classify_bmi(b):
        if b < 18.5: return 'underweight'
        elif 18.5 <= b < 25: return 'normal'
        elif 25 <= b < 30: return 'overweight'
        else: return 'obese'
    
    weight_status = classify_bmi(bmi)
    
    # Prepare the input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region],
        'weight_status': [weight_status]
    })
    
    # Make prediction
    prediction = pipeline.predict(input_data)[0]
    
    # Return formatted result (Similar to the screenshot style)
    return f"Predicted Insurance Charge: ${np.round(prediction, 2):,.2f}"

# 3. Define inputs (Matching the list style in your screenshot)
inputs = [
    gr.Slider(18, 100, step=1, label="Age", value=25),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Slider(10, 60, step=0.1, label="BMI", value=22.5),
    gr.Slider(0, 10, step=1, label="Children", value=0),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
]

# 4. Launch Interface
interface = gr.Interface(
    fn=predict_insurance_charges,
    inputs=inputs,
    outputs="text",
    title="Insurance Cost Prediction System",
    description="Enter the details below to estimate medical insurance costs."
)

if __name__ == "__main__":
    interface.launch()