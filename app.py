import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the trained pipeline
with open("insurance_rf_pipeline.pkl", "rb") as f:
        rf_pipeline = pickle.load(f)


#  Define prediction function
def predict_insurance_charges(age, sex, bmi, children, smoker, region):
    
   
    def cata_bmi(b):
        if b < 18.5: return 'underweight'
        elif 18.5 <= b < 25: return 'normal'
        elif 25 <= b < 30: return 'overweight'
        else: return 'obese'
    
    weight_status = cata_bmi(bmi)
    
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region],
        'weight_status': [weight_status]
    })
    
    # Predict
    prediction = rf_pipeline.predict(input_data)[0]
    return f"Predicted Charge: ${np.round(prediction, 2):,.2f}"

#  Setup Inputs
inputs = [
    gr.Slider(18, 100, step=1, label="Age", value=25),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Slider(10, 60, step=0.1, label="BMI", value=22.5),
    gr.Slider(0, 10, step=1, label="Children", value=0),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
]

#  Launch
interface = gr.Interface(
    fn=predict_insurance_charges,
    inputs=inputs,
    outputs="text",
    title="Medical Insurance Predictor",
    description="Enter details to estimate costs."
)

if __name__ == "__main__":
    interface.launch()