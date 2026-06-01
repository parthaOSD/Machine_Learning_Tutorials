import gradio as gr
import pickle
import numpy as np

# Load trained model
with open("linear_model.pkl", "rb") as file:
    model = pickle.load(file)

# Prediction function
def predict_marks(hours):
    hours = np.array([[hours]])
    prediction = model.predict(hours)
    return float(prediction[0])

# Gradio UI
interface = gr.Interface(
    fn=predict_marks,
    inputs=gr.Number(label="Study Hours"),
    outputs=gr.Number(label="Predicted Result"),
    title="Student Result Predictor",
    description="Enter study hours to predict exam result using Linear Regression"
)

interface.launch() 