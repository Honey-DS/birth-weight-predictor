from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleaned_data(form_data):
    Gestational_Days = float(form_data['Gestational.Days'])
    Maternal_Age = float(form_data['Maternal.Age'])
    Maternal_Height = float(form_data['Maternal.Height'])  # fixed here
    Maternal_Pregnancy_Weight = float(form_data['Maternal.Pregnancy.Weight'])
    Maternal_Smoker = bool(form_data['Maternal.Smoker'])

    cleaned_data = {
        "Gestational.Days": [Gestational_Days],
        "Maternal.Age": [Maternal_Age],
        "Maternal.Height": [Maternal_Height],  
        "Maternal.Pregnancy.Weight": [Maternal_Pregnancy_Weight],
        "Maternal.Smoker": [Maternal_Smoker]
    }

    return cleaned_data

# Load model once at startup
with open("model/model.pkl", 'rb') as obj:
    model = pickle.load(obj)
    
    @app.route('/', methods = ['GET'])
    def home():
        return render_template("index.html")

# Define an endpoint
@app.route('/predict', methods=['POST'])
def get_prediction():
    baby_data_form = request.form
    baby_data_cleaned = get_cleaned_data(  baby_data_form)

    # Convert into data frame
    baby_df = pd.DataFrame(baby_data_cleaned)

    # Make prediction
    predictions = model.predict(baby_df)
    prediction = round(float(predictions[0]), 2)

    # Return JSON response
    response = { "Prediction": prediction }
    return render_template("index.html", prediction=prediction)
    

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
