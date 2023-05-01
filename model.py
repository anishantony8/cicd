from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np
application = Flask(__name__)

app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Age = int(request.form.get('Age')),
            Diabetes = (request.form.get('Diabetes')),
            BloodPressureProblems = (request.form.get('BloodPressureProblems')),
            AnyTransplants = request.form.get('AnyTransplants'),
            AnyChronicDiseases = (request.form.get('AnyChronicDiseases')),
            Height = float(request.form.get('Height')),
            Weight = float(request.form.get('Weight')),
            KnownAllergies = request.form.get('KnownAllergies'),
            HistoryOfCancerInFamily= request.form.get('HistoryOfCancerInFamily'),
            NumberOfMajorSurgeries = request.form.get('NumberOfMajorSurgeries')
        )

        pred_df = data.get_data_as_data_frame()
        
        print(pred_df)

        predict_pipeline = PredictPipeline()
        np_pred = np.array(pred_df)
        pred = predict_pipeline.predict(np_pred)
        results = round(pred[0],2)
        return render_template('index.html',results=results,pred_df = pred_df)
    
@app.route('/predictAPI',methods=['POST'])
@cross_origin()
def predict_api():
    if request.method=='POST':
        data = CustomData(
            Age = int(request.json['Age']),
            Diabetes = (request.json['Diabetes']),
            BloodPressureProblems = (request.json['BloodPressureProblems']),
            AnyTransplants = (request.json['AnyTransplants']),
            AnyChronicDiseases = (request.json['AnyChronicDiseases']),
            Height = float(request.json['Height']),
            Weight = float(request.json['Weight']),
            KnownAllergies = request.json['KnownAllergies'],
            HistoryOfCancerInFamily = request.json['HistoryOfCancerInFamily'],
            NumberOfMajorSurgeries = request.json['NumberOfMajorSurgeries']
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        dct = {'Premium':round(pred[0],2)}
        return jsonify(dct)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
