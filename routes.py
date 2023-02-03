import os
import pandas as pd
import numpy as np
import pickle
import json
import traceback
from joblib import load
from app import app
from flask import request, jsonify
from api.credit_card_fraud import data_preparation


def scaler_data(num_data, scaler_choice):
    predictors_param = pd.read_csv('./modelo/predictors_params.csv')

    radian_dict = json.loads(predictors_param
                            .query("parametro == 'radian'")[['mean', 'min', 'max', 'std']]
                            .T
                            .to_json()
                            )["0"]

    is_prime_dict = json.loads(predictors_param
                            .query("parametro == 'is_prime'")[['mean', 'min', 'max', 'std']]
                            .T
                            .to_json()
                            )["1"]

    dia_sem_dict = json.loads(predictors_param
                            .query("parametro == 'dia_semana'")[['mean', 'min', 'max', 'std']]
                            .T
                            .to_json()
                            )["2"]

    dia_mes_dict = json.loads(predictors_param
                            .query("parametro == 'dia_mes'")[['mean', 'min', 'max', 'std']]
                            .T
                            .to_json()
                            )["3"]

    monto_neto_dict = json.loads(predictors_param
                            .query("parametro == 'monto_neto'")[['mean', 'min', 'max', 'std']]
                            .T
                            .to_json()
                            )["4"]

    pct_cup_dict = json.loads(predictors_param
                            .query("parametro == 'pct_cupo'")[['mean', 'min', 'max', 'std']]
                            .T
                            .to_json()
                            )["5"]
    if scaler_choice > 0.5 :
        num_data.radian = (num_data.radian.values[0] - radian_dict['mean']) / radian_dict['std']
        num_data.is_prime = (num_data.is_prime.values[0] - is_prime_dict['mean']) / is_prime_dict['std']
        num_data.dia_semana = (num_data.dia_semana.values[0] - dia_sem_dict['mean']) / dia_sem_dict['std']
        num_data.dia_mes = (num_data.dia_mes.values[0] - dia_mes_dict['mean']) / dia_mes_dict['std']
        num_data.monto_neto = (num_data.monto_neto.values[0] - monto_neto_dict['mean']) / monto_neto_dict['std']
        num_data.pct_cupo = (num_data.pct_cupo.values[0] - pct_cup_dict['mean']) / pct_cup_dict['std']
    else:
        num_data.radian = (num_data.radian.values[0] - radian_dict['min']) / (radian_dict['max'] - radian_dict['min'])
        num_data.is_prime = (num_data.is_prime.values[0] - is_prime_dict['min']) / (is_prime_dict['max'] - is_prime_dict['min'])
        num_data.dia_semana = (num_data.dia_semana.values[0] - dia_sem_dict['min']) / (dia_sem_dict['max'] - dia_sem_dict['min'])
        num_data.dia_mes = (num_data.dia_mes.values[0] - dia_mes_dict['min']) / (dia_mes_dict['max'] - dia_mes_dict['min'])
        num_data.monto_neto = (num_data.monto_neto.values[0] - monto_neto_dict['min']) / (monto_neto_dict['max'] - monto_neto_dict['min'])
        num_data.pct_cupo = (num_data.pct_cupo.values[0] - pct_cup_dict['min']) / (pct_cup_dict['max'] - pct_cup_dict['min'])
        
    return num_data


@app.route('/fraud_predict', methods=['POST'])
def predict():
    model = load(f'./modelo/fraud_model.pkl')
    if model:
        try:
            model_params = pd.read_csv('./modelo/model_params.csv')
            scaler_choice = (model_params.loc[model_params.parametro == 'scaler_choice',
                                              'valor'].values[0])
            
            predictors_param = pd.read_csv('./modelo/predictors_params.csv')
            
            json_data = request.json
            df = data_preparation(json_data)
            # df = data_preparation(data)
            
            df1 = scaler_data(df, scaler_choice)
            
            prediction = model.predict_proba(df1)[:, 1]
            
            return jsonify({'prediction': str(prediction)})
            
        except:
            return jsonify({'trace': traceback.format_exc()})

    else:
        print("No se ha cargado el modelo")
        