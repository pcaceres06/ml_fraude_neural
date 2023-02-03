import pickle
import json
import pandas as pd

def save_model(filename, object):
    with open(f"{filename}", 'wb') as file:
        pickle.dump(object, file)
        
def load_fraud_model(filename):
    with open(f'{filename}', 'rb') as file:
        model = pickle.load(file)
    return model


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