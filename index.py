# -*- coding: utf-8 -*-
import os
import pickle
from joblib import load
from app import app, server
from routes import predict
from api.credit_card_fraud import data_preparation
from utils.serialized_model import load_fraud_model


if __name__ == '__main__':
    print("Cargando modelo")
    model = load(f'./modelo/fraud_model.pkl')
    app.run(port=3000, debug=True)