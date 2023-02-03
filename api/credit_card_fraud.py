# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import pycircular as pc


# data = {
#     "ID_USER": 0,
# 	"genero": "F",
# 	"monto": "608,3456335",
# 	"fecha": "21\\/01\\/2020",
# 	"hora": 20,
# 	"dispositivo": "{'model': 2020; 'device_score': 3; 'os': 'ANDROID'}",
# 	"establecimiento": "Super",
# 	"ciudad": "Merida",
# 	"tipo_tc": "F\\u00c3\\u00adsica",
# 	"linea_tc": 71000,
# 	"interes_tc": 51,
# 	"status_txn": "Aceptada",
# 	"is_prime": "False",
# 	"dcto": "60,83456335",
# 	"cashback": "5,475110702"
# }


def data_preparation(data:dict) -> pd.DataFrame:
    try:
        df  = pd.DataFrame([[key, data[key]] for key in data.keys()])
        df = df.rename(columns={0:'campo', 1:'valor'})
        df = df.set_index('campo')
        df.index.name = ''
        df = df.T
        df = df.reset_index(drop=True)
        df.columns = df.columns.str.lower()

        # Ajustar formato fecha
        df.fecha = df.fecha.str.replace(r"\\", '',)
        df.fecha = pd.to_datetime(df.fecha, dayfirst=True)

        # Ajustar valores decimales
        df.monto = df.monto.str.replace(',', '.').astype(float)
        df.dcto = df.dcto.str.replace(',', '.').astype(float)
        df.cashback = df.cashback.str.replace(',', '.').astype(float)
        df.linea_tc = df.linea_tc.astype(float)

        # Tipo tarjeta
        text_val = df.tipo_tc.str.contains(r'^[Ff]', regex=True).values[0]
        if text_val:
            df.tipo_tc = 'Fisica'

        # Genero
        genre_val = df.genero.isin(['M', 'F']).values[0]
        if ~genre_val:
            df.genero = np.nan

        # Dispositivo
        df.dispositivo = df.dispositivo.str.replace(';', ',')
        df.dispositivo = df.dispositivo.str.replace("'", "\"")
        df[['device_model', 
            'device_score', 
            'device_os']] = (pd.json_normalize([json.loads(d) 
                                                for d in df.dispositivo]))

        device_os_val = df.device_os.isin(['%%', ',']).values[0]
        if device_os_val:
            df.device_os.genero = np.nan

        # Convertir variables object a category
        cat_cols = ['genero', 'device_score',
                    'device_os', 'establecimiento', 
                    'ciudad', 'tipo_tc','status_txn']
        df[cat_cols] = df[cat_cols].astype('category')

        # Converitr is_prime a int 
        df.is_prime = 1 if df.is_prime.values[0].lower() == 'true' else 0
        
        df['fecha_hora'] = [
                            datetime.strptime(f"{df.fecha[row].strftime('%Y-%m-%d')} {df.hora[row]}:00:00",
                                            '%Y-%m-%d %H:%M:%S')
                            for row in range(0, len(df))
                            ]

        # Tipo de dia
        df['dia_semana'] = df.fecha.dt.day_of_week
        df['dia_mes'] = df.fecha.dt.day

        # Radianes
        df['radian'] = pc.utils._date2rad(df.fecha_hora, time_segment='hour')

        # Monto neto (validar con ana)
        df['monto_neto'] = df.monto - df.cashback - df.dcto

        # Porcentaje cupo
        df['pct_cupo'] = df.monto / df.linea_tc
        
        cols_keep = ['genero', 'radian', 'device_score', 'tipo_tc',
             'status_txn', 'is_prime', 'dia_semana', 'dia_mes',
             'monto_neto', 'pct_cupo']
        df = df[cols_keep]
        
        return df
    except:
        return "Error en los datos enviados"    