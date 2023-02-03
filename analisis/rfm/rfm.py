# -*- coding: utf-8 -*-
import os
import re
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.options.display.max_columns = None
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.2f}'.format

#FILE_PATH = f'{os.getcwd()}/data/processed'
FILE_PATH = '../../data/processed/'

# Cargar datos
df = pd.read_csv(f'{FILE_PATH}/transactions.csv', 
                 sep=',',
                 parse_dates=['fecha'])

df = df[df.fraude == 0]

# Calcular el recency
recency_df = df.groupby(by='id_user', as_index=False)['fecha'].max()
max_date = recency_df.fecha.max()
recency_df['recency'] = recency_df.fecha.apply(lambda x: (max_date - x).days)

# Calcular la frecuencia
frecuency_df = df.groupby(by='id_user', as_index=False)['fecha'].count()
frecuency_df = frecuency_df.rename(columns={'fecha':'cantidad'})

# Calcular valor neto de las compra
money_df = df.groupby(by='id_user', as_index=False)['monto_neto'].sum()

# Consolidar data set 
rfm_df = recency_df.merge(frecuency_df, on='id_user')
rfm_df = rfm_df.merge(money_df, on='id_user')
rfm_df = rfm_df.drop('fecha', axis=1)

# Crear ranking de usuario
rfm_df['r_rank'] = rfm_df.recency.rank(ascending=False)
rfm_df['f_rank'] = rfm_df.cantidad.rank(ascending=False)
rfm_df['m_rank'] = rfm_df.monto_neto.rank(ascending=False)

# Normalizar los ranking
rfm_df['r_rank'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
rfm_df['f_rank'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
rfm_df['m_rank'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100

# Crear el score RFM
r_weight = 0.2
f_weight = 0.3
m_weight = 0.5

rfm_df['score'] = round(r_weight*rfm_df.r_rank + 
                        f_weight*rfm_df.f_rank + 
                        m_weight*rfm_df.m_rank, 0 )

# Grupo
quintiles = rfm_df.score.quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
rfm_df.loc[rfm_df.score < quintiles[0.2] ,'grupo'] = 1

rfm_df.loc[(rfm_df.score >= quintiles[0.2]) &
           (rfm_df.score < quintiles[0.4]),'grupo'] = 2

rfm_df.loc[(rfm_df.score >= quintiles[0.4]) &
           (rfm_df.score < quintiles[0.6]),'grupo'] = 3

rfm_df.loc[(rfm_df.score >= quintiles[0.6]) &
           (rfm_df.score < quintiles[0.8]),'grupo'] = 4

rfm_df.loc[rfm_df.score >= quintiles[0.8] ,'grupo'] = 5

rfm_df.grupo = rfm_df.grupo.astype(int)


# Clasificacion
rfm_df.loc[rfm_df.grupo == 5 ,'segmento'] = 'Cliente tradicional'

rfm_df.loc[rfm_df.grupo == 4,'segmento'] = 'Cliente premium'

rfm_df.loc[rfm_df.grupo == 3,'segmento'] = 'Cliente silver'

rfm_df.loc[rfm_df.grupo == 2,'segmento'] = 'Cliente gold'

rfm_df.loc[rfm_df.grupo == 1,'segmento'] = 'Cliente platinum'

rfm_df.grupo.value_counts()


plt.figure(figsize=(15, 10))
fig = px.parallel_coordinates(rfm_df,
                              color='grupo',
                              dimensions=['recency', 
                                          'cantidad', 'monto_neto',
                                          'grupo'],
                              color_continuous_midpoint=3)

fig.show()

# Matriz RFM
rfm_df.groupby(by=['grupo', 'segmento'], as_index=False)['recency', 'cantidad', 'monto_neto'].mean()