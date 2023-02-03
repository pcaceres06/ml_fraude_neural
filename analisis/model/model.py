# -*- coding: utf-8 -*-
import os
import collections
from cmath import inf

# Manipulacion de datos
import pandas as pd
import numpy as np
import pycircular as pc

# Visualizacion
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
# %matplotlib inline

# Train y validation set
from sklearn.model_selection import train_test_split

# Metricas
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score

# Pipelines
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline

# Preprocesamiento
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer, make_column_selector

# Imputacion
from sklearn.impute import KNNImputer, SimpleImputer

# Modelamiento
from sklearn.discriminant_analysis import LinearClassifierMixin, LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

#Optimización Hiperparametros
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.model_selection import KFold

# Generar modelo
import pickle
from utils.serialized_model import save_model

pd.options.display.max_columns = None
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.2f}'.format

FILE_PATH = f'{os.getcwd()}/data/processed'

# Cargar datos
df = pd.read_csv(f'{FILE_PATH}/transactions.csv', sep=',')

# Seleccionar variables de acuerdo con el EDA
cols_keep = ['genero', 'radian', 'device_score', 'tipo_tc',
             'status_txn', 'is_prime', 'dia_semana', 'dia_mes',
             'monto_neto', 'pct_cupo', 'fraude']
df = df[cols_keep]
df.device_score = df.device_score.astype('category')

#======================================
# Particion de datos
#======================================
X = df.drop('fraude', axis=1)
y = df.fraude

# Entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    train_size=0.8,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=2023)

train_df = pd.concat([x_train, y_train.to_frame()], axis=1)

# Entrenamiento y validacion
X = x_train
y = y_train
x_train, x_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  train_size=0.8,
                                                  stratify=y,
                                                  shuffle=True,
                                                  random_state=202302)

train_df = pd.concat([x_train, y_train.to_frame()], axis=1)
val_df = pd.concat([x_val, y_val.to_frame()], axis=1)


# Validación de proporciones
df.fraude.value_counts() / len(df)
y_train.value_counts() / len(y_train)
y_val.value_counts() / len(y_val)
y_test.value_counts() / len(y_test)

def live_plot(data_dict, figsize=(7,5), title='', win_size: int = 100):
    """
    Función para mostrar en tiempo real el progreso de la optmización bayesiana.
    """
    # clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        if len(data) > win_size:
            data = data[-win_size:]
            iterations = np.arange(len(data))[-win_size:] 
        else:
            iterations = np.arange(len(data))
        plt.plot(iterations, data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show()

# Función para generar pipeline completo
def add_model(data_pipeline, model) -> Pipeline:
    whole_pipeline = Pipeline([
        ("data_pipeline", data_pipeline),
        ("model", model)
    ])
    return whole_pipeline


# Funcion para entrenar y evaluar el modelo con KFold
def train_and_evaluate(scaler_choice, 
                        imputer_strategy,
                        knn_imputer, 
                        knn_imputer_k,
                        pca_components, 
                        model_penalty,
                        model_C, 
                        model_pos_class_weight,
                        # RNA
                        model_hidden_layer_size_exp,
                        model_lr_init,
                        model_alpha,
                        model_batch_size,
                        model_max_iter,
                        model_solver,
                        verbose=False,
                        show_live_plot=True
                        ) -> float:
    """  Funcion para entrenamiento y validacion del modelo

    Args:
        scaler_choice (_type_): _description_
        imputer_strategy (_type_): _description_
        knn_imputer (_type_): _description_
        knn_imputer_k (_type_): _description_
        pca_components (_type_): _description_
        model_penalty (_type_): _description_
        model_C (_type_): _description_
        model_pos_class_weight (_type_): _description_
        model_hidden_layer_size_exp (_type_): _description_
        model_lr_init (_type_): _description_
        model_alpha (_type_): _description_
        model_batch_size (_type_): _description_
        model_max_iter (_type_): _description_
        model_solver (_type_): _description_
        verbose (int, optional): _description_. Defaults to 1.


        model_selection: Indica el modelo a entrenar
                        1 - Regresion logística
                        2 - SVM
                        3 - LDA
                        4 - QDA
                        5 - MLPClassifier 
                        6 - Ensamble (RL, LDA, QDA)
                        7 - Ensamble (RL, SVM)
                        8 - Ensamble (RL, SVM, LDA, QDA)
                        9 - Ensamble (RL, SVM, QDA, MLP)

    Returns:
        float: ROC AUC score
    """
    
    #############################
    # Configuracion inicial
    #############################
    
    # Datos
    X_train = x_train
    Y_train = y_train
    Y_train = Y_train.astype(np.float32)
    
    # Modelo a utilizar 
    model_selection = model_selection_conf
    cv = cv_conf

    # Steps pipeline preprocesor
    pca_step = pca_step_conf
    lda_step = lda_step_conf

    # Definicion de columnas numericas y categoricas

    var_categoricas = [*X_train.select_dtypes(include='object').columns[:5]]
    var_numericas = [*X_train.select_dtypes(exclude='object').columns]

    # Estrategia de escalamiento de datos
    scaler_cls = StandardScaler if scaler_choice > 0.5 else MinMaxScaler
    
    # Estrategia de imputacion de dato numericos
    imputer_strategy = "mean" if imputer_strategy > 0.5 else "median"
    if knn_imputer > 0.5:
        imputer = KNNImputer(n_neighbors=int(knn_imputer_k))
    else:
        imputer = SimpleImputer(strategy=imputer_strategy)
        

    # Transformacion variables numericas
    numeric_transformer = Pipeline(
        steps=[("imputer", imputer), ("scaler", scaler_cls())]
    )

    # Transformacion variables categoricas
    categorical_transformer = Pipeline(
        steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")), 
                ('ohe', OneHotEncoder(handle_unknown="ignore"))
            ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, var_numericas),
            ("cat", categorical_transformer, var_categoricas),
        ],
        remainder='passthrough'
    )
    
    # Pasos pipeline
    if pca_step:
        data_pipeline = Pipeline(steps=[
                                    ("data_processor", preprocessor),
                                    ("pca", PCA(n_components=int(pca_components))),
                                ])
    elif lda_step:
        data_pipeline = Pipeline(steps=[
                                    ("data_processor", preprocessor),
                                    ("lda", LinearDiscriminantAnalysis(n_components=1)),
                                ])
    else:
        data_pipeline = Pipeline(steps=[
                                    ("data_processor", preprocessor)
                                ])
    
    # Seleccion del modelo
    if model_selection == 1:
        model =  LogisticRegression(
            penalty="l1" if model_penalty > 0.5 else "l2",
            solver='liblinear',
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )
    elif model_selection == 2:
        if model_penalty < 0.33:
            kernel = 'linear'
        elif model_penalty < 0.67:
            kernel = 'rbf'
        else:
            kernel = 'sigmoid'

        model = SVC(
            probability=True,
            kernel=kernel,
            gamma=model_C,
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )
    elif model_selection == 3:
        model = LinearDiscriminantAnalysis()

    elif model_selection == 4:
        model = QuadraticDiscriminantAnalysis()

    elif model_selection == 5:

        max_exponent = int(model_hidden_layer_size_exp)
        
        model_kwargs = dict(
            hidden_layer_sizes = [2**(n) for n in reversed(range(3, max_exponent))],
            solver="adam" if model_solver > 0.5 else "sgd",
            batch_size=2**int(model_batch_size),
            learning_rate_init=model_lr_init,
            alpha=model_alpha,
            max_iter=int(model_max_iter),
            early_stopping=True,
            random_state=42,
        )
        
        if verbose:
            print("MLP Classifier params: ")
            print(model_kwargs)
        
        model = MLPClassifier(**model_kwargs)

    elif model_selection == 6:
        lr = LogisticRegression(
            penalty="l1" if model_penalty > 0.5 else "l2",
            solver='liblinear',
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()

        model = VotingClassifier(
            estimators=[('lr', lr), ('lda', lda), ('qda', qda)],
            voting='soft'
        )
    elif model_selection == 7:
        lr = LogisticRegression(
            penalty="l1" if model_penalty > 0.5 else "l2",
            solver='liblinear',
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )

        if model_penalty < 0.33:
            kernel = 'linear'
        elif model_penalty < 0.67:
            kernel = 'rbf'
        else:
            kernel = 'sigmoid'

        svm = SVC(
            probability=True,
            kernel=kernel,
            gamma=model_C,
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )

        model = VotingClassifier(
            estimators=[('lr', lr), ('svm', svm)],
            voting='soft'
        )

    elif model_selection == 8:
        lr = LogisticRegression(
            penalty="l1" if model_penalty > 0.5 else "l2",
            solver='liblinear',
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()

        if model_penalty < 0.33:
            kernel = 'linear'
        elif model_penalty < 0.67:
            kernel = 'rbf'
        else:
            kernel = 'sigmoid'

        svm = SVC(
            probability=True,
            kernel=kernel,
            gamma=model_C,
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )

        model = VotingClassifier(
            estimators=[('lr', lr), ('lda', lda), ('qda', qda), ('svm', svm)],
            voting='soft'
        )

    elif model_selection == 9:
        lr = LogisticRegression(
            penalty="l1" if model_penalty > 0.5 else "l2",
            solver='liblinear',
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()

        if model_penalty < 0.33:
            kernel = 'linear'
        elif model_penalty < 0.67:
            kernel = 'rbf'
        else:
            kernel = 'sigmoid'

        svm = SVC(
            probability=True,
            kernel=kernel,
            gamma=model_C,
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )

        max_exponent = int(model_hidden_layer_size_exp)
        
        model_kwargs = dict(
            hidden_layer_sizes = [2**(n) for n in reversed(range(3, max_exponent))],
            solver="adam" if model_solver > 0.5 else "sgd",
            batch_size=2**int(model_batch_size),
            learning_rate_init=model_lr_init,
            alpha=model_alpha,
            max_iter=int(model_max_iter),
            early_stopping=True,
            random_state=42,
        )
        
        rna = MLPClassifier(**model_kwargs)

        model = VotingClassifier(
            estimators=[('lr', lr), ('qda', qda), ('svm', svm), ('rna', rna)],
            voting='soft'
        )

    elif model_selection == 10:
        lr = LogisticRegression(
            penalty="l1" if model_penalty > 0.5 else "l2",
            solver='liblinear',
            C=model_C,
            class_weight={0: 1, 1: model_pos_class_weight}
        )
        lda = LinearDiscriminantAnalysis()
        qda = QuadraticDiscriminantAnalysis()
    
        max_exponent = int(model_hidden_layer_size_exp)    
        model_kwargs = dict(
            hidden_layer_sizes = [2**(n) for n in reversed(range(3, max_exponent))],
            solver="adam" if model_solver > 0.5 else "sgd",
            batch_size=2**int(model_batch_size),
            learning_rate_init=model_lr_init,
            alpha=model_alpha,
            max_iter=int(model_max_iter),
            early_stopping=True,
            random_state=42,
        )
        
        rna = MLPClassifier(**model_kwargs)

        model = VotingClassifier(
            estimators=[('lr', lr), ('qda', qda), ('lda', lda), ('rna', rna)],
            voting='soft'
        )

    # Pipeline modelo
    pipeline = add_model(data_pipeline, model)
    kf = KFold(n_splits=cv, random_state=42, shuffle=True)
    
    # metricas del modelo
    roc_auc_train = [] 
    roc_auc_val = []
    
    # K-Fold cross val
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        #print(f"Fold number: {i+1}")
        kX_train, kX_val = X_train.iloc[train_index], X_train.iloc[test_index]
        ky_train, ky_val = Y_train.iloc[train_index], Y_train.iloc[test_index]
        #print(f"Training with {kX_train.shape}")
        #print(f"Validating with {kX_val.shape}")
        pipeline.fit(kX_train, ky_train.astype(np.float32))
        
        val_preds = pipeline.predict_proba(kX_val)[:, 1]
        val_auc = roc_auc_score(ky_val.astype(np.float32), val_preds)
        
        train_preds = pipeline.predict_proba(kX_train)[:, 1]
        train_auc = roc_auc_score(ky_train.astype(np.float32), train_preds)
        
        roc_auc_train.append(train_auc)
        roc_auc_val.append(val_auc)
    
    roc_auc_val = np.array(roc_auc_val)
    roc_auc_train = np.array(roc_auc_train)
    
    adj_val_roc_auc = (roc_auc_val.mean() -  roc_auc_val.std())
    adj_train_roc_auc = (roc_auc_train.mean() -  roc_auc_train.std())
    objective = adj_val_roc_auc - abs(adj_val_roc_auc - adj_train_roc_auc)
    print(f"Validation ROC AUC adjusted score: {adj_val_roc_auc}")
    print(f"Train ROC AUC adjusted score: {adj_train_roc_auc}")
    print()

    if show_live_plot:
        data['train_roc_auc'].append(adj_train_roc_auc)
        data['val_roc_auc'].append(adj_val_roc_auc)
        data["objective"].append(objective)
        live_plot(data)

    return pipeline, objective


# Función de caja negra
def target_func(**kwargs):
    """ Funcion de caja negra para el optimizado de bayes
    """
    model, result = train_and_evaluate(**kwargs)
    return result


# Función para probar el mejor modelo con validation set, set completo y test set
def test_best_model(x_data=None, y_data=None, tipo_data_set=1):

    """ Funcion para probar el modelo con el set de datos completos, 
        el set de validación y el set de prueba

    Args:
        tipo_data_set (str, optional):
                                1. Validacion
                                2. Total
                                3. Test
    """
    if tipo_data_set == 1:
        df_x = x_data
        df_y = y_data
        val_preds = best_model.predict_proba(df_x)[:, 1]
        score = roc_auc_score(df_y.astype(np.int32), val_preds)
        return f"Validation set ROC AUC score: {score}"

    elif tipo_data_set == 2:
        df_x = x_data
        df_y = y_data
        tot_preds = best_model.predict_proba(df_x)[:, 1]
        score = roc_auc_score(df_y.astype(np.int32), tot_preds)
        return f"Full set ROC AUC score: {score}"
    else:
        df_x = x_data
        df_y = y_data
        tot_preds = best_model.predict_proba(df_x)[:, 1]
        score = roc_auc_score(df_y.astype(np.int32), tot_preds)
        return f"Test set ROC AUC score: {score}"
        


"""
--------------------------------------------------
CONFIGURACION DE LA FUNCION train_and_evaluate
--------------------------------------------------

En esta sección se realizan las configuraciones de la función
train_and_evaluate para el entrenamiento de los modelos.

#################
Instrucciones: ##
#################

########################################################################################
1. Seleccionar el modelo a usar; asignar un valor de 1 a 10 en model_selection_conf ####
########################################################################################

# Seleccion del modelo
model_selection_conf :  1 - Regresion logística
                        2 - SVM
                        3 - LDA
                        4 - QDA
                        5 - MLPClassifier 
                        6 - Ensamble (RL, LDA, QDA)
                        7 - Ensamble (RL, SVM)
                        8 - Ensamble (RL, SVM, LDA, QDA)
                        9 - Ensamble (RL, SVM, QDA, MLP)
                        10 - Ensamble (RL, LDA, QDA, MLP)

# Croos validation
cv_conf : numero entero para cross validation


#######################################################
2. Definir uso de PCA o LDA para feature selection ####
#######################################################

# PCA step
pca_step_conf : False - No agregar PCA al pipeline
                True - Agregar PCA al pipeline

# LDA step
lda_step_conf : False - No agregar LDA al pipeline
                True - Agregar LDA al pipeline

*Nota: Lasso y Ridge se usan en el modelo de Regresión Logistica con el parametro Penalty

###########################
3. Optimización Bayesiana #
###########################

pbounds = parámetros del modelo (no modificar)

*Nota: Por temas de compatibilidad de algunas librerias con la liberia de 
Opt. bayesiana, es poible que algunas combinaciones generen error
"""

# Data para visualizacion de la optimizacion
data = collections.defaultdict(list)

# Modelo a utilizar 
model_selection_conf = 1
cv_conf=10

# Steps pipeline preprocesor
pca_step_conf=False
lda_step_conf=True

# Parametros para la optimizacion
pbounds = dict(
    # Data
    scaler_choice=(0, 1),
    imputer_strategy=(0, 1),
    knn_imputer=(0,1),
    knn_imputer_k=(3, 7),
    pca_components=(2, x_train.shape[1]),
    # Model
    model_penalty=(0, 1), # Ridge - Lasso 
    model_C=(0.00001, 1), # alpha
    model_pos_class_weight=(1, 100),
    # Parametros adicionales redes neuronales 
    model_solver=(0, 1),
    model_hidden_layer_size_exp=(5, 7),
    model_lr_init=(0.0001, 1),
    model_alpha=(0.00001, 1),
    model_batch_size=(6, 12), # from 2**6=64 to 2**12=4096
    model_max_iter=(100, 500)
)

optimizer = BayesianOptimization(
    f=target_func,
    pbounds=pbounds,
    random_state=42,
    verbose=False,
)

optimizer.maximize(
    n_iter=30
)


# Mejores hiperparametros
optimizer.max["params"]

# Creación del modelo con los mejores parámetros
best_model, best_result = train_and_evaluate(**optimizer.max["params"])

#===============================================
# Probar mejor modelo con los demas datasets
#===============================================

# Validation set
test_best_model(x_data=x_val, y_data=y_val, tipo_data_set=1)

# Dataset completo
test_best_model(x_data=x_test, y_data=y_test, tipo_data_set=2)

# Test set
test_best_model(x_data=df.loc[:, 'genero':'pct_cupo'], 
                y_data=df.fraude, 
                tipo_data_set=3)


#==========================================
# Generacion del modelo
#==========================================
predictors_params = (x_train
                     .select_dtypes(exclude=['category', 'object'])
                     .describe()
                     .T)
predictors_params = predictors_params.reset_index()
predictors_params = predictors_params.rename(columns={'index': 'parametro'})

cols = [*predictors_params.parametro]
predictors_columns = pd.DataFrame(cols, columns=['campo'])


best_model_params = pd.DataFrame(optimizer.max["params"], index=[*range(14)])
best_model_params = best_model_params.loc[0].T.to_frame()
best_model_params = best_model_params.reset_index()
best_model_params = best_model_params.rename(columns={'index':'parametro',
                                                      0:'valor'
                                                      })

# StandardScaler if scaler_choice > 0.5 else MinMaxScaler
predictors_params
best_model_params

# Serializar modelo
file_path = f'{os.getcwd()}/modelo'
save_model(f'{file_path}/fraud_model.pkl', best_model)

# Almacenar parametros del train set
predictors_params.to_csv(f'{file_path}/predictors_params.csv', 
                         sep=',',
                         encoding='utf-8',
                         index=False)

# Almacenar parametros del mejor modelo
best_model_params.to_csv(f'{file_path}/model_params.csv', 
                         sep=',',
                         encoding='utf-8',
                         index=False)

# Almacenar columns del train set
predictors_columns.to_csv(f'{file_path}/model_columns.csv', 
                         sep=',',
                         encoding='utf-8',
                         index=False)