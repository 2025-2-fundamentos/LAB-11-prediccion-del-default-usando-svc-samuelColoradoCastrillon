# flake8: noqa: E501
import pandas as pd
import pickle
import gzip
import os
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer


def pregunta_1():
    def cargar_dataset(ruta_archivo):
        """Carga datos desde archivo CSV comprimido"""
        dataset = pd.read_csv(ruta_archivo, index_col=False, compression="zip")
        return dataset

    def preprocesar_dataset(dataframe):
        """Limpia y preprocesa el dataset"""
        # Renombrar columna objetivo
        dataframe = dataframe.rename(columns={"default payment next month": "default"})
        
        # Eliminar columna ID si existe
        if "ID" in dataframe.columns:
            dataframe = dataframe.drop(columns=["ID"])
        
        # Filtrar registros con valores no disponibles en MARRIAGE y EDUCATION
        dataframe = dataframe[dataframe["MARRIAGE"] != 0]
        dataframe = dataframe[dataframe["EDUCATION"] != 0]
        
        # Agrupar niveles educativos superiores
        dataframe["EDUCATION"] = dataframe["EDUCATION"].apply(
            lambda valor: valor if valor <= 3 else 4
        )
        
        # Eliminar filas con valores nulos
        dataframe = dataframe.dropna()
        
        return dataframe

    def construir_pipeline_clasificacion(datos_entrenamiento):
        """Construye el pipeline de procesamiento y clasificación"""
        columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
        columnas_numericas = [
            col for col in datos_entrenamiento.columns if col not in columnas_categoricas
        ]

        transformador_columnas = ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
                ("scaler", StandardScaler(with_mean=True, with_std=True), columnas_numericas),
            ],
            remainder="passthrough",
        )

        pipeline_completo = Pipeline(
            [
                ("preprocessor", transformador_columnas),
                ("pca", PCA()),
                ("feature_selection", SelectKBest(score_func=f_classif)),
                ("classifier", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
            ]
        )

        return pipeline_completo

    def optimizar_hiperparametros(pipeline_modelo, X_entrenamiento, y_entrenamiento):
        """Optimiza hiperparámetros usando GridSearchCV"""
        grilla_parametros = {
            "pca__n_components": [20, X_entrenamiento.shape[1] - 2],
            "feature_selection__k": [12],
            "classifier__kernel": ["rbf"],
            "classifier__gamma": [0.1],
        }

        validacion_cruzada = StratifiedKFold(n_splits=10)
        metrica_scoring = make_scorer(balanced_accuracy_score)

        busqueda_grilla = GridSearchCV(
            estimator=pipeline_modelo,
            param_grid=grilla_parametros,
            scoring=metrica_scoring,
            cv=validacion_cruzada,
            n_jobs=-1,
        )

        busqueda_grilla.fit(X_entrenamiento, y_entrenamiento)
        return busqueda_grilla

    def guardar_modelo_comprimido(modelo_entrenado, ruta_destino="files/models/model.pkl.gz"):
        """Guarda el modelo comprimido con gzip"""
        directorio = os.path.dirname(ruta_destino)
        os.makedirs(directorio, exist_ok=True)
        
        with gzip.open(ruta_destino, "wb") as archivo:
            pickle.dump(modelo_entrenado, archivo)

    def calcular_metricas_rendimiento(y_real_train, y_real_test, y_pred_train, y_pred_test):
        """Calcula métricas de rendimiento para conjuntos de train y test"""
        
        def generar_diccionario_metricas(etiquetas_reales, etiquetas_predichas, tipo_conjunto):
            return {
                "type": "metrics",
                "dataset": tipo_conjunto,
                "precision": round(precision_score(etiquetas_reales, etiquetas_predichas, zero_division=0), 4),
                "balanced_accuracy": round(balanced_accuracy_score(etiquetas_reales, etiquetas_predichas), 4),
                "recall": round(recall_score(etiquetas_reales, etiquetas_predichas), 4),
                "f1_score": round(f1_score(etiquetas_reales, etiquetas_predichas), 4),
            }

        metricas_entrenamiento = generar_diccionario_metricas(y_real_train, y_pred_train, "train")
        metricas_prueba = generar_diccionario_metricas(y_real_test, y_pred_test, "test")

        return [metricas_entrenamiento, metricas_prueba]

    def escribir_metricas_archivo(lista_metricas, ruta_archivo="files/output/metrics.json", modo_append=False):
        """Escribe métricas en archivo JSON"""
        directorio = os.path.dirname(ruta_archivo)
        os.makedirs(directorio, exist_ok=True)
        
        modo_escritura = "a" if modo_append else "w"
        with open(ruta_archivo, modo_escritura) as archivo:
            for metrica in lista_metricas:
                archivo.write(json.dumps(metrica) + "\n")

    def generar_matriz_confusion(etiquetas_verdaderas, etiquetas_predichas, nombre_dataset):
        """Genera diccionario con matriz de confusión"""
        matriz_conf = confusion_matrix(etiquetas_verdaderas, etiquetas_predichas, labels=[0, 1])
        
        diccionario_matriz = {
            "type": "cm_matrix",
            "dataset": nombre_dataset,
            "true_0": {
                "predicted_0": int(matriz_conf[0][0]),
                "predicted_1": int(matriz_conf[0][1]),
            },
            "true_1": {
                "predicted_0": int(matriz_conf[1][0]),
                "predicted_1": int(matriz_conf[1][1]),
            },
        }
        
        return diccionario_matriz

    # Cargar datasets
    datos_train = cargar_dataset("files/input/train_data.csv.zip")
    datos_test = cargar_dataset("files/input/test_data.csv.zip")

    # Preprocesar datasets
    datos_train = preprocesar_dataset(datos_train)
    datos_test = preprocesar_dataset(datos_test)

    # Separar características y variable objetivo
    X_test = datos_test.drop(columns=["default"])
    y_test = datos_test["default"]

    X_train = datos_train.drop(columns=["default"])
    y_train = datos_train["default"]

    # Construir y optimizar pipeline
    pipeline_clasificacion = construir_pipeline_clasificacion(X_train)
    modelo_optimizado = optimizar_hiperparametros(pipeline_clasificacion, X_train, y_train)

    # Guardar modelo
    guardar_modelo_comprimido(modelo_optimizado)

    # Realizar predicciones
    predicciones_train = modelo_optimizado.best_estimator_.predict(X_train)
    predicciones_test = modelo_optimizado.best_estimator_.predict(X_test)

    # Calcular y guardar métricas
    metricas_rendimiento = calcular_metricas_rendimiento(
        y_train, y_test, predicciones_train, predicciones_test
    )
    escribir_metricas_archivo(metricas_rendimiento)

    # Calcular y guardar matrices de confusión
    matriz_confusion_train = generar_matriz_confusion(y_train, predicciones_train, "train")
    matriz_confusion_test = generar_matriz_confusion(y_test, predicciones_test, "test")
    escribir_metricas_archivo([matriz_confusion_train, matriz_confusion_test], modo_append=True)


if __name__ == "__main__":
    pregunta_1()