import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             balanced_accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import xgboost as xgb


# Classificador base
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    missing=1,
    seed=42,
    eval_metric='mlogloss'    # métrica para multiclass
)

# Espaço de hiperparâmetros para testar
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# GridSearch com 3-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

def xgboost_model(df, target):
    X = df
    y = target

    # Split original 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # 2. Aplicar SMOTE apenas no treino
    smote = SMOTE(random_state=44)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Modelo XGBoost
    #grid_search.fit(X_train_resampled, y_train_resampled)
    ## Melhor modelo encontrado
    # best_xgb = grid_search.best_estimator_
    # print("Melhores hiperparâmetros:", grid_search.best_params_)

    # Calcular pesos para as classes
    weights = compute_sample_weight(class_weight='balanced', y=y_train_resampled)

    # Treinar o modelo
    xgb_best = xgb.XGBClassifier(
        colsample_bytree=0.7, 
        learning_rate=0.1, 
        max_depth=7, 
        n_estimators=100, 
        subsample=1.0,
        objective='multi:softmax',
        num_class=3,
        missing=1,
        seed=42,
        eval_metric=['merror', 'mlogloss']
    )

    xgb_best.fit(
        X_train_resampled, 
        y_train_resampled,
        eval_set=[(X_train_resampled, y_train_resampled), (X_test, y_test)],
        verbose=False,
        sample_weight=weights
    )

    # Importância das features
    importance = xgb_best.feature_importances_
    feat_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # print(feat_importance)
    importance = xgb_best.feature_importances_


    # Colocar em um DataFrame para facilitar visualização
    feat_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)


    # ---------------------- Plot importância ------------------ #
    plt.figure(figsize=(10,6))
    plt.barh(feat_importance['Feature'], feat_importance['Importance'], color='skyblue')
    plt.gca().invert_yaxis()  # inverte para mostrar a mais importante no topo
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.show()

    # ------------------ Plots de evolução ------------------ #
    results = xgb_best.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    # Logloss
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('mlogloss')
    plt.title('XGBoost mlogloss')
    plt.show()

    # Merror
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    ax.plot(x_axis, results['validation_1']['merror'], label='Test')
    ax.legend()
    plt.ylabel('merror')
    plt.title('XGBoost merror')
    plt.show()

    # ------------------ Avaliação final ------------------ #
    y_pred = xgb_best.predict(X_test)

    print('\n------------------ Confusion Matrix -----------------\n')
    print(confusion_matrix(y_test, y_pred))

    print('\n-------------------- Key Metrics --------------------')
    print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    print('\n--------------- Classification Report ---------------\n')
    print(classification_report(y_test, y_pred))
    print('---------------------- XGBoost ----------------------')

