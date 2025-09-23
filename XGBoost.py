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

def plot_distributions(original_df, imputed_df, num_features):
    for col in num_features:
        plt.figure(figsize=(12, 5))

        # Histograma
        plt.subplot(1, 2, 1)
        sns.histplot(original_df[col], bins=30, kde=True, color="blue", label="Original")
        sns.histplot(imputed_df[col], bins=30, kde=True, color="orange", label="KNN", alpha=0.6)
        plt.title(f"Distribuição - {col}", fontsize=16)  # Aumenta o tamanho do título
        plt.xlabel(col, fontsize=20)  # Aumenta o tamanho do rótulo do eixo X
        plt.ylabel("Frequência", fontsize=20)  # Aumenta o tamanho do rótulo do eixo Y
        plt.legend(fontsize=20)  # Aumenta o tamanho da legenda
        plt.tick_params(axis='both', labelsize=20)  # Aumenta o tamanho dos números nos eixos
        plt.show()

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

# Letras que representam condições
hyper_letters = ['A', 'B', 'C', 'D']  # hyperthyroid
hypo_letters = ['E', 'F', 'G', 'H']   # hypothyroid
valid_letters = hyper_letters + hypo_letters + ['-']

# Target multiclass: 0 = outros, 1 = hypothyroid, 2 = hyperthyroid
def map_target(x):
    if any(c in hypo_letters for c in x):
        return 1
    elif any(c in hyper_letters for c in x):
        return 2
    else:
        return 0
    
features = [
    # 6 contínuas
    'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI',
    
    # 15 categóricas
    'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_meds', 'query_hypothyroid', 'query_hyperthyroid', 
    'pregnant', 'sick', 'tumor', 'hypopituitary', 'psych',
    'lithium', 'goitre', 'thyroid_surgery', 'I131_treatment'
]  
num_features = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']
categ_features = [
    'sex', 'on_thyroxine', 'query_on_thyroxine',
    'on_antithyroid_meds', 'query_hypothyroid', 'query_hyperthyroid', 
    'pregnant', 'sick', 'tumor', 'hypopituitary', 'psych',
    'lithium', 'goitre', 'thyroid_surgery', 'I131_treatment'
]

# categorias para binário
cat_to_bin = {
    'f': 0, 't': 1,
    'F': 0, 'M': 1
}

# Definir limites para colunas (remover outliers)
limits = {
    'TSH': (0, 500),
    'T3': (0, 20),
    'TT4': (0, 400),
    'T4U': (0, 2),
    'FTI': (0, 300),
    'age': (0, 100),
}

# Carrega o dataset
df = pd.read_csv("thyroidDF.csv")

# Limpar valores da coluna target (remover espaços, maiúsculas)
df = df.dropna(subset=['target'])
df['target'] = df['target'].str.strip().str.upper()

# Filtrar apenas os diagnósticos válidos
pattern = '|'.join(valid_letters)
df = df[df['target'].str.contains(pattern, na=False)]

# Mapear target para 0,1,2 e remover colunas desnecessárias
df['target'] = df['target'].apply(map_target)
df = df[features + ['target']]

# limitar idade
df = df[df['age'] < 100]

# changing sex of observations with ('pregnant' == True) & ('sex' == null) to Female
df['sex'] = np.where((df['sex'].isnull()) & (df['pregnant'] == 't'), 'F', df['sex'])

# Remover linhas com muitos NaNs 
df = df.dropna(thresh=18)

#print(df.shape)

print("Classes target:\n", df['target'].value_counts())
# Aplicar limites para cada coluna numerica
for col, (min_val, max_val) in limits.items():
    df = df[df[col].isna() | ((df[col] >= min_val) & (df[col] <= max_val))]

print("Classes target:\n", df['target'].value_counts())
# Converter categóricas para binário
for col in categ_features:
    df[col] = df[col].map(cat_to_bin)

# Aplicar KNN em todas as features
imputer = KNNImputer(n_neighbors=10, weights="distance")
df_imputed = imputer.fit_transform(df[features])
df_imputed = pd.DataFrame(df_imputed, columns=features, index=df.index)

#plot_distributions(df, df_imputed, num_features)
# Atualizar df com as colunas imputadas
df[features] = df_imputed
X = df[features]
y = df['target']

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

