import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             balanced_accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import xgboost as xgb



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
    
    # 16 categóricas
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
    'TSH': (0.1, 100),
    'T3': (0.1, 350),#17.5
    'TT4': (0.1, 300),
    'T4U': (0.1, 20),
    'FTI': (0.1, 300),
    'age': (0, 100),
}

# Carrega o dataset
df = pd.read_excel("thyroid.xlsx")

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

# Remover linhas com muitos NaNs 
df = df.dropna(thresh=21)

# Converter categóricas para binário
#for col in categ_features:
#    df[col] = df[col].map(cat_to_bin)

print(df.shape)
print("Classes target:\n", df['target'].value_counts())

###################################################################
#df.info()

knn = KNNImputer(n_neighbors=10)
df_imputed = knn.fit_transform(df[num_features])
df_imputed = pd.DataFrame(df_imputed, columns=num_features, index=df.index)

# # Garantir que colunas categóricas continuam como inteiros 0/1
# for col in categ_features:
#     df_imputed[col] = df_imputed[col].round().astype(int)

# Atualizar df com as colunas imputadas
df[num_features] = df_imputed
X = df[features]
y = df['target']


# One-Hot Encoding
X = pd.get_dummies(X, columns=categ_features, drop_first=True)

# Split original 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=90, stratify=y)

# 2. Aplicar SMOTE apenas no treino
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# xgb_clf = XGBClassifier(max_depth=3)
# xgb_clf.fit(X_train, y_train)
# Modelo XGBoost
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=3, 
    missing=1, 
    early_stopping_rounds=10, 
    eval_metric=['merror','mlogloss'], 
    seed=42
)

xgb_clf.fit(
    X_train_resampled, y_train_resampled,
    verbose=0,
    eval_set=[(X_train_resampled, y_train_resampled), (X_test, y_test)]
)

# ------------------ Plots de evolução ------------------ #
results = xgb_clf.evals_result()
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
y_pred = xgb_clf.predict(X_test)

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
# Verificar classes

#df_clean = df[features + ['target_multi']].copy() 
# df_clean.info()
# for col in num_features:
#     Q1 = df_clean[col].quantile(0.25)
#     Q3 = df_clean[col].quantile(0.75)
#     IQR = Q3 - Q1
#     limite_superior = Q3 + 1.5 * IQR
#     limite_inferior = Q1 - 1.5 * IQR
#     # Substitui outliers por NaN para serem tratados pelo imputer depois
#     # Ou você pode substituí-los pelos limites (capping)
#     df_clean.loc[df_clean[col] > limite_superior, col] = None # ou limite_superior
#     df_clean.loc[df_clean[col] < limite_inferior, col] = None # ou limite_inferior

# # Aplicar limites para colunas numericas
# for col, (min_val, max_val) in limits.items():
#     df.loc[~df[col].between(min_val, max_val), col] = pd.NA
# #for col, (min_val, max_val) in limits.items():
# #    df_clean = df_clean[( (df_clean[col] >= min_val) & (df_clean[col] <= max_val) ) | (df_clean[col].isna()) ]

# # df_clean['T3'] = df_clean['T3'].fillna(df_clean['T3'].median())
# # df_clean['TSH'] = df_clean['TSH'].fillna(3.0)
# # df_clean['TT4'] = df_clean['TT4'].fillna(df_clean['TT4'].median())
# # df_clean['T4U'] = df_clean['T4U'].fillna(df_clean['T4U'].median())
# # df_clean.loc[df_clean['FTI'].isna(), 'FTI'] = df_clean['TT4'] * df_clean['T4U'] / 100
# # df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())
# # df_clean['TBG'] = df_clean['TBG'].fillna(df_clean['TBG'].median())


# #df_clean.loc[df_clean['FTI'].isna(), 'FTI'] = df_clean['TT4'] * df_clean['T4U'] / 100
# # Preenchimento de NaNs numéricos com KNNImputer
# X_num = df[num_features]
# imputer = KNNImputer(n_neighbors=5, weights="distance")
# X_num_imputed = imputer.fit_transform(X_num)
# df[num_features] = X_num_imputed

# # Preencher NaNs categóricas com a mais frequente(moda)
# for col in categ_features:
#     df[col] = df[col].fillna(df[col].mode()[0])

# # print(df_clean[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']].describe())
# # for col in num_features:
# #     plt.figure(figsize=(8, 4))
# #     sns.histplot(df_clean[col].dropna(), kde=True, bins=30)  # kde=True mostra a curva de densidade
# #     plt.title(f"Distribuição de {col}")
# #     plt.xlabel(col)
# #     plt.ylabel("Frequência")
# #     plt.show()

# # Normalização com StandardScaler
# #scaler = StandardScaler()
# #df_clean[num_features] = scaler.fit_transform(df_clean[num_features])

# #print(df_clean[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']].describe())

# # Criar DataFrame X apenas com essas features
# X = df[features]

# # One-hot encoding para as categóricas
# X = pd.get_dummies(X, columns=categ_features, drop_first=True)

# #print(X[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']].describe())
# #print("Classes target_multi:\n", df_clean['target_multi'].value_counts())

# # target
# y = df['target']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# smote = SMOTE(random_state=44)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # model= XGBClassifier(base_score=None, booster=None, callbacks=None,
# #               colsample_bylevel=None, colsample_bynode=None,
# #               colsample_bytree=None, early_stopping_rounds=None,
# #               enable_categorical=False, eval_metric=None, feature_types=None,
# #               gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
# #               interaction_constraints=None, learning_rate=None, max_bin=None,
# #               max_cat_threshold=None, max_cat_to_onehot=None,
# #               max_delta_step=None, max_depth=3, max_leaves=None,
# #               min_child_weight=None,monotone_constraints=None,
# #               n_estimators=100, n_jobs=None, num_parallel_tree=None,
# #               objective='multi:softprob', predictor=None)
# model = XGBClassifier(
#     subsample=0.7,
#     reg_lambda=10,
#     reg_alpha=0.5,
#     n_estimators=200, 
#     max_depth=6, 
#     learning_rate=0.2, 
#     gamma=0.3, 
#     colsample_bytree=1.0,
#     num_class=3,
#     #scale_pos_weight = (len(y_train_res[y_train_res==0]) / len(y_train_res[y_train_res==1])) 
#     #objective='multi:softmax', 
#     #random_state=42
# )

# # Modelo XGBoost
# # model = XGBClassifier(
# #     n_estimators=200,
# #     learning_rate=0.2,
# #     max_depth=5,
# #     subsample=0.7,
# #     colsample_bytree=1.0,
# # )

# model.fit(X_train_res, y_train_res)
# y_pred = model.predict(X_test)

# # Avaliação
# print("Acurácia:", accuracy_score(y_test, y_pred))
# print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Importância das features
# feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
# feat_importances.sort_values().plot(kind='barh')
# plt.title("Importância das features")
# plt.show()
