import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# --- 1. Carregamento e Definições Iniciais ---
df = pd.read_excel("thyroid.xlsx")

# Letras que representam condições
hyper_letters = ['A', 'B', 'C', 'D']  # hyperthyroid
hypo_letters = ['E', 'F', 'G', 'H']   # hypothyroid

# Target multiclass: 0 = normal, 1 = hypothyroid, 2 = hyperthyroid
def map_target(x):
    if any(c in hypo_letters for c in str(x)):
        return 1
    elif any(c in hyper_letters for c in str(x)):
        return 2
    else:
        return 0

features = [
    'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'sex', 'on_thyroxine', 
    'query_on_thyroxine', 'on_antithyroid_meds', 'query_hypothyroid', 
    'query_hyperthyroid', 'pregnant', 'sick', 'tumor', 'hypopituitary', 
    'psych', 'lithium', 'goitre', 'thyroid_surgery', 'I131_treatment'
]
num_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
categ_features = [
    'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'query_hypothyroid', 'query_hyperthyroid', 'pregnant', 'sick', 'tumor',
    'hypopituitary', 'psych', 'lithium', 'goitre', 'thyroid_surgery', 'I131_treatment'
]

# --- 2. Limpeza Inicial ---
# Limpar e mapear a coluna target
df['target'] = df['target'].str.strip().str.upper()
df['target_multi'] = df['target'].apply(map_target)

# Substituir '?' por NaN (valor nulo) para tratamento padronizado
df.replace('?', np.nan, inplace=True)

# Converter colunas numéricas para o tipo float, forçando erros a virarem NaN
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Definir X e y
X = df[features]
y = df['target_multi']

# --- 3. DIVISÃO DE DADOS (Passo CRÍTICO para evitar Data Leakage) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criar cópias para evitar SettingWithCopyWarning
X_train = X_train.copy()
X_test = X_test.copy()

# --- 4. Tratamento de Outliers (APENAS com base nos dados de TREINO) ---
print("Tratando outliers...")
for col in num_features:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 1.5 * IQR
    
    # "Capping": Substitui valores extremos pelo limite superior calculado no TREINO
    X_train.loc[X_train[col] > limite_superior, col] = limite_superior
    X_test.loc[X_test[col] > limite_superior, col] = limite_superior # Usa o mesmo limite do treino

# --- 5. Scaling de Features Numéricas (Necessário para KNNImputer) ---
scaler = StandardScaler()

# Ajusta o scaler APENAS nos dados de treino e transforma ambos
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# --- 6. Imputação de Dados Faltantes ---
# Imputação de features numéricas com KNN
print("Preenchendo valores numéricos faltantes...")
imputer_num = KNNImputer(n_neighbors=5, weights="distance")

# Ajusta o imputer APENAS nos dados de treino e transforma ambos
X_train[num_features] = imputer_num.fit_transform(X_train[num_features])
X_test[num_features] = imputer_num.transform(X_test[num_features])

# Imputação de features categóricas com a moda
print("Preenchendo valores categóricos faltantes...")
for col in categ_features:
    moda = X_train[col].mode()[0]
    X_train[col].fillna(moda, inplace=True)
    X_test[col].fillna(moda, inplace=True)

# --- 7. One-Hot Encoding para Variáveis Categóricas ---
X_train = pd.get_dummies(X_train, columns=categ_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categ_features, drop_first=True)

# Alinhar colunas para garantir que treino e teste tenham as mesmas features
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols] # Garante a mesma ordem e número de colunas

# --- 8. Balanceamento de Classes com SMOTE (APENAS nos dados de TREINO) ---
print("Balanceando as classes de treino com SMOTE...")
print("Classes antes do SMOTE:\n", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Classes depois do SMOTE:\n", y_train_res.value_counts())

# --- 9. Treinamento do Modelo ---
print("Treinando o modelo XGBClassifier...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X_train_res, y_train_res)

# --- 10. Avaliação do Modelo ---
print("\nAvaliação do modelo no conjunto de teste:")
y_pred = model.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Hypothyroid', 'Hyperthyroid']))

# --- 11. Importância das Features ---
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10, 8))
feat_importances.nlargest(20).plot(kind='barh')
plt.title("Top 20 Features mais importantes")
plt.xlabel("Importância")
plt.tight_layout()
plt.show()