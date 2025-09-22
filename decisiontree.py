import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

# Letras que representam condições
hyper_letters = ['A', 'B', 'C', 'D']  # hyperthyroid
hypo_letters = ['E', 'F', 'G', 'H']   # hypothyroid

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

# Definir limites para colunas (remover outliers)
limits = {
    'TSH': (0.1, 500),
    'T3': (0.1, 100),#17.5
    'TT4': (0.1, 150),
    'T4U': (0.1, 150),
    'FTI': (0.1, 300),
    'age': (0, 120),
}
#standardscaler = normalizacao do z-score)
#reamostragem - desequilibrio (data resampling methods)
#80% treino e 20% teste

# Carrega o dataset
df = pd.read_excel("thyroid.xlsx")

# Limpar valores da coluna target (remover espaços e maiúsculas)
df['target'] = df['target'].str.strip().str.upper()

# Aplicar valores numericos à coluna target
df['target_multi'] = df['target'].apply(map_target)

# Verificar classes
#print("Classes target_multi:\n", df['target_multi'].value_counts())

df_clean = df.copy()
# Aplicar limites para colunas numericas
#for col, (min_val, max_val) in limits.items():
#    df_clean.loc[~df_clean[col].between(min_val, max_val), col] = pd.NA
#for col, (min_val, max_val) in limits.items():
#    df_clean = df_clean[( (df_clean[col] >= min_val) & (df_clean[col] <= max_val) ) | (df_clean[col].isna()) ]

df_clean.loc[df_clean['FTI'].isna(), 'FTI'] = df_clean['TT4'] * df_clean['T4U'] / 100
# Preenchimento de NaNs numéricos com KNNImputer
X_num = df_clean[num_features]
imputer = KNNImputer(n_neighbors=7, weights="distance")
X_num_imputed = imputer.fit_transform(X_num)
df_clean[num_features] = X_num_imputed

# Preencher NaNs categóricas com a mais frequente(moda)
for col in categ_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

#print(df_clean[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']].describe())

# Normalização com StandardScaler
#scaler = StandardScaler()
#df_clean[num_features] = scaler.fit_transform(df_clean[num_features])

#print(df_clean[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']].describe())

# Criar DataFrame X apenas com essas features
X = df_clean[features]

# One-hot encoding para as categóricas
X = pd.get_dummies(X, columns=categ_features, drop_first=True)

# print(X[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']].describe())
# print("Classes target_multi:\n", df_clean['target_multi'].value_counts())

# target
y = df_clean['target_multi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

smote = SMOTE(random_state=44)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Modelo Decision Tree
model = DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Importância das features
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind='barh')
plt.title("Importância das features")
plt.show()

# Plot da árvore (opcional, dependendo do tamanho da árvore)
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X_train.columns, class_names=['0','1','2'], filled=True, rounded=True)
plt.show()
