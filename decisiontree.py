import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

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
    'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG',
    
    # 16 categóricas (exemplos, substituir pelos nomes corretos do seu dataset)
    'sex', 'on_thyroxine', 'on_antithyroid_meds', 'query_on_thyroxine',
    'query_hypothyroid', 'pregnant', 'sick', 'tumor', 
    'lithium', 'goitre', 'TSH_measured', 'T3_measured', 
    'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured'
]  

categorical_features = [
    'sex', 'on_thyroxine', 'on_antithyroid_meds', 'query_on_thyroxine',
    'query_hypothyroid', 'pregnant', 'sick', 'tumor', 
    'lithium', 'goitre', 'TSH_measured', 'T3_measured', 
    'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured'
]

# Definir limites para colunas (remover outliers)
limits = {
    'TSH': (0.1, 80),
    'T3': (0.1, 145),
    'TT4': (0.1, 150),
    'T4U': (0.1, 150),
    'FTI': (0.1, 800),
    'age': (0, 120),
    'TBG': (0.1, 200)
}

# Carrega o dataset
df = pd.read_excel("thyroid.xlsx")

# Limpar valores da coluna target (remover espaços e maiúsculas)
df['target'] = df['target'].str.strip().str.upper()

# Aplicar valores numericos à coluna target
df['target_multi'] = df['target'].apply(map_target)

# Verificar classes
#print("Classes target_multi:\n", df['target_multi'].value_counts())

df_clean = df.copy()
# Aplicar filtro para cada coluna
for col, (min_val, max_val) in limits.items():
    df_clean = df_clean[( (df_clean[col] >= min_val) & (df_clean[col] <= max_val) ) | (df_clean[col].isna()) ]

#print(df_clean[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age', 'TBG']].describe())
# Preencher NaNs numéricos com a mediana ou valor específico
df_clean['T3'] = df_clean['T3'].fillna(df_clean['T3'].median())
df_clean['TSH'] = df_clean['TSH'].fillna(3.0)
df_clean['TT4'] = df_clean['TT4'].fillna(df_clean['TT4'].median())
df_clean['T4U'] = df_clean['T4U'].fillna(df_clean['T4U'].median())
df_clean.loc[df_clean['FTI'].isna(), 'FTI'] = df_clean['TT4'] * df_clean['T4U'] / 100
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())
df_clean['TBG'] = df_clean['TBG'].fillna(df_clean['TBG'].median())

# Preencher NaNs categóricas com a mais frequente
for col in categorical_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Criar DataFrame X apenas com essas features
X = df_clean[features]

# One-hot encoding para as categóricas
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print(X[['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age', 'TBG']].describe())
print("Classes target_multi:\n", df_clean['target_multi'].value_counts())

# Escolher target
y = df_clean['target_multi'] 

# Preencher valores faltantes
X = X.fillna(X.mean())

# Aplicar SMOTE - oversampling para balancear as classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# smote = SMOTE(random_state=42, sampling_strategy='not majority')  # gera exemplos para todas as classes menores que a maior
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# print("Classes após SMOTE:\n", pd.Series(y_train_res).value_counts())

# Usar pesos de classe
# Em vez de balancear com SMOTE, você pode dar pesos maiores para classes minoritárias no modelo 
# (class_weight='balanced' em DecisionTreeClassifier ou RandomForest).
# Prós: não altera os dados originais, evita overfitting de dados sintéticos.

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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
