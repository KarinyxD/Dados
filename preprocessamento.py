import pandas as pd
from sklearn.impute import KNNImputer
import analise as an
import numpy as np

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

def preprocessing():
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

    # print(df.shape)

    # print("Classes target:\n", df['target'].value_counts())
    # Aplicar limites para cada coluna numerica
    for col, (min_val, max_val) in limits.items():
        df = df[df[col].isna() | ((df[col] >= min_val) & (df[col] <= max_val))]

    # print("Classes target:\n", df['target'].value_counts())

    # Converter categóricas para binário
    for col in categ_features:
        df[col] = df[col].map(cat_to_bin)

    # Aplicar KNN em todas as features
    imputer = KNNImputer(n_neighbors=10, weights="distance")
    df_imputed = imputer.fit_transform(df[features])
    df_imputed = pd.DataFrame(df_imputed, columns=features, index=df.index)

    # Visualizar distribuições antes e depois do imputamento
    an.plot_distributions(df, df_imputed, num_features)
    print("Classes target:\n", df['target'].value_counts())

    return df_imputed, df['target']