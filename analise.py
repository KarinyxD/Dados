import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
y_limits = {
    'TSH': (0.1, 60),
    'T3': (0.1, 20),
    'TT4': (0.1, 400),
    'T4U': (0.1, 2),
    'FTI': (0.1, 300),
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
df = df.dropna(thresh=21)

# Converter categóricas para binário
#for col in categ_features:
#    df[col] = df[col].map(cat_to_bin)

print(df.shape)
print(df.describe())
print("Classes target:\n", df['target'].value_counts())

# Definir cores para cada classe do target
palette = {0: "blue", 1: "red", 2: "green"}  # 0=outros, 1=hypo, 2=hyper

# Configurar a grade de plots: 3 linhas x 2 colunas
fig, axes = plt.subplots(3, 2, figsize=(20, 16))
fig.suptitle('Numerical Attributes vs. Target', fontsize=22)
sns.set_style('whitegrid')

# Mapear os plots
for i, feature in enumerate(num_features):
    row = i // 2
    col = i % 2
    sns.stripplot(
        x='target', 
        y=feature, 
        data=df, 
        hue='target', 
        palette=palette,
        linewidth=0.6, 
        jitter=0.32,
        size= 3, 
        ax=axes[row, col],
        dodge=False
    )
    axes[row, col].set_ylim(y_limits[feature])
    axes[row, col].set_title(f'{feature} vs Target', fontsize=16)
    axes[row, col].set_xlabel('Target', fontsize=14)
    axes[row, col].set_ylabel(feature, fontsize=14)
    axes[row, col].set_xlim(-0.6, len(df['target'].unique()) - 0.6)
    axes[row, col].set_ylim(y_limits[feature])  # zoom no eixo y
    axes[row, col].set_xticks([0,1,2])
    axes[row, col].set_xticklabels(['Outros', 'Hypo', 'Hyper'], fontsize=12)

# Adicionar legenda geral
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, ['Outros (0)', 'Hypothyroid (1)', 'Hyperthyroid (2)'], loc='upper right', fontsize=14)

# Ajustar layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()