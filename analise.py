import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Letras que representam condições
hyper_letters = ['A', 'B', 'C', 'D']  # hyperthyroid
hypo_letters = ['E', 'F', 'G', 'H']   # hypothyroid
valid_letters = hyper_letters + hypo_letters + ['-']

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

        # Ajustar o eixo y para o gráfico de TSH
        if col == 'TSH':
            plt.xlim(0, 300)

        plt.show( )
    return

# Target multiclass: 0 = outros, 1 = hypothyroid, 2 = hyperthyroid
def map_target(x):
    if any(c in hypo_letters for c in x):
        return 1
    elif any(c in hyper_letters for c in x):
        return 2
    else:
        return 0
    
num_features = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']
# Definir limites para colunas (remover outliers)
y_limits = {
    'TSH': (0.1, 60),
    'T3': (0.1, 20),
    'TT4': (0.1, 400),
    'T4U': (0.1, 2),
    'FTI': (0.1, 300),
    'age': (0, 100),
}

def graph_attributes_vs_target(df):
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
    return