import preprocessamento as pp
import analise as an
import pandas as pd
# import XGBoost as xgb

if __name__ == "__main__":
    # pré-processamento
    df, target = pp.preprocessing()
    
    # Análise gráfica
    df_combined = pd.concat([df, target], axis=1)
    an.graph_attributes_vs_target(df)
    
    # xgb.xgboost_model(df, target)