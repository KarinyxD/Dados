import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega o dataset
df = pd.read_excel("thyroid.xlsx")
print(df['target_binary'].value_counts())
