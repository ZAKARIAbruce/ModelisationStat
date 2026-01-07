import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Charger les données
data = pd.read_csv("cleansepare.csv", sep=';')

# 2. SUPPRIMER les colonnes PROBLÉMATIQUES (URGENT)
data_clean = data.drop(columns=['User_ID', 'Height', 'Gender'])
print("Colonnes après suppression :", data_clean.columns.tolist())

# 3. Nouvelle matrice de corrélation
corr = data_clean.corr(numeric_only=True)
print("\nMatrice de corrélation (sans User_ID, Height, Gender) :\n")
print(corr.round(3))

# 4. Visualisation de la matrice de corrélation
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                 ha='center', va='center',
                 color='black' if abs(corr.iloc[i, j]) < 0.5 else 'white')

plt.colorbar(label='Corrélation')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Matrice de corrélation (sans colonnes problématiques)")
plt.tight_layout()
plt.show()

# 5. Corrélations avec Calories
print("\n=== Corrélations avec Calories (après nettoyage) ===")
for col in corr.columns:
    if col != 'Calories':
        print(f"{col} : {corr.loc['Calories', col]:.3f}")

# ===============================
# Scatter plots des variables IMPORTANTES
# ===============================

variables_importantes = ["Duration", "Heart_Rate", "Body_Temp"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, var in enumerate(variables_importantes):
    axes[idx].scatter(data_clean[var], data_clean["Calories"], alpha=0.6)
    axes[idx].set_xlabel(var)
    axes[idx].set_ylabel("Calories")
    axes[idx].set_title(f"Calories vs {var}\n(corr: {corr.loc['Calories', var]:.3f})")
plt.tight_layout()
plt.show()

# ===============================
# Scatter plots des variables OPTIONNELLES
# ===============================

variables_optionnelles = ["Age", "Weight"]

if len(variables_optionnelles) > 0:
    fig, axes = plt.subplots(1, len(variables_optionnelles), figsize=(5*len(variables_optionnelles), 5))
    if len(variables_optionnelles) == 1:
        axes = [axes]
    for idx, var in enumerate(variables_optionnelles):
        axes[idx].scatter(data_clean[var], data_clean["Calories"], alpha=0.6, color='orange')
        axes[idx].set_xlabel(var)
        axes[idx].set_ylabel("Calories")
        axes[idx].set_title(f"Calories vs {var}\n(corr: {corr.loc['Calories', var]:.3f})")
    plt.tight_layout()
    plt.show()

# ===============================
# Boxplot de Calories
# ===============================

plt.figure(figsize=(8, 4))
plt.boxplot(data_clean["Calories"], vert=False)
plt.xlabel("Calories")
plt.title("Distribution des Calories")
plt.grid(True, alpha=0.3)
plt.show()

# ===============================
# Statistiques descriptives
# ===============================

print("\n=== Statistiques descriptives des variables ===")
print(data_clean.describe().round(2))

# ===============================
# Vérification des valeurs manquantes
# ===============================

print("\n=== Valeurs manquantes ===")
print(data_clean.isnull().sum())

# ===============================
# Préparation pour la régression
# ===============================

print("\n=== Variables pour la régression ===")
print("Variables principales (fortes corrélations): Duration, Heart_Rate, Body_Temp")
print("Variables optionnelles (faibles corrélations): Age, Weight")
print("\nRecommandation: Commencez avec les 3 variables principales.")
from statsmodels.stats.outliers_influence import variance_inflation_factor

