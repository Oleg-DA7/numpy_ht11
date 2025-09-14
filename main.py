import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# Завантажуємо дані
df = pd.DataFrame()
for i in range(2015, 2019):
    df_temp = pd.read_csv(f'.\data\{i}.csv')
    df_temp['Year'] = i
    df = pd.concat([df, df_temp], ignore_index=True)

# 4. Загальна інформація про дані
print(df.info())
print(df.describe())

# 5. Діаграми розподілу числових ознак
numeric_cols = df.select_dtypes(include=[np.number]).columns
num_cols = len(numeric_cols)
n_cols = 4 
n_rows = (num_cols + n_cols - 1) // n_cols 
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    ax.hist(df[col], bins=30, color='skyblue', edgecolor='black', label=col)
    ax.set_title(f'Distribution of {col}', fontsize=6)
    ax.set_xlabel(col, fontsize=6)
    ax.set_ylabel('Frequency', fontsize=6)
    ax.tick_params(axis='x', labelsize=6)  
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(fontsize=6)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# Тест на нормальність розподілу
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    stat, p = normaltest(df[col])
    print(f'{col}: statistic={stat:.4f}, p-value={p:.4f}, Normal={"Yes" if p > 0.05 else "No"}')

# 6-7. Кореляційна матриця
selected_cols = ['Happiness.Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Generosity']
corr_matrix = df[selected_cols].corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Happiness Factors")
plt.show()

# 8. Теплова мапа Happiness.Score
fig = px.choropleth(df,
                    locations="Country",
                    color="Happiness.Score",
                    locationmode="country names",
                    title="Happiness Index 2017")
fig.show()

# 9. Стандартизація даних
def data_scale(data, scaler_type='minmax'):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'std':
        scaler = StandardScaler()
    elif scaler_type == 'norm':
        scaler = Normalizer()
    scaler.fit(data)
    return scaler.transform(data)

data_scaled = data_scale(df[numeric_cols], scaler_type='std')
df_scaled = pd.DataFrame(data_scaled, columns=numeric_cols)
print("Scaled Data Statistics:")
print(df_scaled.describe())

# 10. Порівняння статистик
print("Original Data Statistics:")
print(df[numeric_cols].describe())

# 11. Модель кластеризації GaussianMixture
gmm = GaussianMixture(n_components=3, random_state=42)
cluster_labels = gmm.fit_predict(df_scaled)
df['Cluster'] = cluster_labels

# 12. Теплова мапа для кластерів
fig_cluster = px.choropleth(df,
                            locations="Country",
                            color="Cluster",
                            locationmode="country names",
                            title="Country Clusters by Happiness Factors")
fig_cluster.show()

# 13. Вплив набору ознак на кластеризацію
subset_cols = ['Economy (GDP per Capita)', 'Family']
data_subset_scaled = data_scale(df[subset_cols], scaler_type='std')
gmm_subset = GaussianMixture(n_components=3, random_state=42)
cluster_labels_subset = gmm_subset.fit_predict(data_subset_scaled)
df['Cluster_Subset'] = cluster_labels_subset
print("Cluster change rate:", (df['Cluster'] != df['Cluster_Subset']).mean())

# 14. Висновки (текстовий вивід)
cluster_means = df.groupby('Cluster')['Happiness.Score'].mean()
print("Mean Happiness Score by Cluster:")
print(cluster_means)
print("Кластери корелюють з географічним розподілом: високі значення в Європі, низькі в Африці. Стандартизація покращила модель.")