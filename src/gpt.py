######################################### 1

# Carga del dataset
import pandas as pd
file_path = '../assets/data/preprocessed_cirrhosis.csv'
data = pd.read_csv(file_path)

# Análisis Exploratorio Inicial
data.head()
data.describe()
data.isnull().sum()
data.nunique()

# División del dataset en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X = data.drop('Status', axis=1)
y = data['Stage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Imputación de valores faltantes
from sklearn.impute import SimpleImputer
num_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
cat_columns = X_train.select_dtypes(include=['object']).columns
imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')
X_train[num_columns] = imputer_num.fit_transform(X_train[num_columns])
X_train[cat_columns] = imputer_cat.fit_transform(X_train[cat_columns])
X_test[num_columns] = imputer_num.transform(X_test[num_columns])
X_test[cat_columns] = imputer_cat.transform(X_test[cat_columns])

# Codificación one-hot de variables categóricas
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[cat_columns]))
X_train_encoded.columns = encoder.get_feature_names_out(cat_columns)
X_train = X_train.drop(cat_columns, axis=1)
X_train = pd.concat([X_train, X_train_encoded], axis=1)
X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_columns]))
X_test_encoded.columns = encoder.get_feature_names_out(cat_columns)
X_test = X_test.drop(cat_columns, axis=1)
X_test = pd.concat([X_test, X_test_encoded], axis=1)

# Visualizaciones para el Análisis Estadístico y Estudio de Outliers
import matplotlib.pyplot as plt
import seaborn as sns
for column in num_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(data=X_train, x=column, kde=True)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(x=y_train)
plt.title('Distribución de clases (Variable objetivo)')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()

column_example = 'Bilirubin'
Q1 = X_train[column_example].quantile(0.25)
Q3 = X_train[column_example].quantile(0.75)
IQR = Q3 - Q1
outliers = ((X_train[column_example] < (Q1 - 1.5 * IQR)) | (X_train[column_example] > (Q3 + 1.5 * IQR)))
plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train[column_example])
plt.title(f'Boxplot de {column_example} con Outliers')
plt.xlabel(column_example)
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=X_train[~outliers][column_example])
plt.title(f'Boxplot de {column_example} sin Outliers')
plt.xlabel(column_example)
plt.show()


######################################### 2
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Normalización de Variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_columns])
X_test_scaled = scaler.transform(X_test[num_columns])

# Gráficos de distribución después de la normalización
for column, data_scaled in zip(num_columns, X_train_scaled.T):
    plt.figure(figsize=(10, 4))
    sns.histplot(data_scaled, kde=True)
    plt.title(f'Distribución de {column} (Normalizada)')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.show()

# 2. Análisis de Correlaciones entre Variables Numéricas
correlation_matrix = pd.DataFrame(X_train_scaled, columns=num_columns).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# 3. Análisis de Variables Categóricas y la Variable Objetivo
for column in cat_columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=data[column], hue=data['Stage'])
    plt.title(f'Distribución de {column} vs Variable Objetivo')
    plt.show()

# 5. Estudio de Dimensionalidad con PCA
from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulativa')
plt.title('Gráfico de Varianza Explicada por PCA')
plt.show()

def keep_classes_in_partitions(train: pd.DataFrame, test: pd.DataFrame):
	"""
	Certifica que tant el conjunt d'entrenament com el de prova continguin almenys un exemple de cada classe de totes les variables categòriques (excepte 'ID').
	En cas que no hi hagi cap exemple d'una classe en un dels dos conjunts, es mou una mostra del conjunt que en tingui a l'altre.
	Això evita problemes en cas que es faci encoding.
	"""
	categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
	categorical_cols.remove('ID')  # Excluir 'ID'

	# Comprovar que tant el conjunt d'entrenament com el de prova continguin almenys un exemple de cada classe
	for col in categorical_cols:
		train_classes = train[col].notna().unique()
		test_classes = test[col].notna().unique()

		missing_classes_test = set(train_classes) - set(test_classes)
		missing_classes_train = set(test_classes) - set(train_classes)
		
		for missing_class in missing_classes_test:
			print(f"Missing class '{missing_class}' in test set for column '{col}'")
			# Moure una mostra amb la classe faltant del conjunt de entrenament al de prova
			missing_class_index = train[train[col] == missing_class].index[0]
			test = pd.concat([test, train.loc[[missing_class_index]]])
			train.drop(missing_class_index, inplace=True)

		for missing_class in missing_classes_train:
			print(f"Missing class '{missing_class}' in train set for column '{col}'")
			# Moure una mostra amb la classe faltant del conjunt de prova al de entrenament
			missing_class_index = test[test[col] == missing_class].index[0]
			train = pd.concat([train, test.loc[[missing_class_index]]])
			test.drop(missing_class_index, inplace=True)

