# https://www.cienciadedatos.net/documentos/py19-pca-python.html


import pandas as pd
import numpy as np

data = pd.read_csv("cantidad_entradas_segun_split_segundos.csv", index_col=0)

data["Entradas_luego_extraer_caracteristicas"] = [26 for _ in data["Division_segundos"] ]


# Dado que la cantidad de segundos a elegir en el split no afecta la cantidad de entradas, pero si afecta la cantidad de registros
# entonces inicialmente se eligen 3 segundos

def prepare_data_tensorflow(data):
    instrument_list = data.iloc[:, -11:]
    train = data.iloc[:, 1:-11]
    X = np.array(train, dtype=float)
    y = instrument_list
    return X, y

train = pd.read_csv('train.csv')
X, y = prepare_data_tensorflow(train)


from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
data_scaler=StandardScaler()
data_scaler.fit(X)


import matplotlib.pyplot as plt
def ver_posibles_reducciones_PCA(data):
    pca = PCA()
    pca.fit(data)
    X_pca = pca.transform(data)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    #plt.axvline(x=24, color='b', label='axvline - full height')
    #plt.axhline(y=0.94, color='b', label='axvline - full height')

    plt.show()


def normalizar(data):
    data_scaled = data_scaler.transform(data)  # transforma los datos a su nueva escala
    return data_scaled


ver_posibles_reducciones_PCA( normalizar(X) )

clumnas = [f"Feature{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=clumnas)
#print(df)

print('----------------------')
print('Media de cada variable')
print('----------------------')
print(df.var(axis=0))


from sklearn.pipeline import make_pipeline
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(X)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']

# Se combierte el array a dataframe para añadir nombres a los ejes.
ae = pd.DataFrame(
    data    = modelo_pca.components_,
    columns = df.columns,
    #index   = ['PC1', 'PC2', 'PC3', 'PC4']
)

print(ae.head())


















# Heatmap componentes
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
componentes = modelo_pca.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(df.columns)), df.columns)
plt.xticks(range(len(df.columns)), np.arange(modelo_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar()
plt.show()
fig.savefig("PCA_Heatmap_influencia_variables_cada_componentes.jpg")
















# Porcentaje de varianza explicada por cada componente
# ==============================================================================
print('----------------------------------------------------')
print('Porcentaje de varianza explicada por cada componente')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 30))
ax.bar(
    x      = np.arange(modelo_pca.n_components_) + 1,
    height = modelo_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(df.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 0.3) # ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada')
plt.show()
fig.savefig("PCA_porcentaje_varianza_explicada_cada_componenete.jpg")



















# Porcentaje de varianza explicada acumulada
# ==============================================================================
prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('------------------------------------------')
print('Porcentaje de varianza explicada acumulada')
print('------------------------------------------')
print(prop_varianza_acum)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
ax.plot(
    np.arange(len(df.columns)) + 1,
    prop_varianza_acum,
    marker='o'
)

for x, y in zip(np.arange(len(df.columns)) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=20
    )

ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title('Porcentaje de varianza explicada acumulada')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza acumulada')
plt.show()
fig.savefig("PCA_porcentaje_varianza_explicada_resumen.jpg")