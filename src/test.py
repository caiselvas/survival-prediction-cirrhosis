import pandas as pd
import numpy as np

data = pd.read_csv('../assets/data/no_outliers_cirrhosis.csv', index_col=0)

data[['Drug', 'Sex', 'Edema', 'Stage']] = data[['Drug', 'Sex', 'Edema', 'Stage']].astype('category')

print(data.shape, '\n', data.head(-10))

def encode_variables(data: pd.DataFrame, save_to_csv: bool = True):
    """
    Codifica les variables categòriques que calgui per a poder-les utilitzar en els models de ML. 
    A més, guarda el mapping per a poder decodificar-les.
    Els NaNs es mantenen (en comptes de considerar-los una classe més) per poder imputar-los posteriorment.
    """
    from sklearn.preprocessing import OneHotEncoder

    global ohe_mapping

    columns_to_encode = ['Drug', 'Sex', 'Edema', 'Stage'] # Sense la variable 'Status' perquè és la target i, a més, no té valors NaN

    na_indexs_per_old_encoded_column = {col: set(data[data[col].isna()].index) for col in columns_to_encode} # Guardem els indexs dels NaNs per a cada columna a codificar
    new_encoded_columns_per_old_encoded_column = {col: set() for col in columns_to_encode} # Guardem les classes de cada columna a codificar

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    data_encoded = ohe.fit_transform(data[columns_to_encode])
    encoded_columns = ohe.get_feature_names_out(columns_to_encode)

    # Guardem el mapping per a poder decodificar les variables
    ohe_mapping = {}
    for i, col in enumerate(columns_to_encode):
        for category in ohe.categories_[i]:
            new_encoded_column_name = f"{col}_{category}"
            ohe_mapping[new_encoded_column_name] = (col, category)
            new_encoded_columns_per_old_encoded_column[col].add(new_encoded_column_name)

    encoded_df = pd.DataFrame(data_encoded, columns=encoded_columns)
    data.drop(columns_to_encode, axis=1, inplace=True)
    data = pd.concat([data, encoded_df], axis=1)

    # Tornem a posar els NaNs per poder imputar-los
    for col in columns_to_encode:
        for na_index in na_indexs_per_old_encoded_column[col]:
            for new_column in new_encoded_columns_per_old_encoded_column[col]:
                data.loc[na_index, new_column] = np.nan

    if save_to_csv:
        # Guardem el dataset
        data.to_csv('../assets/data/encoded_cirrhosis.csv')

encode_variables(data=data, save_to_csv=True)

print('\n\n ------------------------------------ \n\n')

print(data.shape, '\n', data.head(-10))