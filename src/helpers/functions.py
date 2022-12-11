import pandas as pd
import numpy as np

#Create helper DataFrame
def create_df(data, important_columns, flag):
    if flag:
        data['host_response_rate'] = pd.to_numeric(data['host_response_rate'].str.replace("%",''))
        data['instant_bookable'] = np.where(data['instant_bookable'] == 't', True, False)
    df = data[important_columns]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df

#Create Feature Engineering Function:
def get_dummies(df):
    df = pd.get_dummies(df, columns=['instant_bookable'], drop_first=True)
    df = pd.get_dummies(df, columns=['room_type','property_type'], drop_first=True)
    df = pd.get_dummies(df, columns=['neighbourhood_cleansed','neighbourhood_group_cleansed'], drop_first=True)
    df = pd.get_dummies(df, columns=['host_response_time'], drop_first=True)
    df['host_location_in_ny'] = np.where(df['host_location'].str.contains("NY|New York"), True, False)
    df.drop("host_location", axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['host_location_in_ny'], drop_first=True)
    return df

#Convert User Input Into Array
def convert_input(data, important_columns, input_array, not_important_features):
    df = create_df(data, important_columns, False)
    df2 = pd.DataFrame(np.vstack((df.values, input_array)), columns=df.columns)
    df2 = get_dummies(df2)
    borough = f"neighbourhood_cleansed_{input_array[3]}"
    neighborhood = f"neighbourhood_group_cleansed_{input_array[4]}"
    not_important_features.append(borough)
    not_important_features.append(neighborhood)
    df2 = df2.drop(not_important_features, axis=1)

    # for c in df2.columns:
    #     if c not in important_features:
    #         print(c)

    input_data = df2.iloc[-1].values
    return input_data

#Making prediction: get_prediction
def get_prediction(model, data, important_columns, not_important_features, input_array):
    input_data = convert_input(data, important_columns, input_array, not_important_features)
    prediction = model.predict([input_data])
    return prediction

