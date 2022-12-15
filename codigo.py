#!/usr/bin/env python3



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import ccf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster



def remove_outliers(df):
    """Function to remove outliers"""

    # Calculate the Q3 and Q1
    Q3 = float(df.quantile(0.75, numeric_only=True))
    Q1 = float(df.quantile(0.25, numeric_only=True))

    # Calculate the IQR
    IQR = Q3 - Q1
    
    # Calculate the upper and lower limits of the dataframe
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    
    # Add index of rows that are outliers to a list
    indexesToDelete = []
    for j in range(len(df)):
        if df.iloc[j, 1] > upper_limit or df.iloc[j, 1] < lower_limit:
            indexesToDelete.append(j)

    # Drop from the dataframe the rows whose indexes are in the list
    df.drop(indexesToDelete, inplace=True)

    return df


def mean_filter(dataseries, window_size=5):
    """Function applies rolling mean into given dataseries"""
    
    filtered = dataseries.rolling(window_size).mean()
    return filtered


def prediction(datos,c1,c2):
    """Function to predict time series"""

    data = datos.copy()
    data[c1] = pd.to_datetime(data[c1], format='%d/%m/%Y')
    data = data.set_index(c1)
    data = data.asfreq('MS')
    data_train = data[:int(0.8 * (len(data)))]
    
    forecaster = ForecasterAutoreg(
        regressor=Ridge(),
        lags=150,
        transformer_y=StandardScaler()
    )

    forecaster.fit(y=data_train[c2].ffill())

    metrica, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=data[c2].ffill(),initial_train_size = len(data_train),
                            fixed_train_size   = False,
                            steps      = 24,
                            metric     = 'mean_absolute_error',
                            refit      = False,
                            verbose    = False)

    _, ax = plt.subplots(figsize=(12, 3.5))
    data.plot(linewidth=2, label='Test', ax=ax)
    predictions.plot(linewidth=2, label='Predición', ax=ax)
    print(f'Erro absoluto no modelo preditivo de {c2}: {metrica}')
    return ax



def main():
    """Main function"""

    # Read the data
    df_IRRA = pd.read_csv('IRRA_Santiago.csv', sep='|', index_col=False, encoding='unicode_escape', na_values=-9999)
    df_P = pd.read_csv('P_Santiago.csv', sep='|', index_col=False, encoding='unicode_escape', na_values=-9999)
    df_TM = pd.read_csv('TM_Santiago.csv', sep='|', index_col=False, encoding='unicode_escape', na_values=-9999)


    # Remove the outliers
    df_IRRA = remove_outliers(df_IRRA)
    df_P = remove_outliers(df_P)
    df_TM = remove_outliers(df_TM)


    # Calculate the mean of the data
    mean_IRRA = float(df_IRRA.mean(numeric_only=True))
    mean_P = float(df_P.mean(numeric_only=True))
    mean_TM = float(df_TM.mean(numeric_only=True))


    # Save the graphically representation of the autocorrelation function
    ts_ACF = df_IRRA.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(mean_IRRA)
    plot_acf(ts_ACF)
    plt.savefig('ACF_IRRA.png')
    plt.clf()
    
    ts_ACF = df_IRRA.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(mean_IRRA)
    plot_acf(ts_ACF)
    plt.savefig('ACF_P.png')
    plt.clf()

    ts_ACF = df_IRRA.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(mean_IRRA)
    plot_acf(ts_ACF)
    plt.savefig('ACF_TM.png')
    plt.clf()


    # Calculate and plot the first difference
    df_diff1 = df_IRRA.copy()
    df_diff1['IRRA'] = df_IRRA['IRRA'].diff()
    df_diff1['P'] = df_P['P'].diff()
    df_diff1['TM'] = df_TM['TM'].diff()


    # Save the graphically representation of the first difference
    df_diff1['IRRA'].plot(title='Primeira diferenza da irradiación solar')
    plt.savefig('diff1_IRRA.png')
    plt.clf()

    df_diff1['P'].plot(title='Primeira diferenza da presión atmosférica')
    plt.savefig('diff1_P.png')
    plt.clf()

    df_diff1['TM'].plot(title='Primeira diferenza da temperatura')
    plt.savefig('diff1_TM.png')
    plt.clf()

    # Save the graphically representation of the autocorrelation of the first difference
    ts_ACF = df_diff1[['Fecha' ,'IRRA']]
    ts_ACF = ts_ACF.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(0)
    plot_acf(ts_ACF)
    plt.savefig('ACF_diff1_IRRA.png')
    plt.clf()

    ts_ACF = df_diff1[['Fecha' ,'P']]
    ts_ACF = ts_ACF.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(0)
    plot_acf(ts_ACF)
    plt.savefig('ACF_diff1_P.png')
    plt.clf()

    ts_ACF = df_diff1[['Fecha' ,'TM']]
    ts_ACF = ts_ACF.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(0)
    plot_acf(ts_ACF)
    plt.savefig('ACF_diff1_TM.png')
    plt.clf()


    # Calculate the second difference
    df_diff2 = df_IRRA.copy()
    df_diff2['IRRA'] = df_diff1['IRRA'].diff()
    df_diff2['P'] = df_diff1['P'].diff()
    df_diff2['TM'] = df_diff1['TM'].diff()


    # Save the graphically representation of the second difference
    df_diff2['IRRA'].plot(title='Segunda diferenza da irradiación solar')
    plt.savefig('diff2_IRRA.png')
    plt.clf()

    df_diff2['P'].plot(title='Segunda diferenza da presión atmosférica')
    plt.savefig('diff2_P.png')
    plt.clf()

    df_diff2['TM'].plot(title='Segunda diferenza da temperatura')
    plt.savefig('diff2_TM.png')
    plt.clf()


    # Save the graphically representation of the autocorrelation of the second difference
    ts_ACF = df_diff2[['Fecha' ,'IRRA']]
    ts_ACF = ts_ACF.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(0)
    plot_acf(ts_ACF)
    plt.savefig('ACF_diff2_IRRA.png')
    plt.clf()

    ts_ACF = df_diff2[['Fecha' ,'P']]
    ts_ACF = ts_ACF.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(0)
    plot_acf(ts_ACF)
    plt.savefig('ACF_diff2_P.png')
    plt.clf()

    ts_ACF = df_diff2[['Fecha' ,'TM']]
    ts_ACF = ts_ACF.set_index('Fecha')
    ts_ACF = ts_ACF.fillna(0)
    plot_acf(ts_ACF)
    plt.savefig('ACF_diff2_TM.png')
    plt.clf()


    # Decompose the time series into trend and seasonal components and represent linear model of the trend
    ts_decompose = df_IRRA.set_index('Fecha')
    ts_decompose = ts_decompose.fillna(mean_IRRA)
    decomposition = seasonal_decompose(ts_decompose, period=365)
    fig = plt.figure()
    fig.set_figwidth(15)
    decomposition.trend.plot(title='Tendencia da irradiación solar')
    plt.savefig('trend_IRRA.png')
    plt.clf()
    fig = plt.figure()
    fig.set_figwidth(15)
    decomposition.seasonal.plot(title='Estacionalidade da irradiación solar')
    plt.savefig('seasonal_IRRA.png')
    plt.clf()

    trend_not_null = decomposition.trend.fillna(mean_P)
    enumerated_list = [i for i in range(len(df_IRRA))]
    enumerated_list = np.array(enumerated_list).reshape(-1, 1)
    linear_model = LinearRegression().fit(enumerated_list, trend_not_null)
    decomposition.trend.plot()
    plt.plot(linear_model.predict(enumerated_list), color='red')
    plt.title('Regresión linear da tendencia da irradiación solar')
    plt.xlabel('Fecha')
    plt.ylabel('IRRA')
    plt.savefig('linear_regression_IRRA.png')

    ts_decompose = df_P.set_index('Fecha')
    ts_decompose = ts_decompose.fillna(mean_P)
    decomposition = seasonal_decompose(ts_decompose, period=365)
    fig = plt.figure()
    fig.set_figwidth(15)
    decomposition.trend.plot(title='Tendencia da presión atmosférica')
    plt.savefig('trend_P.png')
    plt.clf()
    fig = plt.figure()
    fig.set_figwidth(15)
    decomposition.seasonal.plot(title='Estacionalidade da presión atmosférica')
    plt.savefig('seasonal_P.png')
    plt.clf()

    trend_not_null = decomposition.trend.fillna(mean_P)
    enumerated_list = [i for i in range(len(df_P))]
    enumerated_list = np.array(enumerated_list).reshape(-1, 1)
    linear_model = LinearRegression().fit(enumerated_list, trend_not_null)
    decomposition.trend.plot()
    plt.plot(linear_model.predict(enumerated_list), color='red')
    plt.title('Regresión linear da tendencia da presión atmosférica')
    plt.xlabel('Fecha')
    plt.ylabel('P')
    plt.savefig('linear_regression_P.png')

    ts_decompose = df_TM.set_index('Fecha')
    ts_decompose = ts_decompose.fillna(mean_TM)
    decomposition = seasonal_decompose(ts_decompose, period=365)
    fig = plt.figure()
    fig.set_figwidth(15)
    decomposition.trend.plot(title='Tendencia da temperatura')
    plt.savefig('trend_TM.png')
    plt.clf()
    fig = plt.figure()
    fig.set_figwidth(15)
    decomposition.seasonal.plot(title='Estacionalidade da temperatura')
    plt.savefig('seasonal_TM.png')
    plt.clf()

    trend_not_null = decomposition.trend.fillna(mean_TM)
    enumerated_list = [i for i in range(len(df_TM))]
    enumerated_list = np.array(enumerated_list).reshape(-1, 1)
    linear_model = LinearRegression().fit(enumerated_list, trend_not_null)
    decomposition.trend.plot()
    plt.plot(linear_model.predict(enumerated_list), color='red')
    plt.title('Regresión linear da tendencia da temperatura')
    plt.xlabel('Fecha')
    plt.ylabel('TM')
    plt.savefig('linear_regression_TM.png')
    plt.clf()
    

    # Cross correlation between the time series
    arr_IRRA = df_IRRA.iloc[:, 1][~np.isnan(df_IRRA.iloc[:, 1])]
    arr_TM = df_TM.iloc[:, 1][~np.isnan(df_TM.iloc[:, 1])]
    arr_P = df_P.iloc[:, 1][~np.isnan(df_P.iloc[:, 1])]
    plt.plot(ccf(np.array(arr_IRRA), np.array(arr_TM), adjusted=False), "bo")
    plt.savefig("CCF_IRRA_TM.png")
    plt.clf()
    plt.plot(ccf(np.array(arr_IRRA), np.array(arr_P), adjusted=False), "bo")
    plt.savefig("CCF_IRRA_P.png")
    plt.clf()
    plt.plot(ccf(np.array(arr_TM), np.array(arr_P), adjusted=False), "bo")
    plt.savefig("CCF_TM_P.png")
    plt.clf()


    # Filter the time series with a moving average
    df_IRRA = df_IRRA.fillna(mean_IRRA)
    mov_avg_IRRA = mean_filter(df_IRRA.IRRA, 10)
    plt.figure(figsize=(15, 6))
    plt.scatter(df_IRRA.index, df_IRRA.IRRA)
    plt.plot(mov_avg_IRRA, color = '#ffa500')
    plt.legend(["Orixinal", "Filtrado"])
    plt.savefig("mean_filter_IRRA.png")
    plt.clf()

    df_P = df_P.fillna(mean_P)
    mov_avg_P = mean_filter(df_P.P, 10)
    plt.figure(figsize=(15, 6))
    plt.scatter(df_P.index, df_P.P)
    plt.plot(mov_avg_P, color = '#ffa500')
    plt.legend(["Orixinal", "Filtrado"])
    plt.savefig("mean_filter_P.png")
    plt.clf()

    df_TM = df_TM.fillna(mean_TM)
    mov_avg_TM = mean_filter(df_TM.TM, 10)
    plt.figure(figsize=(15, 6))
    plt.scatter(df_TM.index, df_TM.TM)
    plt.plot(mov_avg_TM, color = '#ffa500')
    plt.legend(["Orixinal", "Filtrado"])
    plt.savefig("mean_filter_TM.png")
    plt.clf()


    # Prediction of the time series
    train = df_IRRA[:int(0.8 * (len(df_IRRA)))]
    test = df_IRRA[int(0.8 * (len(df_IRRA))):]
    fig, _ = plt.subplots(1, 1, figsize=(15, 5))
    plt.plot(train.index, train.IRRA, color = "black")
    plt.plot(test.index, test.IRRA, color = "red")
    plt.ylabel('IRRA')
    plt.xlabel('Fecha')
    plt.xticks(rotation=45)
    plt.title("Conxuntos de entrenamento e test para a irradiación solar")
    plt.savefig("traintest_IRRA.png")
    plt.clf()
    ax = prediction(df_IRRA, df_IRRA.columns[0], df_IRRA.columns[1])
    ax.legend()
    ax.set_title('Modelo predito respecto aos datos para a irradiación solar')
    plt.savefig("pred_IRRA.png")
    plt.clf()

    train = df_P[:int(0.8 * (len(df_P)))]
    test = df_P[int(0.8 * (len(df_P))):]
    fig, _ = plt.subplots(1, 1, figsize=(15, 5))
    plt.plot(train.index, train.P, color = "black")
    plt.plot(test.index, test.P, color = "red")
    plt.ylabel('P')
    plt.xlabel('Fecha')
    plt.xticks(rotation=45)
    plt.title("Conxuntos de entrenamento e test para a presión atmosférica")
    plt.savefig("traintest_P.png")
    plt.clf()
    ax = prediction(df_P, df_P.columns[0], df_P.columns[1])
    ax.legend()
    ax.set_title('Modelo predito respecto aos datos para a presión atmosférica')
    plt.savefig("pred_P.png")
    plt.clf()

    train = df_TM[:int(0.8 * (len(df_TM)))]
    test = df_TM[int(0.8 * (len(df_TM))):]
    fig, _ = plt.subplots(1, 1, figsize=(15, 5))
    plt.plot(train.index, train.TM, color = "black")
    plt.plot(test.index, test.TM, color = "red")
    plt.ylabel('TM')
    plt.xlabel('Fecha')
    plt.xticks(rotation=45)
    plt.title("Conxuntos de entrenamento e test para a temperatura")
    plt.savefig("traintest_TM.png")
    plt.clf()
    ax = prediction(df_TM, df_TM.columns[0], df_TM.columns[1])
    ax.legend()
    ax.set_title('Modelo predito respecto aos datos para a temperatura')
    plt.savefig("pred_TM.png")
    plt.clf()



# START OF EXECUTION
if __name__ == '__main__':
    plt.rcParams.update({'figure.max_open_warning': 0})
    print()
    main()
    print()