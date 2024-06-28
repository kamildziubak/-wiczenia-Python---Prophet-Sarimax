plik = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
import numpy as np
import pandas as pd
import prophet
import matplotlib.pyplot as plt
from prophet import Prophet
df = pd.read_csv(plik)

df.head()

df['sma50'] = df['Sales'].rolling(50).mean()

df.dropna(inplace=True)

df.head()

df_sma=df[['Month', 'sma50']]

df_sma.head()

df_sma.columns=['ds', 'y']

df_sma.head()

model_sma = Prophet()
model_sma.fit(df_sma)

future = list()
for i in range(1, 13):
  date = '1969-%02d' % i
  future.append(date)
for i in range(1, 13):
  date = '1970-%02d' % i
  future.append(date)
future=pd.DataFrame(future)
future.columns=['ds']
future['ds']= pd.to_datetime(future['ds'])

prediction_sma = model_sma.predict(future)
model_sma.plot(prediction_sma)
plt.show()

# Zadanie domowe
# Proszę użyć metody SARIMAX (moduł statsmodels) w celu uzyskania
# prognoz, dotyczących inflacji w Polsce


from statsmodels.tsa.statespace.sarimax import SARIMAX

inflation = [['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01',
         '2021-11-01', '2021-12-01',
         '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01',
         '2022-11-01', '2022-12-01',
         '2023-01-01', '2023-02-01', '2023-03-01'],
        [2.6, 2.4, 3.2, 4.3, 4.7, 4.4, 5.0, 5.5, 6.9, 6.8, 7.8, 8.6,
         9.4, 8.5, 11.0, 12.4, 13.9, 15.5, 15.6, 16.1, 17.2, 17.9, 17.5, 16.6,
         16.6, 18.4, 16.1]]
inflation=pd.DataFrame(np.array(inflation).T)

inflation.columns=['x', 'y']
inflation['x'] = pd.to_datetime(inflation['x'])
inflation['y'] = pd.to_numeric(inflation['y'])
inflation.info()
inflation.head()

np_inflation = inflation.values
np_inflation

new_inflation=list()
for i in range(0,27):
  new_inflation.append([i, np_inflation[i][1]])
new_inflation = np.array(new_inflation)

model=SARIMAX(new_inflation[:,1])

prediction=model.fit()

forecast = prediction.get_forecast(step=5)
forecast

forecast.predicted_mean


