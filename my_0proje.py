#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TARIMSAL ÜRETİMİN GIDA ENFLASYONUNA ETKİLERİNİN ARAŞTIRILMASI


# In[2]:


import pandas as pd 
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


# In[3]:


df_one = pd.read_excel('C:/veri/alan.xlsx')


# In[4]:


df_one.index = pd.to_datetime(df_one['Year'], format='%Y')


# In[5]:


df_one.drop('Unnamed: 3', axis=1 , inplace=True)


# In[6]:


df_one.drop('Year', axis=1 , inplace=True)


# In[7]:


df_one.head()


# In[8]:


df_one.info()


# In[9]:


df_one.columns


# In[10]:


df_one.info()


# In[11]:


df = df_one[['Total ']]


# In[12]:


print(df)


# In[13]:


fig, ax = plt.subplots(figsize = (20,12))
plt.plot(df)
plt.xlabel('Yıl (2003-2021)')
plt.ylabel('Toplam Ekilen Tahıl Arazileri (dekar)')
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# In[14]:


#pip install pmdarima


# In[15]:


from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima import auto_arima


# In[16]:


trn = df.loc[df.index < '2020-01-01']
tst = df.loc[df.index >= '2019-01-01']


# In[17]:


fig, ax = plt.subplots(figsize=(20,12))
plt.plot(trn)
plt.xlabel('Yıl (2003-2021)')
plt.ylabel('Toplam Ekilen Tahıl Arazileri (dekar)')
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# In[18]:


# ARIMA tekniğini kullanabilmek için verinin "durağan olmaması" gerekir (non-stationary).
trn_acf = plot_acf(trn, lags=16)


# In[19]:


# Doğrusal Regresyon (Linear Regression)
lr = sm.OLS(endog=trn["Total "],exog=sm.add_constant(np.arange(1,1+trn.shape[0]))).fit()
print(lr.summary)


# In[20]:


y_hat = lr.fittedvalues


# In[21]:


y_ci = lr.get_prediction().conf_int(alpha=0.05)


# In[22]:


fig, ax = plt.subplots(figsize=(20,12))
plt.xlabel('Yıl: 2003-2019')
plt.ylabel('Toplam Ekilen Tahıl Arazileri (dekar)')
plt.title('Yıllık Ekilen Alan')
plt.plot(trn, color='green', label='Eğitim verisi')
plt.plot(y_hat, color='blue', label='Doğrusal regresyon eğrisi')
plt.fill_between(y_hat.index, y_ci[:,0],y_ci[:,1], color='lightblue', alpha=0.5, label='%95 güven aralığı')
plt.legend(bbox_to_anchor=(1.05, 1))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# In[23]:


#ARIMA Modeli


# In[24]:


auto_arima_model = auto_arima(trn, with_intercept=False, initialization='approximate_diffuse')

print(auto_arima_model.summary())


# In[25]:


auto_arima_model.order


# In[26]:


# TEST MODEL
# Eğitim verisi kullanarak en iyi modeli fit
auto_arima_model.fit(trn)


# In[27]:


arima_tahmin = auto_arima_model.predict(n_periods=6, alpha=0.05, return_conf_int=True)


# In[28]:


y_tahmin = pd.Series(arima_tahmin[0], index=tst.index)


# In[29]:


y_tahmin_lb, y_tahmin_up = arima_tahmin[1][:,0], arima_tahmin[1][:,1]


# In[30]:


fig, ax = plt.subplots(figsize=(20,12))
plt.xlabel("Yıllar: 2003-2024")
plt.ylabel('Toplam Ekilen Tahıl Arazileri (dekar)')
plt.title('Yıllık Toplam Ekilen Arazi + Tahmin')
plt.fill_between(tst.index, y_tahmin_lb, y_tahmin_up, color='lightblue', alpha=0.5, label='%95 güven aralığı')
plt.plot(trn, color='black', label='Eğitim verisi -mevcut-')
plt.plot(y_tahmin, color='red', label='Tahmin')
plt.legend(bbox_to_anchor=(1.05, 1))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# In[31]:


y_tahmin


# In[32]:


#Yıllık Tahıl Üretim


# In[33]:


df_two = pd.read_excel('C:/veri/uretim_ton.xlsx')


# In[34]:


df_two.index = pd.to_datetime(df_two['Date'], format='%Y')


# In[35]:


df_two.drop('Date', axis=1 , inplace=True)


# In[36]:


df_two.head()


# In[37]:


df_two.columns


# In[38]:


df_two.info()


# In[39]:


data = df_two[['Total ']]


# In[40]:


trn_1 = data.loc[data.index < '2021-01-01']
tst_1 = data.loc[data.index >= '2020-01-01']


# In[41]:


fig, ax = plt.subplots()
plt.plot(trn_1)
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# In[42]:


lr_2 = sm.OLS(endog=trn_1["Total "],exog=sm.add_constant(np.arange(1,1+trn_1.shape[0]))).fit()
print(lr_2.summary)


# In[43]:


y_hat_2 = lr_2.fittedvalues


# In[44]:


y_ci_2 = lr_2.get_prediction().conf_int(alpha=0.05)


# In[45]:


fig, ax = plt.subplots()
plt.xlabel('Yıl: 2003-2019')
plt.ylabel('Toplam Tahıl Üretimi (Ton)')
plt.title('Yıllık Toplam Tahıl Üretimi')
plt.plot(trn_1, color='green', label='Eğitim verisi')
plt.plot(y_hat_2, color='blue', label='Doğrusal regresyon eğrisi')
plt.fill_between(y_hat_2.index, y_ci_2[:,0],y_ci_2[:,1], color='lightblue', alpha=0.5, label='%95 güven aralığı')
plt.legend(bbox_to_anchor=(1.05, 1))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# In[46]:


auto_arima_model_2 = auto_arima(trn_1, with_intercept=False)
print(auto_arima_model.summary())


# In[47]:


auto_arima_model_2.order


# In[48]:


auto_arima_model_2.fit(trn_1)


# In[49]:


arima_tahmin_2 = auto_arima_model_2.predict(n_periods=5, alpha=0.05, return_conf_int=True)


# In[50]:


y_tahmin_2 = pd.Series(arima_tahmin_2[0], index=tst_1.index)


# In[51]:


y_tahmin_lb_2, y_tahmin_up_2 = arima_tahmin_2[1][:,0], arima_tahmin_2[1][:,1]


# In[52]:


fig, ax = plt.subplots()
plt.xlabel("Yıllar: 2003-2024")
plt.ylabel('Toplam Üretilen Tahıl Arazileri (ton)')
plt.title('Yıllık Toplam Üretim + Tahmin')
plt.fill_between(tst_1.index, y_tahmin_lb_2, y_tahmin_up_2, color='lightblue', alpha=0.5, label='%95 güven aralığı')
plt.plot(trn_1, color='black', label='Eğitim verisi -mevcut-')
plt.plot(y_tahmin_2, color='red', label='Tahmin')
plt.legend(bbox_to_anchor=(1.05, 1))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))


# In[53]:


y_tahmin_2


# In[54]:


# Tahıl ve Ekmek Enflasyon Tahmini - Prophet


# In[55]:


import pandas as pd 
import numpy as np


# In[56]:


df = pd.read_excel('C:/veri/Enflasyon.xlsx')


# In[58]:


df['Tarih'] = pd.date_range("2003-01-01", periods=232, freq="M")


# In[60]:


df.columns


# In[61]:


df.info()


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns


# In[63]:


ax = df.set_index('Ekmek ve tahıllar').plot(figsize=(20, 12))
ax.set_ylabel('Yıllar')
ax.set_xlabel('Enflasyon')
plt.title('Yıllara Göre Ekmek ve Tahıl Grubundaki Enflasyon Oranı')
plt.show()


# In[64]:


#Prophet Kurulum


# In[65]:


#pip install fbprophet
#pip install plotly


# In[66]:


from prophet import Prophet


# In[67]:


df_2 = df.rename(columns={'Tarih': 'ds' ,
                       'Ekmek ve tahıllar' : 'y'}) 


# In[85]:


ax = df_2.set_index('ds').plot(figsize=(20, 12))
ax.set_ylabel('Aylık Ekmek ve Tahıl Grubu Enflasyonu')
ax.set_xlabel('Tarih')

plt.show()


# In[86]:


p_model = Prophet()
p_model.fit(df_2)


# In[87]:


future_dates = p_model.make_future_dataframe(periods=700)
forecast =p_model.predict(future_dates)


# In[88]:


figure = p_model.plot_components(forecast)


# In[89]:


forecast_one = forecast['ds']
forecast_two = forecast['yhat']

forecast_one = pd.concat([forecast_one,forecast_two], axis=1)

mask = (forecast_one['ds'] > "2022-04-30") & (forecast_one['ds'] <= "24-01-31")
forecastedvalues = forecast_one.loc[mask]

mask = (forecast_one['ds'] > "2003-01-31") & (forecast_one['ds'] <= "2022-04-30")
forecast_one = forecast_one.loc[mask]


# In[92]:


ig, ax1 = plt.subplots(figsize=(20, 12))
ax1.plot(forecast_one.set_index('ds'), color='g')
ax1.plot(forecastedvalues.set_index('ds'), color='r')
ax1.set_ylabel('Enflasyon')
ax1.set_xlabel('Tarih')
print("Kırmızı = Tahminler , Blue = geçmiş veriler")


# In[93]:


# Veriler arası ilişkiyi daha iyi anlamak için görselleştirelim :


# In[116]:


#ENFLASYON


# In[217]:


enf_veri = pd.concat([forecast_one,forecastedvalues])


# In[135]:


#EKİLEN ALAN


# In[83]:


dekar_veri = pd.read_excel('C:/veri/tahmin_alan.xlsx')


# In[122]:


# ÜRETİM (TON)


# In[82]:


ton_df = pd.read_excel('C:/veri/uretim_2.xlsx')


# In[597]:


plt.figure(1)
plt.subplot(211)
plt.title('Enflasyon')
plt.plot(enf_veri['yhat'])
plt.subplot(212)
plt.title('Üretim')
plt.plot(ton_df['Total '])


# In[598]:


plt.figure(1)
plt.subplot(211)
plt.title('Enflasyon')
plt.plot(enf_veri['yhat'])
plt.subplot(212)
plt.title('Ekilen Alan')
plt.plot(dekar_veri['Total '])


# In[607]:


plt.subplot(1,2,1)
plt.title('ENFLASYON')
plt.plot(enf_veri['yhat'],'r--')
plt.subplot(1,2,2)
plt.title('EKİLEN ALAN')
plt.plot(dekar_veri['Total '],'g*-')


# In[606]:


plt.subplot(1,2,1)
plt.title('ENFLASYON')
plt.plot(enf_veri['yhat'],'r--')
plt.subplot(1,2,2)
plt.title('ÜRETİM(TON)')
plt.plot(ton_df['Total '],'g*-')


# In[ ]:


#KORELASYON ANALİZİ


# In[69]:


new_enf = pd.read_excel('C:/veri/enflasyon_yil.xlsx')


# In[85]:


ton_df.drop([21], axis=0, inplace=True)


# In[87]:


dekar_veri.drop([21], axis=0, inplace=True)


# In[93]:


a = new_enf['Enflasyon']
b = ton_df['Total ']
c = dekar_veri['Total ']


# In[94]:


stats.spearmanr(a,b)


# In[95]:


stats.spearmanr(a,c)


# In[ ]:




