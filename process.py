import kaggle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import fbprophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from plotly.offline import plot, iplot
import plotly.io as pio
from statsmodels.tsa.arima_model import ARIMA
import datetime
import matplotlib.colors as mcolors
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import operator 
import re
#plt.style.use('fivethirtyeight')
import os
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go
import cufflinks
cufflinks.go_offline(connected=True)
import folium
from IPython.display import Image
sns.set(style="darkgrid", palette="pastel", color_codes=True)
sns.set_context("paper")
pio.templates.default = "seaborn"
from plotly.subplots import make_subplots
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from folium import plugins
from keras.models import Sequential
from keras.layers import LSTM, Dense



def prepare_datasets(date):
    download_path = "/home/eric/Desktop/csir_srtp/downloaded_dtsets"
    github_dtset = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
    
    """##
    temp_loc = '/home/eric/Desktop/csir_srtp/temp_folder/'
    world_confirmed = pd.read_csv('{}/time_series_covid19_confirmed_global.csv'.format(temp_loc))
    world_deaths = pd.read_csv('{}/time_series_covid19_deaths_global.csv'.format(temp_loc))
    world_recovered = pd.read_csv('{}/time_series_covid19_recovered_global.csv'.format(temp_loc))
    latest_data = pd.read_csv('{}/latest_data.csv'.format(temp_loc))
    """
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("sudalairajkumar/covid19-in-india", path=download_path, unzip=True)
    
    india_statewise = pd.read_csv('{}/covid_19_india.csv'.format(download_path), header = 0)
    
    world_confirmed = pd.read_csv('{}/time_series_covid19_confirmed_global.csv'.format(github_dtset))
    world_deaths = pd.read_csv('{}/time_series_covid19_deaths_global.csv'.format(github_dtset))
    world_recovered = pd.read_csv('{}/time_series_covid19_recovered_global.csv'.format(github_dtset))
    latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'.format(date))
    
    df1 = world_confirmed.groupby('Country/Region').sum().reset_index()
    df2 = world_deaths.groupby('Country/Region').sum().reset_index()
    df3 = world_recovered.groupby('Country/Region').sum().reset_index()
    
    dates = list(world_confirmed.columns[4:])
    dates = list(pd.to_datetime(dates))
    dates_india = dates[8:]
    
    return india_statewise, world_confirmed, world_deaths, world_recovered, latest_data, df1, df2, df3, dates, dates_india


def plot_covid_india_graph():
    covid_19_india=pd.read_csv('/home/eric/Desktop/csir_srtp/downloaded_dtsets/covid_19_india.csv',parse_dates=['Date'], dayfirst=True)
    df1=covid_19_india.groupby('Date')[['Cured','Deaths','Confirmed']].sum()
    plt.figure(figsize=(20,10))
    plt.style.use('ggplot')
    plt.title('Observed Cases',fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Date',fontsize=15)
    plt.ylabel('Number of cases',fontsize=15)
    plt.plot(df1.index,df1['Confirmed'],linewidth=3,label='Confirmed',color='black')
    plt.plot(df1.index,df1['Cured'],linewidth=3,label='Cured',color='green')
    plt.plot(df1.index,df1['Deaths'],linewidth=3,label='Death',color='red')
    plt.legend(fontsize=20)
    plt.show()
    
    
def plot_fbprophet_forecasting(df, dates, days_in_future, feature):
    k = df[df['Country/Region']=='India'].loc[:,'1/22/20':]
    india_values = k.values.tolist()[0] 
    data = pd.DataFrame(columns = ['ds','y'])
    data['ds'] = dates
    data['y'] = india_values
    pd.set_option('float_format', '{:f}'.format)
    model_prophet=fbprophet.Prophet()
    model_prophet.fit(data)
    future=model_prophet.make_future_dataframe(periods=days_in_future)
    prop_forecast=model_prophet.predict(future)
    forecast = prop_forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days_in_future)
    print(forecast)

    fig = plot_plotly(model_prophet, prop_forecast)
    fig = model_prophet.plot(prop_forecast,xlabel='Date',ylabel='{}'.format(feature))
    fig.show()
    
    
def preproc_for_polyreg(confirmed_cases, deaths_reported, recovered_cases, latest_data):
    confirmed = confirmed_cases.drop(['Province/State','Country/Region','Lat','Long'] , axis = 1)
    deaths = deaths_reported.drop(['Province/State','Country/Region','Lat','Long'] , axis = 1)
    recoveries = recovered_cases.drop(['Province/State','Country/Region','Lat','Long'] , axis = 1)
    
    # calculating cases date wise
    dates = confirmed.keys()

    # Calculating Totle cases at world level
    world_cases = []
    total_deaths = [] 
    total_recovered = [] 
    total_active = []
    mortality_rate = []
    recovery_rate = []

    # Cases at India level
    india_cases = []
    india_deaths = []
    india_recoveries = []
    india_active = []
    
    for i in dates:
        confirmed_sum = confirmed[i].sum()
        death_sum = deaths[i].sum()
        recovered_sum = recoveries[i].sum()
        
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        total_recovered.append(recovered_sum)
        total_active.append(confirmed_sum-death_sum-recovered_sum)
        
        mortality_rate.append(death_sum/confirmed_sum)
        recovery_rate.append(recovered_sum/confirmed_sum)

        india_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='India'][i].sum())
        
        india_deaths.append(deaths_reported[deaths_reported['Country/Region']=='India'][i].sum())

        india_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='India'][i].sum())
    
    for i in range(len(dates)):
        india_active.append(india_cases[i] - india_deaths[i] - india_recoveries[i])
        
    # Calculating increament in confirmed cases in each day
    world_daily_increase = daily_increase(world_cases)
    india_daily_increase = daily_increase(india_cases)
    
    world_daily_death = daily_increase(total_deaths)
    india_daily_death = daily_increase(india_deaths)
    
    world_daily_recovery = daily_increase(total_recovered)
    india_daily_recovery = daily_increase(india_recoveries)
    
    unique_countries = latest_data['Country_Region'].unique().tolist()
    
    confirmed_by_country = []
    death_by_country = [] 
    active_by_country = []
    recovery_by_country = []
    mortality_rate_by_country = [] 

    no_cases = []
    for i in unique_countries:
        cases = latest_data[latest_data['Country_Region']==i]['Confirmed'].sum()
        if cases > 0:
            confirmed_by_country.append(cases)
        else:
            no_cases.append(i)
            
    for i in no_cases:
        unique_countries.remove(i)
        
    # sort countries by the number of confirmed cases
    unique_countries = [k for k, v in sorted(zip(unique_countries, confirmed_by_country), key=operator.itemgetter(1), reverse=True)]

    for i in range(len(unique_countries)):
        confirmed_by_country[i] = latest_data[latest_data['Country_Region']==unique_countries[i]]['Confirmed'].sum()
        death_by_country.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Deaths'].sum())
        recovery_by_country.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Recovered'].sum())
        active_by_country.append(confirmed_by_country[i] - death_by_country[i] - recovery_by_country[i])
        mortality_rate_by_country.append(death_by_country[i]/confirmed_by_country[i])
        
    country_df = pd.DataFrame({'Country Name': unique_countries,
                           'Number of Confirmed Cases': confirmed_by_country,
                          'Number of Deaths': death_by_country,
                           'Number of Recoveries' : recovery_by_country, 
                          'Number of Active Cases' : active_by_country,
                          'Mortality Rate': mortality_rate_by_country})
    India_confirmed = latest_data[latest_data['Country_Region']=='India']['Confirmed'].sum()
    
    Starting_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    india_cases = np.array(india_cases).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    total_recovered = np.array(total_recovered).reshape(-1, 1)
    
    return dates, Starting_1_22, india_cases, india_active, India_confirmed, india_recoveries, india_deaths, world_cases

def polyreg_model_fit(days_in_future, dates, Starting_1_22, india_cases, degree):
    future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forecast[:-(days_in_future)]
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forecast_dates = []
    for i in range(len(future_forecast)):
        future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
        
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(Starting_1_22, india_cases, test_size=0.30, random_state = 0 , shuffle=False)
    
    plt.figure(figsize=(15,9))
    poly = PolynomialFeatures(degree)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forecast = poly.fit_transform(future_forecast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    y_poly_pred = linear_model.predict(poly_X_test_confirmed)

    print("degree %d" % degree)
    print('MAE :', '%.4f' % mean_absolute_error(y_test_confirmed,y_poly_pred))
    print('MSE :','%.4f' % mean_squared_error(y_test_confirmed,y_poly_pred))
    print('RSME :','%.4f' % np.sqrt(mean_squared_error(y_test_confirmed,y_poly_pred)))
    print('r2_test :','%.4f' % r2_score(y_test_confirmed,y_poly_pred))
    print('model_score :','%.4f' % linear_model.score(poly_X_test_confirmed, y_test_confirmed))

    plt.plot(y_test_confirmed)
    plt.plot(y_poly_pred)
    plt.legend(['Test Data', 'Polynomial Regression Predictions'])
    plt.xlabel("Dates")
    plt.ylabel("Predicted Cases")
    plt.title("Variance Explained with Varying Polynomial")
    plt.show()
    
def forecast_polyreg(days_in_future, dates, Starting_1_22, india_cases):
    future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forecast[:-(days_in_future)]
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(Starting_1_22, india_cases, test_size=0.30, random_state = 0 , shuffle=False)
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forecast_dates = []
    for i in range(len(future_forecast)):
        future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
        
    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forecast = poly.fit_transform(future_forecast)
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    y_poly_pred = linear_model.predict(poly_X_test_confirmed)
    
    linear_pred = linear_model.predict(poly_future_forecast)
    linear_pred = linear_pred.reshape(1,-1)[0]
    poly_df = pd.DataFrame({'Date': future_forecast_dates[-(days_in_future):], 
                            'Predicted number of Confirmed Cases Worldwide': np.round(linear_pred[-(days_in_future):])})
    print(poly_df)
    
    plt.figure(figsize=(15, 9))
    plt.plot(adjusted_dates, india_cases)
    plt.plot(future_forecast, linear_pred, linestyle='dashed', color='red')
    plt.title('Number of Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=20)
    plt.ylabel('Number of Cases', size=20)
    plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 15})
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()

def plot_covid_india_agegraph():
    agegroup = pd.read_csv('/home/eric/Desktop/csir_srtp/downloaded_dtsets/AgeGroupDetails.csv')
    perc=[]
    for i in agegroup['Percentage']:
        per=float(re.findall("\d+\.\d+",i)[0])
        perc.append(per)
    agegroup['Percentage']=perc
    plt.figure(figsize=(20,10))
    plt.title('Percentage of cases in the age group',fontsize=20)
    plt.pie(agegroup['Percentage'],autopct='%1.1f%%')
    plt.legend(agegroup['AgeGroup'],loc='best',title='Age Group')
    plt.show()

def plot_covid_india_testsgraph():
    state_testing = pd.read_csv('/home/eric/Desktop/csir_srtp/downloaded_dtsets/StatewiseTestingDetails.csv')
    testing=state_testing.groupby('State').sum().reset_index()
    state_testing['Negative']=state_testing['TotalSamples']-state_testing['Positive']
    testing=testing.sort_values(['TotalSamples'], ascending=True)
    fig = px.bar(testing, x="State",y="TotalSamples", orientation='v',height=800,title='Testing statewise insight',color_discrete_sequence=["blue"])
    fig.show()
    
    testing=state_testing.groupby('State').sum().reset_index()
    testing=testing.sort_values(['Positive'], ascending=True)
    fig = px.bar(testing, x="State",y="Positive", orientation='v',height=800,title='Testing statewise insight',color_discrete_sequence=["green"])
    fig.show()
    
    testing=state_testing.groupby('State').sum().reset_index()
    testing=testing.sort_values(['Negative'], ascending=True)
    fig = px.bar(testing, x="State",y="Negative", orientation='v',height=800,title='Testing statewise insight',color_discrete_sequence=["red"])
    fig.show()

def plot_covid_india_bedsgraph():
    sns.set(font_scale=0.6)
    hospital_beds = pd.read_csv('/home/eric/Desktop/csir_srtp/downloaded_dtsets/HospitalBedsIndia.csv')
    hospital_beds = hospital_beds.drop(index=hospital_beds.shape[0]-1, axis=0)
    plt.figure(figsize=(15,10))
    
    plt.subplot(2,2,1)
    hospital_beds=hospital_beds.sort_values('NumUrbanHospitals_NHP18', ascending= False)
    sns.barplot(data=hospital_beds,y='State/UT',x='NumUrbanHospitals_NHP18',color=sns.color_palette('RdBu')[0])
    plt.title('Urban Hospitals per states')
    plt.xlabel(' ')
    plt.ylabel(' ')
    for i in range(hospital_beds.shape[0]):
        count = hospital_beds.iloc[i]['NumUrbanHospitals_NHP18']
        plt.text(count+10,i,count,ha='center',va='center', fontsize=6)


    plt.subplot(2,2,2)
    hospital_beds=hospital_beds.sort_values('NumRuralHospitals_NHP18', ascending= False)
    sns.barplot(data=hospital_beds,y='State/UT',x='NumRuralHospitals_NHP18',color=sns.color_palette('RdBu')[1])
    plt.title('Rural Hospitals per states')
    plt.xlabel(' ')
    plt.ylabel(' ')
    for i in range(hospital_beds.shape[0]):
        count = hospital_beds.iloc[i]['NumRuralHospitals_NHP18']
        plt.text(count+100,i,count,ha='center',va='center', fontsize=6)

    plt.subplot(2,2,3)
    hospitalBeds=hospital_beds.sort_values('NumUrbanBeds_NHP18', ascending= False)
    sns.barplot(data=hospitalBeds,y='State/UT',x='NumUrbanBeds_NHP18',color=sns.color_palette('RdBu')[5])
    plt.title('Urban Beds per states')
    plt.xlabel(' ')
    plt.ylabel(' ')
    for i in range(hospitalBeds.shape[0]):
        count = hospitalBeds.iloc[i]['NumUrbanBeds_NHP18']
        plt.text(count+1500,i,count,ha='center',va='center', fontsize=6)

    plt.subplot(2,2,4)
    hospitalBeds=hospitalBeds.sort_values('NumRuralBeds_NHP18', ascending= False)
    sns.barplot(data=hospitalBeds,y='State/UT',x='NumRuralBeds_NHP18',color=sns.color_palette('RdBu')[4])
    plt.title('Rural Beds per states')
    plt.xlabel(' ')
    plt.ylabel(' ')
    for i in range(hospitalBeds.shape[0]):
        count = hospitalBeds.iloc[i]['NumRuralBeds_NHP18']
        plt.text(count+1500,i,count,ha='center',va='center', fontsize=6)

    plt.show()
    plt.tight_layout()

def plot_outside_vs_in(latest_data):
    unique_countries = latest_data['Country_Region'].unique().tolist()
    confirmed_by_country = []
    no_cases = []
    for i in unique_countries:
        cases = latest_data[latest_data['Country_Region']==i]['Confirmed'].sum()
        if cases > 0:
            confirmed_by_country.append(cases)
        else:
            no_cases.append(i)
            
    for i in no_cases:
        unique_countries.remove(i)

    unique_countries = [k for k, v in sorted(zip(unique_countries, confirmed_by_country), key=operator.itemgetter(1), reverse=True)]

    for i in range(len(unique_countries)):
        confirmed_by_country[i] = latest_data[latest_data['Country_Region']==unique_countries[i]]['Confirmed'].sum()
    
    India_confirmed = latest_data[latest_data['Country_Region']=='India']['Confirmed'].sum()
    outside_India_confirmed = np.sum(confirmed_by_country) - India_confirmed
    plt.figure(figsize=(14,7))
    plt.barh('Confirmed Cases In INDIA', India_confirmed)
    plt.barh('Confirmed Cases Outside INDIA ', outside_India_confirmed)
    plt.title('Number of Coronavirus Confirmed Cases', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()

def plot_country_wise(dates, days_in_future, y1, y2, y3, y4, country):
    future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    x = future_forecast[:-(days_in_future)]
    plt.figure(num='Dates are from 22-01-2020', figsize=(10,10))
    
    plt.subplot(2,2,1)
    plt.plot(x, y1)
    plt.title('{} Number of Coronavirus Cases Over Time'.format(country), size=10)
    plt.xlabel(' ', size=1)
    plt.ylabel(' ', size=1)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.show()

    plt.subplot(2,2,2)
    plt.plot(x, y2)
    plt.title('{} Number of Coronavirus Active cases Over Time'.format(country), size=10)
    plt.xlabel(' ', size=1)
    plt.ylabel(' ', size=1)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.show()

    plt.subplot(2,2,3)
    plt.plot(x, y3)
    plt.title('{} Number of Coronavirus Recoveries Over Time'.format(country), size=10)
    plt.xlabel(' ', size=1)
    plt.ylabel(' ', size=1)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.show()

    plt.subplot(2,2,4)
    plt.plot(x, y4)
    plt.title('{} Number of Coronavirus Deaths Over Time'.format(country), size=10)
    plt.xlabel(' ', size=1)
    plt.ylabel(' ', size=1)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.show()

def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d

def daily_india_plot(x, temp, feature):
    y = daily_increase(temp)
    plt.figure(figsize=(16, 9))
    plt.bar(x, y)
    plt.title('India: {}'.format(feature), size=20)
    plt.xlabel('Days Since 1/22/2020', size=20)
    plt.ylabel('Number of Cases', size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


def show_world_confirmed(latest_data):
    df = latest_data.copy()
    df = df.drop(['FIPS', 'Admin2','Combined_Key','Incidence_Rate','Case-Fatality_Ratio'], axis = 1)
    df = df.rename(columns={'Province_State':'Province', 'Country_Region': 'Country', 'Last_Update': 'Date', 'Long_': 'Long', })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df = df.drop_duplicates(['Country', 'Province'], keep = "first")
    df = df.dropna()

    cols_to_keep = list(df.columns[0:6])
    df_conf_last = df[cols_to_keep]
    df_conf_last = df_conf_last.drop(['Date'], axis = 1)

    df_conf_last['Confirmed'] = df_conf_last['Confirmed'].astype(float)

    map1 = folium.Map(location=[30.6, 114], zoom_start=3)

    for i in range(0,len(df_conf_last)):
        folium.Circle(location=[df_conf_last.iloc[i]['Lat'], df_conf_last.iloc[i]['Long']], tooltip = "Country: "+df_conf_last.iloc[i]['Country']+"<br>State: "+str(df_conf_last.iloc[i]['Province'])+"<br>Confirmed cases: "+str(df_conf_last.iloc[i]['Confirmed'].astype(int)), radius=df_conf_last.iloc[i]['Confirmed'], color='blue', fill=True, fill_color='blue').add_to(map1)

    map1.save("/home/eric/Desktop/csir_srtp/tomb/map_confirmed.html")
    os.system("firefox /home/eric/Desktop/csir_srtp/tomb/map_confirmed.html")

def show_world_deaths(latest_data):
    df = latest_data.copy()
    df = df.drop(['FIPS', 'Admin2','Combined_Key','Incidence_Rate','Case-Fatality_Ratio'], axis = 1)
    df = df.rename(columns={'Province_State':'Province', 'Country_Region': 'Country', 'Last_Update': 'Date', 'Long_': 'Long', })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df = df.drop_duplicates(['Country', 'Province'], keep = "first")
    df = df.dropna()

    cols_to_keep = list(df.columns[0:7])
    df_conf_last = df[cols_to_keep]
    df_conf_last = df_conf_last.drop(['Date'], axis = 1)
    df_conf_last['Deaths'] = df_conf_last['Deaths'].astype(float)

    map1 = folium.Map(location=[30.6, 114], zoom_start=3)

    for i in range(0,len(df_conf_last)):
        folium.Circle(location=[df_conf_last.iloc[i]['Lat'], df_conf_last.iloc[i]['Long']], tooltip = "Country: "+df_conf_last.iloc[i]['Country']+"<br>State: "+str(df_conf_last.iloc[i]['Province'])+"<br>Deaths: "+str(df_conf_last.iloc[i]['Deaths'].astype(int)), radius=df_conf_last.iloc[i]['Deaths'], color='red', fill=True, fill_color='red').add_to(map1)
    map1.save("/home/eric/Desktop/csir_srtp/tomb/map_deaths.html")
    os.system("firefox /home/eric/Desktop/csir_srtp/tomb/map_deaths.html")

def show_world_active_cases(latest_data):
    df = latest_data.copy()
    df = df.drop(['FIPS', 'Admin2','Combined_Key','Incidence_Rate','Case-Fatality_Ratio'], axis = 1)
    df = df.rename(columns={'Province_State':'Province', 'Country_Region': 'Country', 'Last_Update': 'Date', 'Long_': 'Long', })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df = df.drop_duplicates(['Country', 'Province'], keep = "first")
    df = df.dropna()
    df_conf_last = df.copy()
    df_conf_last = df_conf_last.drop(['Date'], axis = 1)
    df_conf_last['Active'] = df_conf_last['Active'].astype(float)

    map1 = folium.Map(location=[30.6, 114], zoom_start=3)

    for i in range(0,len(df_conf_last)):
        folium.Circle(location=[df_conf_last.iloc[i]['Lat'], df_conf_last.iloc[i]['Long']], tooltip = "Country: "+df_conf_last.iloc[i]['Country']+"<br>State: "+str(df_conf_last.iloc[i]['Province'])+"<br>Active cases: "+str(df_conf_last.iloc[i]['Active'].astype(int)),radius=df_conf_last.iloc[i]['Active'],color='blue',fill=True,fill_color='blue').add_to(map1)

    map1.save("/home/eric/Desktop/csir_srtp/tomb/map_active.html")
    os.system("firefox /home/eric/Desktop/csir_srtp/tomb/map_active.html")

def world_gradient_confirmed(latest_data):
    df = latest_data.copy()
    df = df.drop(['FIPS', 'Admin2','Combined_Key','Incidence_Rate','Case-Fatality_Ratio'], axis = 1)
    df = df.rename(columns={'Province_State':'Province', 'Country_Region': 'Country', 'Last_Update': 'Date', 'Long_': 'Long', })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df = df.drop_duplicates(['Country', 'Province'], keep = "first")
    df = df.dropna()
    top = df[df['Date'] == df['Date'].max()]
    world = top.groupby('Country')['Confirmed','Active','Deaths'].sum().reset_index()
    figure = px.choropleth(world, locations="Country", locationmode='country names', color="Confirmed", hover_name="Country", range_color=[1,20000], color_continuous_scale="Peach", title='Countries with Active Cases')
    figure.show()

def world_gradient_deaths(latest_data):
    df = latest_data.copy()
    df = df.drop(['FIPS', 'Admin2','Combined_Key','Incidence_Rate','Case-Fatality_Ratio'], axis = 1)
    df = df.rename(columns={'Province_State':'Province', 'Country_Region': 'Country', 'Last_Update': 'Date', 'Long_': 'Long', })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df = df.drop_duplicates(['Country', 'Province'], keep = "first")
    df = df.dropna()
    figure = px.choropleth(world, locations="Country", locationmode='country names', color="Deaths", hover_name="Country", range_color=[1,20000], color_continuous_scale="Peach", title='Countries with Active Cases')
    figure.show()

def heatmaps(latest_data):
    df = latest_data.copy()
    df = df.drop(['FIPS', 'Admin2','Combined_Key','Incidence_Rate','Case-Fatality_Ratio'], axis = 1)
    df = df.rename(columns={'Province_State':'Province', 'Country_Region': 'Country', 'Last_Update': 'Date', 'Long_': 'Long', })

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df = df.drop_duplicates(['Country', 'Province'], keep = "first")
    df = df.dropna()
    df1 = df
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1['Date'] = df1['Date'].dt.strftime('%m/%d/%Y')
    df1 = df1.fillna('-')
    fig1 = px.density_mapbox(df1,lat='Lat',lon='Long',z='Confirmed',radius=20,zoom=1,hover_data=["Country",'Province','Confirmed'],mapbox_style="carto-positron",animation_frame = 'Date',range_color= [0, 2000],title='Spread of Covid-19')
    fig1.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    fig1.show()

    fig2 = px.density_mapbox(df1,lat='Lat',lon='Long',z='Deaths',radius=20,zoom=1, hover_data=["Country",'Province','Deaths'], mapbox_style="carto-positron",animation_frame = 'Date',range_color= [0, 2000],title='Spread of Covid-19')
    fig2.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    fig2.show()

    fig3 = px.density_mapbox(df1,lat='Lat',lon='Long',z='Active',radius=20,zoom=1,hover_data=["Country",'Province','Active'],mapbox_style="carto-positron",animation_frame = 'Date',range_color= [0, 2000],title='Spread of Covid-19')
    fig3.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    fig3.show()

    
def predict_dates(df, n_forecast):
    l_date = df.index.values[-1]
    post_dates = pd.date_range(l_date, periods=n_forecast).tolist()
    return post_dates


def lstm_confirmed_cases_prediction(confirmed_cases, days_in_future):
    confirmed = confirmed_cases.drop(['Province/State','Country/Region','Lat','Long'] , axis = 1)
    dates = confirmed.keys()
    india_cases = []
    for i in dates:
        confirmed_sum = confirmed[i].sum()
        india_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='India'][i].sum())
    india_cases = np.array(india_cases).reshape(-1, 1)

    df = pd.DataFrame(index=dates, columns=['confirmed_cases'])
    df["confirmed_cases"] = india_cases

    df_data = df['confirmed_cases'].values
    df_data = df_data.reshape((-1,1))

    n_percent = 0.90
    s = int(n_percent*len(df_data))

    df_train = df_data[:s]
    df_test = df_data[s:]

    date_train = df.index[:s]
    date_test = df.index[s:]
    train_generator = TimeseriesGenerator(df_train, df_train, length=5, batch_size=6)     
    test_generator = TimeseriesGenerator(df_test, df_test, length=5, batch_size=6)
    model = Sequential()
    model.add(LSTM(100,activation='relu',input_shape=(5,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs =90
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
    prediction = model.predict_generator(test_generator)
    pred_list = []
    n_input = 5
    n_features = 1
    n_forecast = days_in_future

    batch = df_data[-n_input:].reshape((1, n_input, n_features))

    for i in range(n_forecast):   
        pred_list.append(model.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
    prediction_list = []
    for item in pred_list:
        prediction_list.append(int(item))
    
    l_date = df.index.values[-1]
    forecast_dates = pd.date_range(l_date, periods=n_forecast).tolist()

    df_predict = pd.DataFrame(index=forecast_dates, columns=['Prediction'])
    df_predict["Prediction"] = prediction_list
    df.index = pd.to_datetime(df.index)
    print(df_predict)
    plt.figure(figsize=(25, 5))
    plt.plot(df.index, df['confirmed_cases'])
    plt.plot(forecast_dates, df_predict["Prediction"],linestyle='dashed', color='r')
    plt.title('Number of Coronavirus Cases Over Time', size=20)
    plt.legend(['Confirmed Cases', 'LSTM_Model_Predictions'], prop={'size': 20}, fontsize='xx-large')
    plt.xlabel('Days Since 1/22/2020', size=20)
    plt.ylabel('Number of Cases', size=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.show()