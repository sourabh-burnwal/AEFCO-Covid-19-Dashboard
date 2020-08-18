from PyQt5 import QtWidgets,uic
import process


def eda_plots():
    date1 = call._date.text()
    days = 20
    india_statewise, world_confirmed, world_deaths, world_recovered, latest_data, df1, df2, df3, dates, dates_india = process.prepare_datasets(date1)
    dates_polyreg, Starting_1_22, india_cases, india_active, India_confirmed, india_recoveries, india_deaths, world_cases = process.preproc_for_polyreg(world_confirmed, world_deaths, world_recovered, latest_data)
    eda_function = call.eda_combo_box.currentText()

    if eda_function == 'Concise graph of Covid-19 in India':
        process.plot_covid_india_graph()

    elif eda_function == 'Available hospitals and beds':
        process.plot_covid_india_bedsgraph()

    
    elif eda_function == 'Patients in different age groups':
        process.plot_covid_india_agegraph()

    elif eda_function == 'No. of tests, and their reports':
        process.plot_covid_india_testsgraph()

    elif eda_function == 'Confirmed cases in India vs outiside':
        process.plot_outside_vs_in(latest_data)

    elif eda_function == 'Covid-19 in India over time':
        process.plot_country_wise(dates, days, india_cases, india_active, india_recoveries , india_deaths , 'INDIA')  
    
    elif eda_function == 'Covid-19 in World over time':
        process.plot_country_wise(dates, days, world_cases, world_confirmed, world_recovered , world_deaths , 'World')

    elif eda_function == 'Daily increase in confirmed cases':
        process.daily_india_plot(dates, india_cases, eda_function)

    elif eda_function == 'Daily increase in recoveries':
        process.daily_india_plot(dates, india_recoveries , eda_function)

    elif eda_function == 'Daily increase in deaths':
        process.daily_india_plot(dates, india_deaths , eda_function)

    elif eda_function == 'Confirmed cases on world map':
        process.show_world_confirmed(latest_data)

    elif eda_function == 'No. of Deaths on world map':
        process.show_world_deaths(latest_data)

    elif eda_function == 'Active cases on world map':
        process.show_world_active_cases(latest_data)

    elif eda_function == 'World gradient confirmed cases':
        process.world_gradient_confirmed(latest_data)

    elif eda_function == 'World gradient number of deaths':
        process.world_gradient_deaths(latest_data)

    elif eda_function == 'Heat map of Covid-19 in world':
        process.heatmaps(latest_data)

    
def fbprophet_forecasting():
    date1 = call._date.text()
    india_statewise, world_confirmed, world_deaths, world_recovered, latest_data, df1, df2, df3, dates, dates_india = process.prepare_datasets(date1)
    days_in_future = int(call.dif.text())
    fbprophet_feature = call.Fbprophet_features.currentText()
    
    if fbprophet_feature == 'Daily Confirmed':
        process.plot_fbprophet_forecasting(df1, dates, days_in_future, fbprophet_feature)
    
    elif fbprophet_feature == 'Daily Deaths':
        process.plot_fbprophet_forecasting(df2, dates, days_in_future, fbprophet_feature)
    
    elif fbprophet_feature == 'Daily Recovered':
        process.plot_fbprophet_forecasting(df3, dates, days_in_future, fbprophet_feature)



def polyreg_model_deploy():
    date1 = call._date.text()
    degree = int(call.degree.text())
    days_in_future = int(call.dif.text())
    india_statewise, world_confirmed, world_deaths, world_recovered, latest_data, df1, df2, df3, dates, dates_india = process.prepare_datasets(date1)
    dates_polyreg, Starting_1_22, india_cases, india_active, India_confirmed, india_recoveries, india_deaths, world_cases = process.preproc_for_polyreg(world_confirmed, world_deaths, world_recovered, latest_data)
    process.polyreg_model_fit(days_in_future, dates, Starting_1_22, india_cases, degree)


def polyreg_forecast():
    date1 = call._date.text()
    degree = int(call.degree.text())
    days_in_future = int(call.dif.text())
    india_statewise, world_confirmed, world_deaths, world_recovered, latest_data, df1, df2, df3, dates, dates_india = process.prepare_datasets(date1)
    dates_polyreg, Starting_1_22, india_cases, india_active, India_confirmed, india_recoveries, india_deaths, world_cases = process.preproc_for_polyreg(world_confirmed, world_deaths, world_recovered, latest_data)
    process.forecast_polyreg(days_in_future, dates, Starting_1_22, india_cases)

def lstm_forecast():
    date1 = call._date.text()
    india_statewise, world_confirmed, world_deaths, world_recovered, latest_data, df1, df2, df3, dates, dates_india = process.prepare_datasets(date1)
    days_in_future = int(call.dif.text())
    process.lstm_confirmed_cases_prediction(world_confirmed, days_in_future)


app = QtWidgets.QApplication([])
call = uic.loadUi('/home/eric/Desktop/csir_srtp/tomb/gui.ui')
call.pb_eda_plot.clicked.connect(eda_plots)
call.pb_fbprophet_forecast.clicked.connect(fbprophet_forecasting)
call.pb_fit_model.clicked.connect(polyreg_model_deploy)
call.pb_poly_forecast.clicked.connect(polyreg_forecast)
call.pb_lstm.clicked.connect()
call.show()
app.exec()