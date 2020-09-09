import sys
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5 import QtGui
import process

class mainclass(QMainWindow):
    def __init__(self):
        super(mainclass, self).__init__()
        loadUi('/home/eric/Desktop/csir_srtp/tomb/gui.ui', self)

        self.date1 = self.call._date.text()
        self.india_statewise, self.world_confirmed, self.world_deaths, self.world_recovered, self.latest_data, self.df1, self.df2, self.df3, self.dates, self.dates_india = process.prepare_datasets(self.date1)
        self.dates_polyreg, self.Starting_1_22, self.india_cases, self.india_active, self.India_confirmed, self.india_recoveries, self.india_deaths, self.world_cases = process.preproc_for_polyreg(self.world_confirmed, self.world_deaths, self.world_recovered, self.latest_data)
        self.days = 20

        self.pb_eda_plot.clicked.connect(eda_plots)
        self.pb_fbprophet_forecast.clicked.connect(fbprophet_forecasting)
        self.pb_fit_model.clicked.connect(polyreg_model_deploy)
        self.pb_poly_forecast.clicked.connect(polyreg_forecast)
        self.pb_lstm.clicked.connect(lstm_forecast)

    def eda_plots(self):
        eda_function = self.call.eda_combo_box.currentText()

        if eda_function == 'Concise graph of Covid-19 in India':
            process.plot_covid_india_graph()

        elif eda_function == 'Available hospitals and beds':
            process.plot_covid_india_bedsgraph()

        
        elif eda_function == 'Patients in different age groups':
            process.plot_covid_india_agegraph()

        elif eda_function == 'No. of tests, and their reports':
            process.plot_covid_india_testsgraph()

        elif eda_function == 'Confirmed cases in India vs outiside':
            process.plot_outside_vs_in(self.latest_data)

        elif eda_function == 'Covid-19 in India over time':
            process.plot_country_wise(self.dates, days, self.india_cases, self.india_active, self.india_recoveries , self.india_deaths , 'INDIA')  
        
        elif eda_function == 'Covid-19 in World over time':
            process.plot_country_wise(self.dates, days, self.world_cases, self.world_confirmed, self.world_recovered , self.world_deaths , 'World')

        elif eda_function == 'Daily increase in confirmed cases':
            process.daily_india_plot(self.dates, self.india_cases, eda_function)

        elif eda_function == 'Daily increase in recoveries':
            process.daily_india_plot(self.dates, self.india_recoveries , eda_function)

        elif eda_function == 'Daily increase in deaths':
            process.daily_india_plot(self.dates, self.india_deaths , eda_function)

        elif eda_function == 'Confirmed cases on world map':
            process.show_world_confirmed(self.latest_data)

        elif eda_function == 'No. of Deaths on world map':
            process.show_world_deaths(self.latest_data)

        elif eda_function == 'Active cases on world map':
            process.show_world_active_cases(self.latest_data)

        elif eda_function == 'World gradient confirmed cases':
            process.world_gradient_confirmed(self.latest_data)

        elif eda_function == 'World gradient number of deaths':
            process.world_gradient_deaths(self.latest_data)

        elif eda_function == 'Heat map of Covid-19 in world':
            process.heatmaps(self.latest_data)

        
    def fbprophet_forecasting(self):
        days_in_future = int(self.call.dif.text())
        fbprophet_feature = self.call.Fbprophet_features.currentText()
        
        if fbprophet_feature == 'Daily Confirmed':
            process.plot_fbprophet_forecasting(self.df1, self.dates, self.days_in_future, fbprophet_feature)
        
        elif fbprophet_feature == 'Daily Deaths':
            process.plot_fbprophet_forecasting(self.df2, self.dates, self.days_in_future, self.fbprophet_feature)
        
        elif fbprophet_feature == 'Daily Recovered':
            process.plot_fbprophet_forecasting(self.df3, self.dates, self.days_in_future, fbprophet_feature)



    def polyreg_model_deploy(self):
        degree = int(self.call.degree.text())
        days_in_future = int(self.call.dif.text())
        process.polyreg_model_fit(days_in_future, self.dates, self.Starting_1_22, self.india_cases, degree)


    def polyreg_forecast(self):
        days_in_future = int(self.call.dif.text())
        process.forecast_polyreg(days_in_future, self.dates, self.Starting_1_22, self.india_cases)

    def lstm_forecast(self):
        days_in_future = int(self.call.dif.text())
        process.lstm_confirmed_cases_prediction(self.world_confirmed, days_in_future)


app = QApplication(sys.argv)
window = mainclass()
window.show()
try:
    sys.exit(app.exec_())
except:
    print('Exiting')
