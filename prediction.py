'''
This module was developed as CLI with the main task to predict the next point of the passed dataset.
It uses "Facade" design pattern, that makes easy to extend existing code.

Uses "statsmodels" module for predicting and "click" for realising comand line interface.
'''
import io
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa import api

import os
import sys
import click

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)



class Deserializer:
    '''
    Accepts bytearray and interpreterns it into dataframe.
    
    ## Parameters:
        :param: index_col - number of the column with dates, starts from 0
        :param: update_freq - pandas frequency offset, details: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        :param: sep - separator in the dataset. For example, usually in csv uses "," or ";"
        :param: data - dataset passed across the pipe, for example: cat path/to/file | python3 prediction.py -ic 0 -s "," -if "15T" -sp 672 -ppq 1
    ## Returns:
        pandas.DataFrame based on the passed dataset
    '''
    def __init__(self, index_col: int | str,
                       update_freq: str,
                       sep: str,
                       data: io.BufferedReader) -> None:
        self._data = data
        self._index_col = index_col
        self._sep = sep
        self._update_freq = update_freq
    
    def get_df_from_data(self):
        '''
        Returns pandas.DataFrane using passed dataset.
        '''
        df = pd.read_csv(
            self._data,
            index_col = self._index_col,
            sep = self._sep,
            parse_dates = True).asfreq(self._update_freq)
        assert df.values.any(), "Wrong separator was choosed or dataset is bad-formatted"
        df.index.names = ['datetime']
        return df


class Predictor:
    '''
    Realises prediction of the passed dataset.

    ## Accepts:
        :param: actual_data - pandas.DataFrame, that represents passed dataset across the pipe.
        :param: seasonal_period - number of points, that expected represents seasonal period.
            For example we have a dataset with period between points in 15 minutes.
            To begin with, we must decompose our data using "show_decomposed_result" and check: how many points does one season take.
        :param: number_of_predicted_points - number of points, that script will predict based on the previous points
        :param: data - passed dataset
        :paran: forecast_path - path to the forecast log

    ## Returns:
        pandas.DataFrane that represents predicted point
    '''
    def __init__(self, actual_data: pd.DataFrame,
                       seasonal_period: int,
                       number_of_predicted_points: int,
                       data: io.BufferedReader,
                       forecast_path: str,
                       index_col: int) -> None:
        self._actual_data = actual_data
        self._seasonal_period = seasonal_period
        self._number_of_predicted_points = number_of_predicted_points
        self._data = data
        self._forecast_path = forecast_path
        self._index_col = index_col


    def check_updates(self, actual_data: pd.DataFrame,
                            last_forecast_data: pd.DataFrame) -> bool:
        '''
        ## Accepts:
            :param: actual_data - pandas.DataFrame, that represents passed dataset
            :param: last_forecast_data - pandas.DataFrame, that represents last predicted point from forecast log.

        ## Returns:
            Boolean value representing whether there are updates to the results of the "check_updates" delegate method of the "Updater" class.
        '''
        return Updater.check_updates(actual_data = actual_data,
                                     last_forecast_data = last_forecast_data)


    def show_decomposed_result(self):
        '''
        Decomposes the dataframe to determine prediction model.
        '''
        decompose_result = api.seasonal_decompose(self._actual_data[self._actual_data.columns[0]],
                                                  model = 'add',
                                                  period = self._seasonal_period)
        decompose_result.plot()
        return plt.show()

    def create_prediction_line(self, update_freq: str):
        '''
        ## Accepts:
            :param: update_freq - pandas frequency offset, details: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        
        ## Returns:
            pandas.DataFrane that represents predicted point
        '''
        self.__normalize_actual_data(update_freq = update_freq)
        exog_title = self._actual_data.columns[0]
        assert len(self._actual_data[exog_title]) / self._seasonal_period >= 2, "time series must have 2 complete cycles with specified seasonal period"

        try:
            fitted_model = api.ExponentialSmoothing(self._actual_data,
                                                    seasonal_periods = self._seasonal_period,
                                                    trend = "add",
                                                    seasonal = "add",
                                                    use_boxcox = False,
                                                    damped_trend = True).fit()
            prediction_line: pd.Series = fitted_model.forecast(self._number_of_predicted_points)
            return prediction_line
        except Exception as e:
            logging.error(e, exc_info = True)

    
    def __normalize_actual_data(self, update_freq: str):
        '''
        ## Accepts:
            :param: update_freq - pandas frequency offset, details: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        
        ## Purpose:
            Fill missed values on the pandas.DataFrame, that represents passed dataset with values from the forecast log if it exists.
        '''

        if os.path.isfile(self._forecast_path):
            forecast_data = pd.read_csv(self._forecast_path, index_col = self._index_col, parse_dates = True)
            updates = self.check_updates(actual_data = self._actual_data,
                                         last_forecast_data = forecast_data.tail(1))

            if not updates:
                prediction_of_the_losted_dataframe = forecast_data.tail(1)
                losted_dataframe = pd.DataFrame(data = prediction_of_the_losted_dataframe.values[0][0],
                                                index = prediction_of_the_losted_dataframe.axes[0],
                                                columns=['decarie.web-dns1.com'])
                self._actual_data = pd.concat([self._actual_data, losted_dataframe]).asfreq(update_freq)

        data_has_empty_cells = (self._actual_data.isna().sum() > 0).bool()
        if data_has_empty_cells:
            self.__fill_missed_cells(forecast_data)

    
    def __fill_missed_cells(self, forecast_data: pd.DataFrame):
        '''
        ## Accepts:
            :param: forecast_data - pandas.DataFrame, that represents forecast log
        
        ## Purpose:
            If missed values wasn\'t in the forecast log, than fills missed values in the forecast data with interpolated values.
        '''

        values_column_name = self._actual_data.columns[0]
        merged_data = self._actual_data.merge(forecast_data, on = 'datetime', how = 'left')['predicted_values']
        self._actual_data[values_column_name] = self._actual_data[values_column_name].fillna(merged_data)
        data_has_empty_cells = (self._actual_data.isna().sum() > 0).bool()
        if data_has_empty_cells:
            self._actual_data.interpolate(method = 'linear', inplace = True)


class Callbacker:
    '''
    ## Purpose:
        Сhecks the previously predicted value with the value that came in the dataset.
    ## Returns:
       Callback if the value is not accurate enough.
    '''
    
    def __init__(self) -> None:
        pass
    
    @classmethod
    def validate_data(cls, actual_value: np.ndarray, predicted_value: np.ndarray):
        '''
        ## Accepts:
            :param: actual_value - last row from pandas.DataFrame, that represents passed dataset.
            :param: predicted_value
        '''

        if actual_value ==  0.0:
            actual_value = 0.001

        if predicted_value > 0:
            difference = predicted_value / actual_value
        else:
            difference = actual_value

        if difference >= 1.8:
            return cls.callback(difference = difference)
        else:
            return logging.info(f'Data passed validation, difference: {difference}')
            
    @classmethod
    def callback(cls, *args, **kwargs):
        cls.fails += 1
        logging.info(f"It`s a callback, difference: {kwargs['difference']}")


class Updater:


    @classmethod
    def check_updates(cls, actual_data: pd.DataFrame,
                           last_forecast_data: pd.DataFrame) -> bool:
        '''
        ## Accepts:
            :param: actual_data - pandas.DataFrame, that represents passed dataset.
            :param: last_forecast_data - pandas.DataFrame, that represents last row from forecast log.
        ## Returns:
            Boolean value representing whether there are updates or no.
        '''

        actual_date = actual_data.index[-1]
        date_for_forecast = last_forecast_data.index
        predicted_value = last_forecast_data.values[0]

        if actual_date == date_for_forecast:
            actual_value = actual_data.values[-1]
            is_update = True
            Callbacker.validate_data(actual_value, predicted_value)

        elif actual_date > date_for_forecast:
            logging.error('There is data added outside of the running script.')
            is_update = True

        else:
            logging.error('Here is no new values, so prediction will be based on the previous prediction value.')
            is_update = False

        return is_update


class Serializer:
    '''
    ## Accepts:
        :param: predicted_line - pd.core.series.Series, that represents last row from forecast log.
        :param: actual_data - pd.DataFrame, that represents passed dataset.
        :param: forecast_path - str, that represents path to forecast log.
    ## Purpose:
        Serialize predicted line into forecast log.
    '''

    def __init__(self, predicted_line: pd.Series,
                       actual_data: pd.DataFrame,
                       forecast_path: str) -> None:

        self._predicted_line = predicted_line
        self._actual_data = actual_data
        self._forecast_path = forecast_path


    def add_prediction_line_to_forecast(self) -> None:
        '''
        ## Purpose:
            Serialize predicted line into forecast log if it exists.
            Otherwise creates forecast log and adds first row.
        '''
        predicted_value = self._predicted_line.values[0]
        predicted_date = self._predicted_line.index

        if not os.path.isfile(self._forecast_path):
            dataframe_with_predicted_values = pd.DataFrame(data = predicted_value,
                                                           index = predicted_date,
                                                           columns = ['predicted_values'])

            dataframe_with_predicted_values.index.names = ['datetime']
            dataframe_with_predicted_values.to_csv(self._forecast_path)
        else:
            dataframe_with_predicted_values = pd.DataFrame(data = predicted_value,
                                                           index = predicted_date, columns=['predicted_value'])
            dataframe_with_predicted_values.to_csv(self._forecast_path,
                                                   mode = 'a',
                                                   header = False)


class Controller:
    '''
    From the "Facade" design pattern this class represents Facade itself.
    Represents comfortable, short interface to start script.
    '''
    def __init__(self, deserializer = Deserializer,
                       predictor = Predictor,
                       serializer = Serializer,
                       updater = Updater,
                       callbacker = Callbacker) -> None:
        self._deserializer = deserializer
        self._predictor = predictor
        self._serializer = serializer
        self._updater = updater
        self._callbacker = callbacker
    
    def main(self, index_col: int,
                   sep: str,
                   update_freq: str,
                   seasonal_period: int,
                   number_of_predicted_points: int,
                   forecast_path: str) -> None:

        data = click.get_binary_stream(name = 'stdin')
        try:
            actual_data = self._deserializer(data = data,
                                             index_col = index_col,
                                             sep = sep,
                                             update_freq = update_freq).get_df_from_data()
        except Exception as e:
            logging.error(e, exc_info = True)

        predicted_line = self._predictor(actual_data = actual_data,
                                         seasonal_period = seasonal_period,
                                         number_of_predicted_points = number_of_predicted_points,
                                         data = data,
                                         forecast_path = forecast_path,
                                         index_col = index_col).create_prediction_line(update_freq)

        self._serializer(actual_data = actual_data,
                         predicted_line = predicted_line,
                         forecast_path = forecast_path
                         ).add_prediction_line_to_forecast()


@click.command()
@click.option('-ic', '--index-col', type = int, default = 0)
@click.option('-s', '--sep', type = str, default = ',')
@click.option('-uf', '--update-frequence', 'update_freq', type = str, required = True)
@click.option('-sp', '--seasonal-period', type = int, default = 4*24*7)
@click.option('-npp', '--number-of-predicted-points', type = int, default = 1)
@click.option('-fp', '--forecast-path', type = str, default = os.getcwd() + '/forecasts.csv')
def main(index_col: int,
         sep: str,
         update_freq: str,
         seasonal_period: int,
         number_of_predicted_points: int,
         forecast_path: str) -> None:

    return Controller().main(index_col, sep,
                             update_freq, seasonal_period,
                             number_of_predicted_points, forecast_path)

if __name__ == '__main__':
    main()
