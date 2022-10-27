from email.policy import default
import io
import logging
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa import api

import os
import sys
import re
import requests
import click

from pathlib import Path
from typing import Dict, Any, Union

import logging
logging.basicConfig(level = logging.INFO)

class Deserializer:


    def __init__(self, index_col: Union[int, str],
                       update_freq: str,
                       sep: str,
                       data: io.TextIOWrapper) -> None:
        self._data = data
        self._index_col = index_col
        self._sep = sep
        self._update_freq = update_freq

    def get_df_from_data(self):
        df = pd.read_csv(
            self._data,
            index_col = self._index_col,
            sep = self._sep,
            parse_dates = True).asfreq(self._update_freq)
        assert df.values.any(), "Wrong separator was choosed or dataset is bad-formatted"
        df.index.names = ['datetime']
        return df


class Predictor:



    def __init__(self, actual_data: pd.DataFrame,
                       seasonal_period: int,
                       predicted_points_quantity: int,
                       data: io.TextIOWrapper,
                       forecast_path: str) -> None:
        self._actual_data = actual_data
        self._seasonal_period = seasonal_period
        self._predicted_points_quantity = predicted_points_quantity
        self._data = data
        self._forecast_path = forecast_path


    def check_updates(self, actual_data: pd.DataFrame,
                            forecast_data: pd.DataFrame) -> bool:

        return Updater.check_updates(actual_data = actual_data,
                                     forecast_data = forecast_data)


    def fill_missed_cells(self, forecast_data: pd.DataFrame):


        values_column_name = self._actual_data.columns[0]
        merged_data = self._actual_data.merge(forecast_data, on = 'datetime', how = 'left')['predicted_values']
        self._actual_data[values_column_name] = self._actual_data[values_column_name].fillna(merged_data)


    def show_decomposed_result(self):
        '''Decomposing the dataframe to determine prediction model.'''
        decompose_result = api.seasonal_decompose(self._actual_data[self._actual_data.columns[0]],
                                                  model = 'add',
                                                  period = self._seasonal_period)
        decompose_result.plot()
        return plt.show()


    def create_prediction_line(self, update_freq: str):

        
        if os.path.isfile(self._forecast_path):
            forecast_data = pd.read_csv(self._forecast_path, index_col = 0, parse_dates = True)
            updates = self.check_updates(actual_data = self._actual_data,
                                         forecast_data = forecast_data)

            if not updates:
                prediction_of_the_losted_dataframe = forecast_data.tail(1)
                losted_dataframe = pd.DataFrame(data = prediction_of_the_losted_dataframe.values[0][0],
                                                index = prediction_of_the_losted_dataframe.axes[0],
                                                columns=['decarie.web-dns1.com'])

                self._actual_data = pd.concat([self._actual_data, losted_dataframe]).asfreq(update_freq)

        data_has_empty_cells = (self._actual_data.isna().sum() > 0).bool()
        if data_has_empty_cells:
            self.fill_missed_cells(forecast_data)

        exog_title = self._actual_data.columns[0]
        assert len(self._actual_data[exog_title]) / self._seasonal_period >= 2, "time series must have 2 complete cycles with specified seasonal period"
        fitted_model = api.ExponentialSmoothing(self._actual_data,
                                                seasonal_periods = self._seasonal_period,
                                                trend = "add",
                                                seasonal = "add",
                                                use_boxcox = False,
                                                damped_trend = True).fit()
        prediction_line: pd.Series = fitted_model.forecast(self._predicted_points_quantity)
        return prediction_line

class Callbacker:
    
    fails = 0
    def __init__(self) -> None:
        pass
    
    @classmethod
    def validate_data(cls, actual_value: float, predicted_value: float):

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
        print(f"It`s a callback, difference: {kwargs['difference']}")


class Updater:


    def __init__(self) -> None:
        pass

    @classmethod
    def check_updates(cls, actual_data: pd.DataFrame,
                           forecast_data: pd.DataFrame) -> bool:


        actual_date = actual_data.index[-1]
        date_for_forecast = forecast_data.index[-1]
        predicted_value = forecast_data.values[-1][0]

        if actual_date == date_for_forecast:
            actual_value = actual_data.values[-1]
            is_update = True
            Callbacker.validate_data(actual_value, predicted_value)

        elif actual_date > date_for_forecast:
            logging.info('There is data added outside of the running script.')
            is_update = True

        else:
            logging.info('Here is no new values, so prediction will be based on the previous prediction value.')
            is_update = False

        return is_update


class Serializer:


    def __init__(self, predicted_line: pd.Series,
                       actual_data: pd.DataFrame,
                       forecast_path: str) -> None:

        self._predicted_line = predicted_line
        self._actual_data = actual_data
        self._forecast_path = forecast_path


    def add_prediction_line_to_forecast(self) -> None:

        predicted_value = self._predicted_line.values[0]
        if predicted_value < 0: predicted_value = 0
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
                   predicted_points_quantity: int,
                   forecast_path: str) -> None:

        data = click.get_binary_stream(name = 'stdin')

        actual_data = self._deserializer(data = data,
                                         index_col = index_col,
                                         sep = sep,
                                         update_freq = update_freq).get_df_from_data()

        predicted_line = self._predictor(actual_data = actual_data,
                                         seasonal_period = seasonal_period,
                                         predicted_points_quantity = predicted_points_quantity,
                                         data = data,
                                         forecast_path = forecast_path
                                         ).create_prediction_line(update_freq)

        self._serializer(actual_data = actual_data,
                         predicted_line = predicted_line,
                         forecast_path = forecast_path
                         ).add_prediction_line_to_forecast()


@click.command()
@click.option('-ic', '--index-col', type = int, default = 0)
@click.option('-s', '--sep', type = str, default = ',')
@click.option('-uf', '--update-frequence', 'update_freq', type = str, required = True)
@click.option('-sp', '--seasonal-period', type = int, default = 4*24*7)
@click.option('-ppq', '--predicted-points-quantity', type = int, default = 1)
@click.option('-fp', '--forecast-path', type = str, default = os.getcwd() + '/forecasts.csv')
def main(index_col: int,
         sep: str,
         update_freq: str,
         seasonal_period: int,
         predicted_points_quantity: int,
         forecast_path: str) -> None:

    return Controller().main(index_col, sep,
                             update_freq, seasonal_period,
                             predicted_points_quantity, forecast_path)

if __name__ == '__main__':
    main()
