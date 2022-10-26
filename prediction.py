import io
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa import api

import os
import sys
import re
import requests

from pathlib import Path
from typing import Dict, Any, Union

import logging
logging.basicConfig(level = logging.INFO)

class Deserializer:


    def __init__(self, index_col: Union[int, str], update_freq: str,
                       data: io.TextIOWrapper) -> None:
        self._data = data
        self._index_col = index_col
        self._update_freq = update_freq

    def get_df_from_data(self):
        df = pd.read_csv(
            self._data,
            index_col = self._index_col, 
            parse_dates = True).asfreq(self._update_freq)
        df.index.names = ['datetime']
        return df


class Predictor:



    def __init__(self, actual_data: pd.DataFrame,
                       seasonal_period: int,
                       predicted_points_quantity: int,
                       data: io.TextIOWrapper,
                       forecast_path: str = os.getcwd() + '/forecasts.csv',) -> None:
        self._actual_data = actual_data
        self._seasonal_period = seasonal_period
        self._predicted_points_quantity = predicted_points_quantity
        self._data = data
        self._forecast_path = forecast_path


    def check_updates(self, actual_data: pd.DataFrame,
                            forecast_data: pd.DataFrame,
                            forecast_path: str = os.getcwd() + '/forecasts.csv') -> bool:

        return Updater.check_updates(actual_data = actual_data,
                                     forecast_data = forecast_data)

    def show_decomposed_result(self):
        '''Decomposing the dataframe to determine prediction model.'''
        decompose_result = api.seasonal_decompose(self._actual_data[self._actual_data.columns[0]],
                                                  model = 'add',
                                                  period = self._seasonal_period)
        decompose_result.plot()
        return plt.show()


    def create_prediction_line(self, update_freq: str):

        exog_title = self._actual_data.columns[0]
        
        if os.path.isfile(self._forecast_path):
            forecast_data = pd.read_csv(self._forecast_path, index_col = 0, parse_dates = True)
            updates = self.check_updates(actual_data = self._actual_data,
                                         forecast_data = forecast_data,
                                         forecast_path = self._forecast_path)

            if not updates:
                prediction_of_the_losted_dataframe = forecast_data.tail(1)
                losted_dataframe = pd.DataFrame(data = prediction_of_the_losted_dataframe.values[0][0],
                                                index = prediction_of_the_losted_dataframe.axes[0],
                                                columns=['decarie.web-dns1.com'])

                values_col_name = self._actual_data.columns[0]
                self._actual_data = pd.concat([self._actual_data, losted_dataframe]).asfreq(update_freq)
                if (self._actual_data.isna().sum() > 0).bool():
                    merged_data = self._actual_data.merge(forecast_data, on = 'datetime', how = 'left')['predicted_values']
                    self._actual_data[values_col_name] = self._actual_data[values_col_name].fillna(merged_data)


        assert len(self._actual_data[exog_title]) / self._seasonal_period >= 2, "time series must have 2 complete cycles with specified seasonal period"
        fitted_model = api.ExponentialSmoothing(self._actual_data[exog_title],
                                                seasonal_periods = self._seasonal_period,
                                                trend = "add",
                                                seasonal = "add",
                                                use_boxcox = False,
                                                damped_trend = True).fit()

        prediction_line: pd.Series = fitted_model.forecast(self._predicted_points_quantity)
        return prediction_line

        

        # fitted_model[self._actual_data.columns.name[0]].plot(legend = True, label = 'FITTED MODEL', figsize = (6,4))
        # prediction_line.plot(legend = True, label ='PREDICTION')
        # plt.show()

class Callbacker:
    
    fails = 0
    def __init__(self) -> None:
        pass
    
    @classmethod
    def validate_data(cls, actual_value: float, predicted_value: float):

        if actual_value ==  0.0:
            actual_value += 0.001

        if predicted_value / actual_value >= 1.8:
            return cls.callback()
        else:
            return logging.info('Data passed validation')
            
    @classmethod
    def callback(cls, *args, **kwargs):
        cls.fails += 1
        print('Что хотите - то и творите.')


class Updater:


    def __init__(self) -> None:
        pass

    @classmethod
    def check_updates(cls, actual_data: pd.DataFrame,
                           forecast_data: pd.DataFrame) -> bool:


        actual_date = actual_data.index[-1]
        date_for_forecast = forecast_data.index[-1]
        
        predicted_value = float(forecast_data.values[-1][0])
        
        if actual_date > date_for_forecast:
            logging.info('There is data added outside of the running script.')
            actual_value = actual_data.loc[date_for_forecast].values[0]
            is_update = True

        elif actual_date != date_for_forecast:
            logging.info('Here is no new values, so prediction will be based on the previous prediction value.')
            actual_value = predicted_value
            is_update = False
        else:
            actual_value = float(actual_data.values[-1])
            is_update = True

        Callbacker.validate_data(actual_value, predicted_value)
        return is_update


class Serializer:


    def __init__(self, predicted_line: pd.Series,
                       actual_data: pd.DataFrame,
                       forecast_path: str = os.getcwd() + '/forecasts.csv') -> None:

        self._predicted_line = predicted_line
        self._actual_data = actual_data
        self._forecast_path = forecast_path

    def __get_endog_and_exog_of_predicted_line(self):


        self._endogs: pd.DatetimeIndex = self._predicted_line.axes[0]
        self._exogs: np.array = self._predicted_line.values


    def add_prediction_line_to_forecast(self) -> None:


        self.__get_endog_and_exog_of_predicted_line()
        if not os.path.isfile(self._forecast_path):

            dataframe_with_predicted_values = pd.DataFrame(data = self._exogs,
                                                           index = self._endogs,
                                                           columns = ['predicted_values'])

            dataframe_with_predicted_values.index.names = ['datetime']
            dataframe_with_predicted_values.to_csv(self._forecast_path)
        else:
            dataframe_with_predicted_values = pd.DataFrame(data = self._exogs,
                                                           index = self._endogs)
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
                   update_freq: str,
                   seasonal_period: int,
                   predicter_points_quantity: int,
                   data: io.TextIOWrapper,
                   forecast_path: str = os.getcwd() + '/forecasts.csv') -> None:

        
        actual_data = self._deserializer(data = data,
                                         index_col = index_col,
                                         update_freq = update_freq).get_df_from_data()

        predicted_line = self._predictor(actual_data = actual_data,
                                          seasonal_period = seasonal_period,
                                          predicted_points_quantity = predicter_points_quantity,
                                          data = data,
                                          forecast_path = forecast_path,
                                          ).create_prediction_line(update_freq)

        self._serializer(actual_data = actual_data,
                         predicted_line = predicted_line,
                         forecast_path = os.getcwd() + '/forecasts.csv'
                         ).add_prediction_line_to_forecast()
    


if __name__ == '__main__':

    
    # train_line = pd.read_csv(sys.stdin, index_col = 0, parse_dates=True).asfreq('15min')
    # for index, row in train_line.iterrows():
    #     Controller().main(data = sys.stdin,
    #                       index_col = 0,
    #                       update_freq = '15min',
    #                       seasonal_period = 4*24*7,
    #                       predicter_points_quantity = 1)
    #     pd.DataFrame({'values': float(row)}, index = [str(index)]).to_csv('/home/levinsxn/Documents/python/prediction/service3/data/cpu/decarie_test.csv', mode = 'a', header = False)
    # print(Callbacker.fails)

    Controller().main(data = sys.stdin,
                      index_col = 0,
                      update_freq = '15min',
                      seasonal_period = 4*24*7,
                      predicter_points_quantity = 1)
