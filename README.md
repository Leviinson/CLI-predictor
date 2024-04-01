# Introduction 
**I was given the task of making a CLI for predicting incoming data of type: date - value for the specified period.
The experience was very captivating. This was my first team work, that helped me to know many new things.
The path traveled gave a new charge of motivation, aesthetic pleasure.
I hope to see something like this within a month.**

# How the script works.
### Requirements
[![](https://img.shields.io/badge/python-3.10.4-blue)](https://www.python.org/downloads/) [![](https://img.shields.io/badge/numpy-1.23.4-4893da)](https://numpy.org/doc/) [![](https://img.shields.io/badge/statsmodels-0.13.2-7f48da)](https://www.statsmodels.org/stable/index.html) [![](https://img.shields.io/badge/matplotlib-3.6.1-daa748)](https://matplotlib.org/stable/index.html) [![](https://img.shields.io/badge/pandas-1.5.1-382282)](https://pandas.pydata.org/) [![](https://img.shields.io/badge/click-8.1-000000)](https://click.palletsprojects.com/en/8.1.x/)

**Initially, I was given the format of the input data:
The first column is the date, the second column is the load level of the process on that date.
It was necessary to implement a convenient acceptance of data sets through the results of sql queries and csv files using pipe.**

**The script itself represented as CLI. It can accept as SQL Query result, CSV file across the pipe.
The most convenient way is, for example**:

	cat path/to/file.csv | pythonX.X.X --update-freq X --seasonal-period X --number-of-predicted-points X --index-col X --sep X

Parameter `update frequence` is standard pandas frequency offset (details: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
It represents update frequence in the dataset. For example, in pinned dateset with project update frequence equals to 15 minutes.

Parameter `seasonal period` - number of points, that expected represents seasonal period.
To begin with, we must decompose our data using "Predictor().show_decomposed_result()" and check: how many points does one season take.

Parameter `number of predicted points` - number of points, that script will predict based on the previous points.

Parameter `index col` - number of the column with dates, starts from 0.

Parameter `sep` - separator in passed dataset.

Parameter `forecast path` - path to the forecast log, default values is working directory + /forecasts.csv

**All predicted values are written to a file called "forecasts.csv" by the standard.
Actions in non-standard situations are also provided in advance. For example: in theory, updates should come to the original data log every 15 minutes, but what else to do when the updated value has not arrived?
According to the standard, the last row with the predicted data is added to the pandas.DataFrame with the original data, and based on this dataframe, the next point is predicted.**

**The next situation: the previous condition was met twice. It turned out that we do not have data for two periods of time.
In this case, interpolation will be applied to empty cells, in a cubic way.**

**And the last case, that I could imagine: what to do, if script was turned off while original dataset was updating, and after some period it was turned on back?
In this situation script will look for values on missed dates from the forecast log, and if it exists - it will fills missed values with predicted values, and based on this it will predict the next point.
But what to do, if not all values were found and then filled? The answer is: script will again use interpolation with cubic method.**

## About smoothing model

**I was advised to use the Holt-Winters model.
Most of the datasets that were given to me had seasonality in the decomposition, plus the dataset itself was stationary. Therefore, I decided to use an exponential smoothing model with an additive trend and seasonality, since the volatility was not large, which made it possible to take into account data from the past with a greater degree.**

# Installation guide
	git clone https://github.com/Leviinson/prediction.git
