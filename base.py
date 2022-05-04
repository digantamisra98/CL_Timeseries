import csv
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.transformer import TransformerEstimator

import utils.data_loader as loader

BASE_DIR = "TSForecasting"

# The name of the column containing time series values after loading data from the .tsf file into a dataframe
VALUE_COL_NAME = "series_value"

# The name of the column containing timestamps after loading data from the .tsf file into a dataframe
TIME_COL_NAME = "start_timestamp"

# Seasonality values corresponding with the frequencies: minutely, 10_minutes, half_hourly, hourly, daily, weekly, monthly, quarterly and yearly
# Consider multiple seasonalities for frequencies less than daily
SEASONALITY_MAP = {
    "minutely": [1440, 10080, 525960],
    "10_minutes": [144, 1008, 52596],
    "half_hourly": [48, 336, 17532],
    "hourly": [24, 168, 8766],
    "daily": 7,
    "weekly": 365.25 / 7,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

# Frequencies used by GluonTS framework
FREQUENCY_MAP = {
    "minutely": "1min",
    "10_minutes": "10min",
    "half_hourly": "30min",
    "hourly": "1H",
    "daily": "1D",
    "weekly": "1W",
    "monthly": "1M",
    "quarterly": "1Q",
    "yearly": "1Y",
}

def get_window_index(df, num_windows):

    for index, row in df.iterrows():
        if VALUE_COL_NAME == 'price':
            series_data = row[VALUE_COL_NAME]
            total_length = len(series_data)

    assert num_windows > 0, "Number of windows cannot be less than 1"
    assert num_windows < total_length, "Number of windows cannot be greater than the length of the series"
    
    window_length = total_length // num_windows
    window_start_index = 0

    window_index_list = []

    for i in range(num_windows):
        window_end_index = window_start_index + window_length
        window_index_list.append((window_start_index, window_end_index))
        window_start_index = window_end_index

    return window_index_list

def get_window(df, num_windows, window_number, replay_window_num, freq):
    train_series_list = []
    test_series_list = []
    train_series_full_list = []
    test_series_full_list = []

    window_index_list = get_window_index(df, num_windows)
    window_start = window_index_list[window_number][0]
    window_end = window_index_list[window_number][1]
    replay_window_index = window_index_list[window_number - replay_window_num][0]

    for index, row in df.iterrows():
        if VALUE_COL_NAME == 'price':
            if TIME_COL_NAME in df.columns:
                train_start_time = row[TIME_COL_NAME]
            else:
                train_start_time = datetime.strptime(
                    "1900-01-01 00-00-00", "%Y-%m-%d %H-%M-%S"
                )  # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

            series_data = row[VALUE_COL_NAME]

            forecast_horizon = np.round(0.2 * len(series_data))

            # Creating training and test series. Test series will be only used during evaluation
            if window_start != 0:
                train_series_data = series_data[replay_window_index: window_end - forecast_horizon]
            else:
                train_series_data = series_data[window_start: window_end - forecast_horizon]
            test_series_data = series_data[
                window_end - forecast_horizon : window_end
            ]

            train_series_list.append(train_series_data)
            test_series_list.append(test_series_data)

            # We use full length training series to train the model as we do not tune hyperparameters
            train_series_full_list.append(
                {
                    FieldName.TARGET: train_series_data,
                    FieldName.START: pd.Timestamp(train_start_time, freq=freq),
                }
            )

            test_series_full_list.append(
                {
                    FieldName.TARGET: series_data,
                    FieldName.START: pd.Timestamp(train_start_time, freq=freq),
                }
            )

        train_ds = ListDataset(train_series_full_list, freq=freq)
        test_ds = ListDataset(test_series_full_list, freq=freq)

    return train_ds, test_ds, forecast_horizon

# Parameters
# dataset_name - the name of the dataset
# lag - the number of past lags that should be used when predicting the next future value of time series
# input_file_name - name of the .tsf file corresponding with the dataset
# method - name of the forecasting method that you want to evaluate
# external_forecast_horizon - the required forecast horizon, if it is not available in the .tsf file
# integer_conversion - whether the forecasts should be rounded or not

def get_deep_nn_forecasts(
    dataset_name,
    num_windows,
    replay_window_num,
    lag,
    input_file_name,
    method,
    external_forecast_horizon=None,
    integer_conversion=False,
):
    print("Started loading " + dataset_name)

    (
        df,
        frequency,
        _,
        contain_missing_values,
        contain_equal_length,
    ) = loader.convert_tsf_to_dataframe(
        BASE_DIR + "/tsf_data/" + input_file_name, "NaN", VALUE_COL_NAME
    )

    final_forecasts = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    if isinstance(seasonality, list):
        seasonality = min(seasonality)  # Use to calculate MASE

    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        if external_forecast_horizon is None:
            raise Exception("Please provide the required forecast horizon")
        else:
            forecast_horizon = external_forecast_horizon

    start_exec_time = datetime.now()

    train_ds, test_ds, replay, forecast_horizon = get_window(df, num_windows, window_number, replay_window_num, freq)

    if method == "nbeats":
        estimator = NBEATSEstimator(
            freq=freq, context_length=lag, prediction_length=forecast_horizon
        )

    elif method == "transformer":
        estimator = TransformerEstimator(
            freq=freq, context_length=lag, prediction_length=forecast_horizon
        )

    predictor = estimator.train(training_data=train_ds)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )

    # Time series predictions
    forecasts = list(forecast_it)

    # Get median (0.5 quantile) of the 100 sample forecasts as final point forecasts
    for f in forecasts:
        final_forecasts.append(f.median)

    if integer_conversion:
        final_forecasts = np.round(final_forecasts)

    if not os.path.exists(BASE_DIR + "/results/fixed_horizon_forecasts/"):
        os.makedirs(BASE_DIR + "/results/fixed_horizon_forecasts/")

    # write the forecasting results to a file
    file_name = dataset_name + "_" + method + "_lag_" + str(lag)
    forecast_file_path = (
        BASE_DIR + "/results/fixed_horizon_forecasts/" + file_name + ".txt"
    )

    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(final_forecasts)

    finish_exec_time = datetime.now()

    # Execution time
    exec_time = finish_exec_time - start_exec_time
    print(exec_time)

    if not os.path.exists(BASE_DIR + "/results/fixed_horizon_execution_times/"):
        os.makedirs(BASE_DIR + "/results/fixed_horizon_execution_times/")

    with open(
        BASE_DIR + "/results/fixed_horizon_execution_times/" + file_name + ".txt", "w"
    ) as output_time:
        output_time.write(str(exec_time))

    # Write training dataset and the actual results into separate files, which are then used for error calculations
    # We do not use the built-in evaluation method in GluonTS as some of the error measures we use are not implemented in that
    temp_dataset_path = (
        BASE_DIR + "/results/fixed_horizon_forecasts/" + dataset_name + "_dataset.txt"
    )
    temp_results_path = (
        BASE_DIR + "/results/fixed_horizon_forecasts/" + dataset_name + "_results.txt"
    )

    with open(temp_dataset_path, "w") as output_dataset:
        writer = csv.writer(output_dataset, lineterminator="\n")
        writer.writerows(train_series_list)

    with open(temp_results_path, "w") as output_results:
        writer = csv.writer(output_results, lineterminator="\n")
        writer.writerows(test_series_list)

    if not os.path.exists(BASE_DIR + "/results/fixed_horizon_errors/"):
        os.makedirs(BASE_DIR + "/results/fixed_horizon_errors/")

    subprocess.call(
        [
            "Rscript",
            "--vanilla",
            BASE_DIR + "/utils/error_calc_helper.R",
            BASE_DIR,
            forecast_file_path,
            temp_results_path,
            temp_dataset_path,
            str(seasonality),
            file_name,
        ]
    )

    # Remove intermediate files
    os.system("rm " + temp_dataset_path)
    os.system("rm " + temp_results_path)


# Experiments
get_deep_nn_forecasts(
    "bitcoin", 9, "bitcoin_dataset_without_missing_values.tsf", "transformer", 30
)
get_deep_nn_forecasts(
    "bitcoin", 9, "bitcoin_dataset_without_missing_values.tsf", "nbeats", 30
)
