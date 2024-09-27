from typing import Callable

from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
from dash import no_update
from dash import html, dcc
from dash.dependencies import ALL

from ._layout import LayoutElements

import numpy as np
import pandas as pd
import time
import json

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from deepdown import *
import os


###########################################################################
#
# Collection of Dash callbacks.
#
# The callback functions should retrieve Dash Inputs and States, utilize
# business logic and props serialization functionality for providing the
# JSON serializable Output for Dash properties callbacks.
#
# The callback Input and States should be converted from JSON serializable
# formats to strongly typed and filtered formats. Furthermore the callback
# can provide the converted arguments to the business logic for retrieving
# data or performing ad-hoc calculations.
#
# Results from the business logic is provided to the props serialization to
# create/build serialized data formats for the JSON serializable callback
# Output.
#
###########################################################################


def plugin_callbacks(get_uuid: Callable, ml_operator: MLOperator, train_dataset: Hdf5Dataset, validation_dataset: Hdf5Dataset):
    
    # Load Features
    @callback(
        Output(get_uuid(LayoutElements.FEATURES_CONTAINER), 'children'),
        Input(get_uuid(LayoutElements.LOAD_FEATURES_BUTTON), 'n_clicks'),
        prevent_initial_call=True
    )
    def _generate_feature_inputs(n_clicks):
        
        if n_clicks > 0:
            # Read the JSON file
            ml_operator.load_json()
            
            # Generate inputs dynamically
            children = [html.Div([
                html.Label(f'Input for {feature}'),
                dcc.Input(id={'type': 'dynamic-input', 'index': feature}, value=1, type='number')
            ]) for feature in ml_operator.features]

            # Load model
            print('\n\n LOADING TRAIN DATASET \n\n')
            train_dataset.initialise(file_path='.\\hdf5_data\\train.hdf5', transform=ToTensor())
            validation_dataset.initialise(file_path='.\\hdf5_data\\validation.hdf5', transform=ToTensor())
            ml_operator.initialise(train_dataset, validation_dataset)
            ml_operator.load_model()

            return children
        return []


    # Predict
    @callback(
        [Output(get_uuid(LayoutElements.PRINT_PREDICT), 'children'),
         Output(get_uuid(LayoutElements.DROPDOWN_DATE), 'options')],
        Input(get_uuid(LayoutElements.PREDICT_BUTTON), 'n_clicks'),
        [State({'type': 'dynamic-input', 'index': ALL}, 'value')],
        prevent_initial_call=True
    )
    def _predict(n_clicks, *args):

        print(f"Callback triggered with n_clicks: {n_clicks}, args: {args}")
        
        # Create a dict where the args are matched with features
        input_dict = {}
        for feature, arg in zip(ml_operator.features, args[0]):
            input_dict[feature] = float(arg)

        # Get initial inputs
        train_dataset.initialise(file_path='.\\hdf5_data\\validation.hdf5', transform=ToTensor())
        inputs = train_dataset[0][0]

        # Scale
        for i, feature in enumerate(ml_operator.features):
            inputs[i] *= input_dict[feature]

        # Predict
        ml_operator.prediction = ml_operator.predict(inputs.unsqueeze(0))

        print(f'shape of y_pred: {ml_operator.prediction.shape}')
        
        return f'grid_shape: {ml_operator.grid_shape}, inputs: {ml_operator.features}', [{'label': date, 'value': i} for i, date in enumerate(ml_operator.time_steps)]
    

    # Plot Prediction
    @callback(
        Output(get_uuid(LayoutElements.PLOT_PREDICTION), 'figure'),
        [Input(get_uuid(LayoutElements.DROPDOWN_DATE), 'value'),
         Input(get_uuid(LayoutElements.PREDICT_BUTTON), 'n_clicks')],
        prevent_initial_call=True
    )
    def _plot_prediction(date, n_clicks):
        
        if ml_operator.prediction is None:
            raise PreventUpdate
        
        print('\n')
        print(f'PLOTTING: ...')
        print(f'prediction shape: {ml_operator.prediction.shape}')
        print(f'date: {date}')
        print('\n')

        return Plot3D(ml_operator.prediction.detach().numpy()[0][date], options={'filter': 2, 'sample': 0.5})
        



