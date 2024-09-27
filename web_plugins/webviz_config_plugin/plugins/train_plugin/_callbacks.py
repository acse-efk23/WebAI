from typing import Callable

from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
from dash import no_update

from ._layout import LayoutElements

import numpy as np
import pandas as pd
import time

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


def plugin_callbacks(get_uuid: Callable, train_dataset: Hdf5Dataset, validation_dataset: Hdf5Dataset, ml_operator: MLOperator):
    
    # Load HDF5 dataset

    @callback(        
        Output(get_uuid(LayoutElements.PRINT_DATASET), "children"),
        [Input(get_uuid(LayoutElements.LOAD_BUTTON), 'n_clicks'),],
        [State(get_uuid(LayoutElements.BATCH_SIZE), 'value'),
         State(get_uuid(LayoutElements.EPOCHS), 'value'),],
         prevent_initial_call=True
    )
    def _load_hdf5(n_clicks, batch_size, epochs):

        if n_clicks is None:
            return 'Please click the button to load hdf5 file.'
        
        if not os.path.exists('.\\hdf5_data\\train.hdf5'):
            return 'Go back to Load Data page and generate the HDF5 dataset first!'

        train_dataset.initialise(file_path='.\\hdf5_data\\train.hdf5', transform=ToTensor())

        validation_dataset.initialise(file_path='.\\hdf5_data\\validation.hdf5', transform=ToTensor())

        ml_operator.initialise(train_dataset, validation_dataset, int(batch_size), int(epochs))

        return f'''HDF5 dataset is loaded with the following shapes and lengths:
        Train dataset: {train_dataset[0][0].shape}, {train_dataset[0][1].shape}, Length: {len(train_dataset)}
        Validation dataset: {validation_dataset[0][0].shape}, {validation_dataset[0][1].shape}, Length: {len(validation_dataset)}
        Batch size: {batch_size}, Epochs: {epochs}
        Number of parameters: {sum(p.numel() for p in ml_operator.model.parameters())}'''
    

    # RUNNING

    @callback(
        [Output(get_uuid(LayoutElements.PRINT_TRAIN), 'children'),
         Output(get_uuid(LayoutElements.INTERVAL), 'disabled'),
         Output(get_uuid(LayoutElements.INTERVAL), 'n_intervals'),
         Output(get_uuid(LayoutElements.LOSS_GRAPH), 'figure'),
         Output(get_uuid(LayoutElements.PREDICTION_GRAPH), 'figure')],
        [Input(get_uuid(LayoutElements.TRAIN_BUTTON), 'n_clicks'),
         Input(get_uuid(LayoutElements.INTERVAL), 'n_intervals'),],
        prevent_initial_call=True,
        running=[(Output(get_uuid(LayoutElements.INTERVAL), "disabled"), True, False)]
    )
    def _update_output(n_clicks, n_intervals):

        epochs = 1
        figure_loss, figure_prediction = ml_operator.train_visual(epochs=epochs)

        if n_intervals * epochs < ml_operator.EPOCHS:
            return f'Training... epoch: {ml_operator.current_epochs}', \
                no_update, no_update, figure_loss, figure_prediction
        else:
            ml_operator.save_model()
            return f'Training finished!', \
                True, 0, figure_loss, figure_prediction












