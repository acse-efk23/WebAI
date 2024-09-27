from typing import Callable
import os

from dash import Input, Output, State, callback

from ._layout import LayoutElements

import numpy as np
import pandas as pd

import plotly.express as px

from deepdown import SimDataset

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


def plugin_callbacks(get_uuid: Callable, sim_dataset: SimDataset):

    # Input Sim Path
    @callback(
        [        
            Output(get_uuid(LayoutElements.SELECT_STATIC_PROPERTIES), 'options'),
            Output(get_uuid(LayoutElements.SELECT_DYNAMIC_PROPERTIES), 'options'),
            Output(get_uuid(LayoutElements.SHOW_SUMMARY), 'children'),
        ],
        Input(get_uuid(LayoutElements.INPUT_SIM_PATH), "value"),
        prevent_initial_call=True
    )
    def _update_property_options(input_sim_path):

        if input_sim_path is None:
            return [], [], 'Please enter a .sim path!'

        input_sim_path = input_sim_path.replace('\\', '\\\\')

        print(input_sim_path)

        # check if the input path is a valid .sim file
        if os.path.exists(input_sim_path):
            print('\nFILE EXISTS\n')
        else:
            print('\nFILE DOES NOT EXIST\n')
            return [], [], 'Please enter a valid .sim path!'

        if sim_dataset.initialise(input_sim_path) == 1:
            return [], [], 'Please enter a .sim path!'

        return  [{'label': prop, 'value': prop} for prop in sim_dataset.static_properties], [{'label': prop, 'value': prop} for prop in sim_dataset.dynamic_properties], sim_dataset.summary()
    

    # Generate HDF5
    @callback(        
        Output(get_uuid(LayoutElements.CREATE_HDF5_DIV), "children"),
        [Input(get_uuid(LayoutElements.BUTTON), 'n_clicks'),],
        [State(get_uuid(LayoutElements.SELECT_STATIC_PROPERTIES), 'value'),
        State(get_uuid(LayoutElements.SELECT_DYNAMIC_PROPERTIES), 'value'),
        State(get_uuid(LayoutElements.SELECT_TIMESTEPS), 'value'),
        State(get_uuid(LayoutElements.SELECT_CASES), 'value'),
        State(get_uuid(LayoutElements.SELECT_VALIDATION_RATIO), 'value'),],
        prevent_initial_call=True
    )
    def _generate_hdf5(n_clicks, static_properties, dynamic_properties, timestep_amount, case_amount, validation_ratio):

        properties_list = []
        properties_list.extend(static_properties)
        properties_list.extend(dynamic_properties)

        sim_dataset.filter_properties(properties_list)

        sim_dataset.filter_time_steps(int(timestep_amount))

        sim_dataset.filter_cases(int(case_amount))

        sim_dataset.create_hdf5(validation_ratio=float(validation_ratio))

        # Save selected static properties into a json file
        sim_dataset.save_json()

        return 'Training dataset is generated!'

    