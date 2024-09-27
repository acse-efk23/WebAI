from typing import Callable

from dash import Input, Output, State, callback

from webviz_config_plugin._utils._fly_model import DataModel

from ._layout import LayoutElements

import numpy as np
import pandas as pd

import plotly.express as px

from dash import html

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


def plugin_callbacks(get_uuid: Callable, data_model: DataModel):
    @callback(
        Output(get_uuid(LayoutElements.STATS), "children"),
        Input(get_uuid(LayoutElements.CASE_SELECTION_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.ATTRIBUTE_SELECTION_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.DATE_SELECTION_DROPDOWN), "value"),
    )
    def _update_graph(
        selected_case: str,
        selected_attribute: str,
        selected_date: str,
    ) -> dict:

        ###############################################################
        # Load the data
        ###############################################################
        values = data_model.load_data(selected_case, selected_attribute, selected_date)

        ###############################################################
        # Create figure
        ###############################################################
        
        # flatten 3d array
        values = values.flatten()

        # create children including statistical values for the selected data
        children = [
            html.Div([
                html.Div([
                    html.H6(f"Statistics for {selected_attribute} in {selected_case} on {selected_date}"),
                    html.Div([
                        html.Div(f"Mean: {np.nanmean(values):.2f}"),
                        html.Div(f"Median: {np.nanmedian(values):.2f}"),
                        html.Div(f"Standard Deviation: {np.nanstd(values):.2f}"),
                        html.Div(f"Min: {np.nanmin(values):.2f}"),
                        html.Div(f"Max: {np.nanmax(values):.2f}"),
                        html.Div(f"25th percentile: {np.nanpercentile(values, 25):.2f}"),
                        html.Div(f"75th percentile: {np.nanpercentile(values, 75):.2f}"),
                    ])
                ])
            ])
        ]
        
        return children
