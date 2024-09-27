from typing import Callable

from dash import Input, Output, State, callback

from webviz_config_plugin._utils._fly_model import DataModel

from ._layout import LayoutElements

import numpy as np
import pandas as pd

import plotly.express as px

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
        Output(get_uuid(LayoutElements.GRAPH), "figure"),
        Input(get_uuid(LayoutElements.ATTRIBUTE_SELECTION_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.DATE_SELECTION_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.NUMBER_OF_BINS), "value"),
    )
    def _update_graph(
        selected_attribute: str,
        selected_date: str,
        number_of_bins: int,
    ) -> dict:

        ###############################################################
        # Load the data
        ###############################################################
        case_names = []
        values = []

        for case in data_model.case_names:
            data = data_model.load_data(case, selected_attribute, selected_date)
            mean = np.nanmean(data)
            case_names.append(case)
            values.append(mean)
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Case Name': case_names,
            'Value': values
        })

        ###############################################################
        # Create figure
        ###############################################################

        fig = px.histogram(df, x='Value', nbins=number_of_bins,
                   title=f"Distribution of {selected_attribute} amongst the cases")
        
        return fig
