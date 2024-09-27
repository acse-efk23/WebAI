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
        Input(get_uuid(LayoutElements.CASE_SELECTION_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.ATTRIBUTE_SELECTION_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.DATE_SELECTION_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.FILTER_TYPE_DROPDOWN), "value"),
        Input(get_uuid(LayoutElements.NUMBER_OF_STDDEV_INPUT), "value"),
        Input(get_uuid(LayoutElements.SAMPLE_RATE_INPUT), "value"),
    )
    def _update_graph(
        selected_case: str,
        selected_attribute: str,
        selected_date: str,
        filter_type: str,
        number_of_stddev: float,
        sample_rate: float,
    ) -> dict:

        ###############################################################
        # Load the data
        ###############################################################
        values = data_model.load_data(selected_case, selected_attribute, selected_date)

        ###############################################################
        # Create a dataframe
        ###############################################################
        shape = values.shape

        indices = np.indices((shape[0], shape[1], shape[2]))

        x = indices[0].flatten()
        y = indices[1].flatten()
        z = indices[2].flatten()
        values_flat = values.flatten()

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            selected_attribute: values_flat  # Use the attribute variable to name the column
        })

        # drop nan values
        df.dropna(inplace=True)

        ###############################################################
        # Options
        ###############################################################

        if filter_type == "stddev" and number_of_stddev is not None:
            mean = df[selected_attribute].mean()
            stddev = df[selected_attribute].std()
            threshold = mean + number_of_stddev * stddev
            df = df[df[selected_attribute] > threshold]

        if sample_rate:
            df = df.sample(frac=sample_rate)


        ###############################################################
        # Create figure
        ###############################################################
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color=selected_attribute,
                            opacity=0.1,)

        title = (
            f"<b>Case:</b> {selected_case}<br>"
            f"<b>Attribute:</b> {selected_attribute}<br>"
            f"<b>Date:</b> {selected_date}"
        )

        fig.update_layout(
            height=800,
            title=title
            )
        
        fig.update_traces(marker=dict(size=5))
        
        return fig
