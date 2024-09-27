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
        # Compute slice
        ###############################################################
        slice = np.nanmean(values, axis=2)


        ###############################################################
        # Create figure
        ###############################################################
        # 2d plot of the slice
        fig = px.imshow(slice, aspect='auto', origin='lower', labels=dict(x='X', y='Y', color='Value'))

        # make it square
        fig.update_xaxes(scaleanchor='y', scaleratio=1)

        # Leave more space between the title and the plot
        fig.update_layout(margin=dict(t=100))

        title = (
            f"<b>Case:</b> {selected_case}<br>"
            f"<b>Attribute:</b> {selected_attribute}<br>"
            f"<b>Date:</b> {selected_date}"
        )

        fig.update_layout(
            height=600,
            title=title
            )
        
        return fig
