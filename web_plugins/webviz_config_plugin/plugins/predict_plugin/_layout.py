from typing import Callable, List

import webviz_core_components as wcc

from dash import html
from dash import dcc

import plotly.graph_objs as go

######################################################################
#
# Collection of Dash layout and ID-ownership
#
# Defines the Dash layout, and builds the HTML-element structure
# for the plugin.
#
# Ownership of the layout element ID's, which is provided to the
# various callback Inputs, States and Outputs.
#
######################################################################


class LayoutElements:
    
    LOAD_FEATURES_BUTTON = "load-features-button"       # html.Button()
    
    FEATURES_CONTAINER = "features-container"           # html.Div()

    PREDICT_BUTTON = "predict-button"                   # html.Button()

    PRINT_PREDICT = "print-predict"                     # html.Div()

    DROPDOWN_DATE = "dropdown-date"                     # dcc.Dropdown()

    PLOT_PREDICTION = "plot-prediction"                 # dcc.Graph()


def main_layout(get_uuid: Callable):
    
    return wcc.FlexBox(
        children=[
            wcc.FlexColumn(
                children=wcc.Frame(
                    style={"height": "90vh"},
                    children=[


                        wcc.Selectors(
                            label="Input Parameters",
                            children=[

                                html.Div([

                                    ####################################################

                                    # Button load 
                                    html.Div([
                                        html.Button(children=['Load Features'], id=get_uuid(LayoutElements.LOAD_FEATURES_BUTTON), n_clicks=0,
                                                    style={'margin-bottom': '20px'})
                                    ]),

                                    ####################################################

                                    # Features container
                                    html.Div(id=get_uuid(LayoutElements.FEATURES_CONTAINER)),


                                ], style={'padding': '5px'}),

                            ]
                        ),


                        wcc.Selectors(
                            label="Predict",
                            children=[

                                html.Div([
                                    # Button load 
                                    html.Div([
                                        html.Button(children=['PREDICT'], id=get_uuid(LayoutElements.PREDICT_BUTTON), n_clicks=0,
                                                    style={'margin-bottom': '20px'})
                                    ]),

                                ], style={'padding': '5px'}),

                            ]
                        ),


                        wcc.Selectors(
                            label="Plot Options",
                            children=[

                                html.Div([
                                    
                                    # Dropdown date selection
                                    html.Div([
                                        html.Label('Select Date'),
                                        wcc.Dropdown(id=get_uuid(LayoutElements.DROPDOWN_DATE))
                                    ]),

                                ], style={'padding': '5px'}),

                            ]
                        ),


                    ],
                )
            ),

            wcc.FlexColumn(
                flex=4,
                children=[
                    
                    # Print predict result
                    html.Div([
                        html.Div(children=[html.Pre("Please click the button to make prediction.", 
                                                    id=get_uuid(LayoutElements.PRINT_PREDICT))]),
                    ]),

                    # Plot prediction
                    html.Div([
                        dcc.Graph(id=get_uuid(LayoutElements.PLOT_PREDICTION), style={'height': '500px', 'width': '100%'})
                    ]),

                ],
            ),
        ],
    )
