from typing import Callable, List

import webviz_core_components as wcc

from dash import html
from dash import dcc

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
    
    INPUT_SIM_PATH = "sim_path_input"                   # dcc.Input()
    SELECT_STATIC_PROPERTIES = "select_static_properties"      # wcc.Select()
    SELECT_DYNAMIC_PROPERTIES = "select_dynamic_properties"     # wcc.Select()
    SELECT_TIMESTEPS = "select_timesteps"               # dcc.Input()
    SELECT_VALIDATION_RATIO = "select_validation_ratio" # dcc.Input()
    SELECT_CASES = "select_cases"                       # wcc.Select()
    BUTTON = "create_dataset_button"                    # html.Button()
    SHOW_SUMMARY = "show_summary"                       # html.Pre()
    CREATE_HDF5_DIV = "create-hdf5-div"                 # html.Div()

def main_layout(get_uuid: Callable):
    
    return wcc.FlexBox(
        children=[
            wcc.FlexColumn(
                children=wcc.Frame(
                    style={"height": "90vh"},
                    children=[


                        wcc.Selectors(
                            label="Prepare Data",
                            children=[

                                html.Div([

                                    # Enter .sim path
                                    html.Div([
                                        html.Label('Enter .sim Path:', style={'font-weight': 'bold'}),
                                        dcc.Input(
                                            id=get_uuid(LayoutElements.INPUT_SIM_PATH),
                                        )
                                    ]),

                                    # Select inputs
                                    html.Div([
                                        html.Label('Select Static Inputs:', style={'font-weight': 'bold'}),
                                        wcc.Select(
                                            id=get_uuid(LayoutElements.SELECT_STATIC_PROPERTIES),
                                        )
                                    ]),

                                    # Select outputs
                                    html.Div([
                                        html.Label('Select Dynamic Outputs:', style={'font-weight': 'bold'}),
                                        wcc.Select(
                                            id=get_uuid(LayoutElements.SELECT_DYNAMIC_PROPERTIES),
                                        )
                                    ]),

                                ], style={'padding': '5px'}),

                            ]
                        ),

                        wcc.Selectors(
                            label="Optional Settings",
                            children=[

                                html.Div([

                                    # Select timesteps
                                    html.Div([
                                        html.Label('Enter Timesteps:', style={'font-weight': 'bold'}),
                                        dcc.Input(
                                            id=get_uuid(LayoutElements.SELECT_TIMESTEPS),
                                            value=2
                                        )
                                    ]),

                                    # Select cases
                                    html.Div([
                                        html.Label('Enter Cases:', style={'font-weight': 'bold'}),
                                        dcc.Input(
                                            id=get_uuid(LayoutElements.SELECT_CASES),
                                            value=2
                                        )
                                    ]),

                                    # Select validation ratio
                                    html.Div([
                                        html.Label('Enter Validation Ratio:', style={'font-weight': 'bold'}),
                                        dcc.Input(
                                            id=get_uuid(LayoutElements.SELECT_VALIDATION_RATIO),
                                            value=0.5
                                        )
                                    ]),

                                ], style={'padding': '5px'}),

                            ]
                        ),

                        # Create hdf5 dataset button
                        html.Div([
                            html.Button(children=['Create training data'], id=get_uuid(LayoutElements.BUTTON), n_clicks=0,
                                        style={'margin-top': '20px'}),
                        ]),

                    ],
                )
            ),

            wcc.FlexColumn(
                flex=4,
                children=[
                    
                    # Show summary
                    html.Div([
                        html.H4("Summary:"),
                        html.Div(children=[html.Pre("Please enter the Intersect simulation results path (.sim).", 
                                                    id=get_uuid(LayoutElements.SHOW_SUMMARY))]),
                    ]),

                    # HIDDEN DIV to store selected properties
                    html.Div(id=get_uuid(LayoutElements.CREATE_HDF5_DIV),),
                    
                ],
            ),
        ],
    )
