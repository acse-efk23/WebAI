from typing import Type, Union

from dash.development.base_component import Component
from webviz_config import WebvizPluginABC

from webviz_config_plugin._utils._fly_model import DataModel

from ._callbacks import plugin_callbacks
from ._layout import main_layout


from pathlib import Path


class Slice(WebvizPluginABC):
    """
    This Webviz plugin, to illustrate a best practice on code structure and how to
    separate code for clarity and usability.
    """

    def __init__(
            self,
            data_path: Path
            ) -> None:

        super().__init__(stretch=True)

        self._data_model = DataModel(data_path=data_path)
        
        self.set_callbacks()

    @property
    def layout(self) -> Union[str, Type[Component]]:
        return main_layout(
            get_uuid=self.uuid,
            case_names=self._data_model.case_names,
            attribute_names=self._data_model.attributes,
            dates=self._data_model.unique_dates,
        )

    def set_callbacks(self) -> None:
        plugin_callbacks(self.uuid, self._data_model)
