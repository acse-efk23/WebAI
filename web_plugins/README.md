## How to add a new plugin

You should have cloned installed [webviz-config](https://github.com/acse-efk23/webviz-config) and this package in editable mode.

Follow the below steps to add a new plugin:

1. Go to webviz_config_plugin/plugins directory.

2. Pick one plugin, copy it, and change it according to your needs. There are simple plugins consisting of one Python file such as [_some_custom_plugin.py](https://github.com/acse-efk23/webviz-config-plugin/blob/c93d14846de2043a758c869e450ee5e11aa36299/webviz_config_plugin/plugins/_some_custom_plugin.py) and complex ones  such as [best_practise_plugin](https://github.com/acse-efk23/webviz-config-plugin/tree/c93d14846de2043a758c869e450ee5e11aa36299/webviz_config_plugin/plugins/best_practice_plugin).

3. Edit the file and class name, layout, callbacks, plotly figures.

4. Add new class to the `__init__.py` file. 

5. Add new plugin to the `setup.py` file's `entry_points` dictionary.

6. Install the package again in editable mode while your environment is activated:
```bash
pip install -e .
```

7. Add your new plugin to the `configuration.yaml` file:
```yaml
title: Data Visualisation

options:
  menu:
    initially_pinned: True
  plotly_theme:
    yaxis:
      showgrid: True
      gridcolor: white

layout:
  - page: My New Plugin
    content:
      - MyNewPlugin:
          data_path: ..\example_data\

```