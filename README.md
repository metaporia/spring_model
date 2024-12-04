# Spring-mass Model

To get started, run the following:

```
$ nix develop
$ just run # or
$ python revision.py
```

Then open the generated PNGs. An ASCII table with the raw data is printed to
stdout and an html version of it is written to ./table.html

Alternatively, go to jupyterlab online (the have a free beta online notebook interface),
paste `revision.py` into a python cell and run!

## Model script

- `revision.py` is the entry point. Just run it to generate table, image(s), &c.
- `run(initial_state: SimState) -> SimState` runs the actual sim
- `plot_all(..)` generates the plot(s)
- `show_table(..)` prints the ASCII table and writes `table.html` for current run

## Dependencies

NB: the dep list is conservative and may include some packages needed for poetry runtime.

Our project needed:
- numpy
- matplotlib
- pandas
- prettytable

Using nix/poetry it should sort itself out, but here's the dep list:

- contourpy       1.3.1       Python library for calculating contours of 2D quadrilateral grids
- cycler          0.12.1      Composable style cycles
- fonttools       4.55.0      Tools to manipulate font files
- kiwisolver      1.4.7       A fast implementation of the Cassowary constraint solver
- matplotlib      3.9.2       Python plotting package
- numpy           2.1.3       Fundamental package for array computing in Python
- packaging       24.2        Core utilities for Python packages
- pandas          2.2.3       Powerful data structures for data analysis, time series, and statistics
- pillow          11.0.0      Python Imaging Library (Fork)
- prettytable     3.12.0      A simple Python library for easily displaying tabular data in a visually appealing ASCII table format
- pyparsing       3.2.0       pyparsing module - Classes and methods to define and execute parsing grammars
- python-dateutil 2.9.0.post0 Extensions to the standard Python datetime module
- pytz            2024.2      World timezone definitions, modern and historical
- six             1.16.0      Python 2 and 3 compatibility utilities
- tzdata          2024.2      Provider of IANA time zone data
- wcwidth         0.2.13      Measures the displayed width of unicode strings in a terminal
