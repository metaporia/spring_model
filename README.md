# Spring-mass Model

To get started, run the following:

```
$ nix develop
$ just run # or
$ python revision.py
```

Then open the generated PNGs. An ASCII table with the raw data is printed to
stdout and an html version of it is written to ./table.html

## Model script

- `revision.py` is the entry point. Just run it to generate table, image(s), &c.
- `run(initial_state: SimState) -> SimState` runs the actual sim
- `plot_all(..)` generates the plot(s)
- `show_table(..)` prints the ASCII table and writes `table.html` for current run
