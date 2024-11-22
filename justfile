
set shell:=["bash", "-c"]

default:
  just run

#poetry run python test.py
run:
  python revision.py


open: run
  feh --auto-zoom --auto-reload --multiwindow *.png

open_each: run
  #!/usr/bin/env bash
  for i in *.png; 
  do 
    feh --auto-zoom --auto-reload -w "$i"& disown;
  done
