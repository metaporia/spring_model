
set shell:=["bash", "-c"]

default:
  just run

#poetry run python test.py
run:
  python revision.py


open: run
  feh --multiwindow *.png

open_each: run
  #!/usr/bin/env bash
  for i in *.png; 
  do 
    feh "$i";
  done
