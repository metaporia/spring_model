
set shell:=["bash", "-c"]

default:
  just run

#poetry run python test.py
run:
  python revision.py


# prefer open to open_each (quit all windows simultaneously)
open: run
  feh --auto-zoom --auto-reload --multiwindow *.png

open_each: run
  #!/usr/bin/env bash
  for i in test/{acceleration,position,velocity}*.png; 
  do 
    feh --auto-zoom --auto-reload -w "$i"& disown;
  done

clean:
  rm *.png; rm ./result
