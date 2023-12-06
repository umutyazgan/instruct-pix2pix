#!/bin/bash
for i in {0..9}
do
    python edit_cli.py --resolution 384 --input ../examples/65_skull/views/cam00000$i.png --output ../examples/skull_out/output$i.png --edit "give the skull a mustache"
done
for i in {10..48}
do
    python edit_cli.py --resolution 384 --input ../examples/65_skull/views/cam0000$i.png --output ../examples/skull_out/output$i.png --edit "give the skull a mustache"
done
