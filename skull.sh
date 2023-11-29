#!/bin/bash

for i in {0..9}
do
  python edit_cli.py --input ../data/65_skull/views/cam00000$i.png --output ../data/give_the_skull_a_mustache/cam00000$i.png --edit "give the skull a mustache"
done
for i in {10..48}
do
  python edit_cli.py --input ../data/65_skull/views/cam0000$i.png --output ../data/give_the_skull_a_mustache/cam0000$i.png --edit "give the skull a mustache"
done
