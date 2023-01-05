#!/bin/bash

objects=$(find ../../../../data/gold/images/RGB -maxdepth 1 -type d -printf '%P\n')
ignored=(bread tomato cabbage drill)
# echo $objects
for del in ${ignored[@]}
do
echo "removing $del"
objects=("${objects[@]/$del}")
done
echo $objects
for subdir in $objects
do
## echo "$subdir: $(ls -l '../../../../data/gold/images/RGB/'$subdir | grep -c ^d)"
echo "$(ls -l '../../../../data/gold/images/RGB/'$subdir | grep -c ^d)"
done