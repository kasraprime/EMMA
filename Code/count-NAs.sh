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

# cat ../data/gold_text.tsv | egrep -wi 'n/a|na|nan|ns'
# cat ../data/gold_text.tsv | egrep -wi 'n/a|na|nan|ns' | cut -d' ' -f4,5 # keeping object name and nan only
# cat ../data/gold_text.tsv | egrep -wi 'n/a|na|nan|ns' | cut -d' ' -f4 # keeping instance name only
# cat ../data/gold_text.tsv | egrep -wi 'n/a|na|nan|ns' | cut -d' ' -f4 | sed -e 's/\(_[0-9]_[0-9]\)*$//g' # keeping object names only

for subdir in $objects
do
# cat ../data/gold_text.tsv | egrep -wi 'n/a|na|nan|ns' | cut -d' ' -f4 | sed -e 's/\(_[0-9]_[0-9]\)*$//g' | grep -c $subdir
echo "$(cat ../data/gold_text.tsv | egrep -wi 'n/a|na|nan|ns' | cut -d' ' -f4 | sed -e 's/\(_[0-9]_[0-9]\)*$//g' | grep -c $subdir)"
done