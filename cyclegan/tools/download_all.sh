#!/usr/bin/env bash

DATASET_DIR=$1

declare -a FILE_LIST=("ae_photos" "apple2orange" "summer2winter_yosemite" "horse2zebra" "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo" "maps" "cityscapes" "facades" "iphone2dslr_flower")

## now loop through the above array
for i in "${FILE_LIST[@]}"
do
   bash ./download_dataset.sh $1 $i &
   # or do whatever with individual element of the array
done

wait
echo "All complete!"