#!/bin/bash

model_path=$1
outputdir=${model_path#*/}"_predictions"

echo "output:" ${outputdir}

echo "compile..."
make
if [ ! -f "$model_path" ];then
        echo "ERROR:model does not exists! "
        return
fi

#check if output dir  exists
if [ ! -d $outputdir ];then
        echo "creating directoy:"$outputdir
        mkdir $outputdir;
fi

echo "empty "$outputdir
rm -rf $outputdir"/*";

cat cfg/damage_detection/new_validation2.txt | ./darknet  detect  cfg/damage_detection/yolo_newdata_damage_light_color_continue_0818.cfg ${model_path} &&
python mkhtml.py $2 $3 $outputdir &&
echo "created html"
