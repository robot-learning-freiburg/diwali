#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "calvin" ]
then
    cd DIWALI_DATA_DIR/ && mkdir calvin && cd calvin
    echo "Downloading calvin task_D_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip task_D_D.zip && rm task_D_D.zip
    echo "saved folder: task_D_D"
elif [ "$1" = "calvin_arm" ]
then
    cd DIWALI_DATA_DIR/ && mkdir calvin_human && cd calvin_human
    echo "Downloading calvin task_D_D_arm ..."
    wget http://diwali.cs.uni-freiburg.de/download/data/task_D_D_arm.zip
    unzip task_D_D_arm.zip && rm task_D_D_arm.zip
    echo "saved folder: task_D_D_arm"
elif [ "$1" = "calvin_wm" ]
then
    cd DIWALI_DATA_DIR/
    echo "Downloading calvin wm task ..."
    wget http://diwali.cs.uni-freiburg.de/download/wm.zip
    unzip wm.zip && rm wm.zip
    echo "saved folder: wm"
elif [ "$1" = "calvin_features" ]
then
    cd DIWALI_DATA_DIR/
    echo "Downloading sim task ..."
    wget http://diwali.cs.uni-freiburg.de/download/features.zip
    unzip features.zip && rm features.zip
    echo "saved folder: features"
elif [ "$1" = "calvin_rewcls" ]
then
    cd DIWALI_DATA_DIR/ && mkdir calvin_rewcls && cd calvin_rewcls
    echo "Downloading calvin rewcls task ..."
    wget http://diwali.cs.uni-freiburg.de/download/rewcls.zip
    unzip rewcls.zip && rm rewcls.zip
    echo "saved folder: rewcls"
else
    echo "Failed: Usage download.sh calvin | calvin_arm | calvin_wm | calvin_features | calvin_rewcls"
    exit 1
fi