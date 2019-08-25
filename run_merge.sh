#!/bin/bash
cd ../deepvenv/bin/
source ./activate
cd ../../panopticSeg/
python3 main_merge.py
deactivate