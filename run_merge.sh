#!/bin/bash
cd ../deepvenv/bin/
source ./activate
cd ../../panopticSeg/
python3 merge_panoptic.py
deactivate