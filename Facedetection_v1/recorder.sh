#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/huy/code/godofeye/lib
python3 ./utils/camera_capture/capture.py $1--ip 10.10.46.139 \
    --user admin --password be123456 \
    --output .
