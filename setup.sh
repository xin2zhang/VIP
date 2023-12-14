#!/bin/bash

echo "Build kernels for svgd..."
cd vip/kernel
RESULT="$(python setup.py build_ext -i 2>&1)" #2>/dev/null
status=$?
if [ $status -eq 0 ]; then
    echo "Building kernel succeeds"
else
    echo "Error: $RESULT"
fi

echo "Build 2d fwi..."
cd ../../forward/fwi2d
RESULT="$(python setup.py build_ext -i 2>&1)"
status=$?
if [ $status -eq 0 ]; then
    echo "Building 2d fwi succeeds"
else
    echo "Error: $RESULT"
fi

echo "Build 2d tomography..."
cd ../tomo2d
RESULT="$(python setup.py build_ext -i 2>&1)"
status=$?
if [ $status -eq 0 ]; then
    echo "Building 2d tomography succeeds"
else
    echo "Error: $RESULT"
fi

cd ../../

if [ "$1" = "install" ]; then
    echo "Install vip..."
    pip install -e .
fi
