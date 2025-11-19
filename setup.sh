#!/bin/bash

ENV_NAME="cvproject"
DEP_LIST="dependencies.txt"

PYTHON_CMD="python3"

echo "ENV '$ENV_NAME' creating..."
$PYTHON_CMD -m venv $ENV_NAME

PIP_PATH="./$ENV_NAME/bin/pip"
if [ ! -f "$PIP_PATH" ]; then
    PIP_PATH="./$ENV_NAME/Scripts/pip"
fi

if [ ! -f "$PIP_PATH" ]; then
    echo "cannot find pip. path: $PIP_PATH"
    exit 1
fi

echo "Checking dependencies..."

if [ -f "$DEP_LIST" ]; then
    echo "   - Upgrading pip..."
    "$PIP_PATH" install --upgrade pip 
    
    echo "   - Installing dependencies from $DEP_LIST..."
    "$PIP_PATH" install -r $DEP_LIST

    if [ $? -eq 0 ]; then
        echo "Dependencies installed."
        echo ""
        echo "Activate environment using:"
        echo "   source $ENV_NAME/bin/activate (Linux/macOS)"
        echo "   .\\$ENV_NAME\\Scripts\\activate (Windows CMD/PowerShell)"
    else
        echo "Dependency installation failed. Check the package names in $DEP_LIST."
    fi
else
    echo "Cannot find $DEP_LIST file. Installation skipped."
    echo "Environment created."
fi