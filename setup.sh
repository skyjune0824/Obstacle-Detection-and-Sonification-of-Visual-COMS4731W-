#!/bin/bash

ENV_NAME="cvproject"
DEP_LIST="dependencies.txt"
DEPTHANYTHING_PATH="models/Depth-Anything-V2"

PYTHON_CMD="python"

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

"pwd"
"ls"

if [ -d "$DEPTHANYTHING_PATH" ]; then
    DA_REQ="$DEPTHANYTHING_PATH/requirements.txt"

    if [ -f "$DA_REQ" ]; then
        echo ">>> Installing Depth Anything V2 dependencies..."
        "$PIP_PATH" install -r "$DA_REQ" || {
            echo "ERROR: Failed to install Depth Anything V2 dependencies."
            exit 1
        }

    else
        echo "WARNING: Depth Anything V2 requirements.txt not found — skipping."
    fi
else
    echo "WARNING: Depth Anything V2 folder not found at $DEPTHANYTHING_PATH — skipping."
fi