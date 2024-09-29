#!/bin/bash

# Start timing the pyinstaller build
echo "Starting pyinstaller build..."
start_time=$(date +%s)

# Run pyinstaller
pyinstaller --clean --noconfirm continuous_predict.spec

# End timing the pyinstaller build
end_time=$(date +%s)

# Calculate the duration
duration=$((end_time - start_time))

# Print the duration
echo "PyInstaller build took $duration seconds."
