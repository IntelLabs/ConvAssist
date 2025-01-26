# Define the path to the Python script or .spec file you want to build
$scriptPath = "ConvAssist.spec"

# Measure the time taken to run the PyInstaller build
$timeTaken = Measure-Command {
    pyinstaller --noconfirm --log-level INFO $scriptPath
}

# Output the time taken
Write-Output "$timeTaken"