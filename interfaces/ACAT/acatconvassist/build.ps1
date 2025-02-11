# Define the path to the Python script or .spec file you want to build
$scriptPath = "ConvAssist.spec"

# Measure the time taken to run the PyInstaller build
$timeTaken = Measure-Command {
    Write-Host "Building ConvAssist.exe using PyInstaller"
    & pyinstaller --noconfirm $scriptPath
}

# Output the time taken
Write-Output "$timeTaken"

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
} 

Write-Host "Updating the version information in the build output"
& pyi-set_version .\file_version_info.txt .\dist\ConvAssist\ConvAssist.exe

if ($LASTEXITCODE -ne 0) {
    Write-Error "pyi-set_version failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
} 

Write-Host "Compressing the build output"
Compress-Archive -Path "dist\ConvAssist\_internal","dist\ConvAssist\ConvAssist.exe" -DestinationPath "ConvAssist.zip" -Force
