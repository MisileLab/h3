# PowerShell script to package the Chrome extension for production

# Get version from manifest.json
$manifestPath = Join-Path -Path $PSScriptRoot -ChildPath "src\manifest.json"
$manifest = Get-Content -Path $manifestPath | ConvertFrom-Json
$version = $manifest.version

# Clean up any previous build artifacts
$buildDir = Join-Path -Path $PSScriptRoot -ChildPath "build"
if (Test-Path $buildDir) {
    Remove-Item -Path $buildDir -Recurse -Force
}

# Create build directory
New-Item -Path $buildDir -ItemType Directory | Out-Null

# Copy only necessary files to build directory
$filesToCopy = @(
    "background.js",
    "content.js",
    "manifest.json",
    "rules.json"
)

foreach ($file in $filesToCopy) {
    Copy-Item -Path (Join-Path -Path $PSScriptRoot -ChildPath "src\$file") -Destination $buildDir
}

# Create zip file with version number
$zipFileName = "youtube-bot-detector-v$version.zip"
$zipFilePath = Join-Path -Path $PSScriptRoot -ChildPath $zipFileName

Compress-Archive -Path "$buildDir\*" -DestinationPath $zipFilePath -Force

Write-Host "Extension packaged as $zipFileName"
Write-Host "You can now upload this file to the Chrome Web Store or use it for distribution." 