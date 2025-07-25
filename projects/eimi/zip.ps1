# PowerShell script to package the Chrome extension

# Create a zip file of the src directory
Compress-Archive -Path .\src\* -DestinationPath .\youtube-request-interceptor.zip -Force

Write-Host "Extension packaged as youtube-request-interceptor.zip"
Write-Host "You can now upload this file to the Chrome Web Store or use it for distribution." 