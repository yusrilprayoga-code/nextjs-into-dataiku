@echo off
REM Dataiku Well Log Analysis Plugin Deployment Script
REM This script helps deploy the plugin to a Dataiku instance

echo ========================================
echo Dataiku Well Log Analysis Plugin Setup
echo ========================================

set PLUGIN_DIR=%~dp0
set PLUGIN_NAME=tes1

echo Plugin directory: %PLUGIN_DIR%
echo Plugin name: %PLUGIN_NAME%

echo.
echo Step 1: Checking plugin structure...
if not exist "%PLUGIN_DIR%plugin.json" (
    echo ERROR: plugin.json not found!
    pause
    exit /b 1
)

if not exist "%PLUGIN_DIR%web-app-templates" (
    echo ERROR: web-app-templates directory not found!
    pause
    exit /b 1
)

if not exist "%PLUGIN_DIR%resources" (
    echo ERROR: resources directory not found!
    pause
    exit /b 1
)

echo ✓ Plugin structure looks good!

echo.
echo Step 2: Checking static files...
if not exist "%PLUGIN_DIR%resources\out\index.html" (
    echo WARNING: Static build files not found in resources/out/
    echo Please ensure you have built the Next.js application and copied the 'out' folder here.
    echo.
) else (
    echo ✓ Static files found!
)

echo.
echo Step 3: Deployment instructions:
echo.
echo To deploy this plugin to Dataiku:
echo.
echo 1. Copy the entire plugin folder to your Dataiku plugins directory:
echo    - Windows: C:\Dataiku\DataScienceStudio\dss_home\plugins\plugins\%PLUGIN_NAME%
echo    - Linux: /home/dataiku/dss_home/plugins/plugins/%PLUGIN_NAME%
echo.
echo 2. Install Python dependencies in the code environment:
echo    - Go to Administration > Code Envs
echo    - Create or update the code environment for this plugin
echo    - Install required packages (pandas, numpy, flask, etc.)
echo.
echo 3. Restart Dataiku DSS
echo.
echo 4. Create a new web app:
echo    - Go to Web Apps section
echo    - Create new web app using "Well Log Analysis" template
echo.
echo 5. Configure data connections:
echo    - Set up connections to your well log data sources
echo    - Configure file paths and dataset references
echo.

echo.
echo Additional notes:
echo - Ensure all Python modules in python-lib/tes1/ are properly implemented
echo - Test the web app after deployment
echo - Check browser console and Dataiku logs for any errors
echo.

pause
