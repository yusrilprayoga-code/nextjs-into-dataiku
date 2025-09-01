# Dataiku Well Log Analysis Plugin

This plugin integrates a Next.js-based well log analysis application into Dataiku DSS.

## Features

- **Interactive Well Log Visualization**: View and analyze well log data with interactive plots
- **Data Processing Modules**: Various geoscience calculations (VSH, porosity, water saturation, etc.)
- **Crossplot Analysis**: Generate crossplots for well log interpretation
- **Quality Control**: Automated QC checks on well log data
- **Data Splicing**: Combine multiple well runs into composite logs

## Installation

1. **Copy Plugin Files**: Copy the entire plugin directory to your Dataiku `plugins` folder
2. **Install Dependencies**: Install required Python packages in the code environment
3. **Restart Dataiku**: Restart DSS to load the new plugin

## Configuration

### Web App Template
The plugin includes a web app template that serves the static Next.js application:

- **Template ID**: `tes1_t123`
- **Type**: Standard web app
- **Backend**: Python Flask application

### API Endpoints
The backend provides the following API endpoints:

- `/list-wells` - List available wells
- `/well-folder-files/<field>/<structure>/<well>` - Get files for a specific well
- `/get-well-columns` - Get column information from well files
- `/run-qc` - Run quality control
- `/run-interval-normalization` - Normalize well log intervals
- `/run-smoothing` - Apply smoothing to well logs
- `/run-vsh-calculation` - Calculate VSH (Volume of Shale)
- `/run-porosity-calculation` - Calculate porosity
- `/run-sw-calculation` - Calculate water saturation
- `/get-crossplot` - Generate crossplot data
- `/get-histogram-plot` - Generate histogram plots
- And many more...

## Usage

1. **Create Web App**: In Dataiku, create a new web app using the "Well Log Analysis" template
2. **Configure Data Sources**: Set up connections to your well log data
3. **Launch Application**: The web app will serve the Next.js interface
4. **Analyze Data**: Use the interactive tools to analyze your well log data

## Development

### Static Files
The static Next.js build is located in `resources/out/` and includes:
- HTML files for each page
- JavaScript chunks in `_next/static/chunks/`
- CSS files in `_next/static/css/`
- Static assets

### Backend Modules
Python modules are located in `python-lib/tes1/`:
- `crossplot.py` - Crossplot generation
- `data_processing.py` - General data processing
- `vsh_calculation.py` - VSH calculations
- `porosity.py` - Porosity calculations
- And many more specialized modules

### Building Static Files
To update the static build:

1. Make changes to the Next.js source code
2. Run `npm run build && npm run export`
3. Copy the `out/` directory to `resources/out/`
4. Update API URLs in JavaScript files if needed

## Security

- All API calls are proxied through Dataiku's backend
- Dataiku's authentication and authorization are respected
- File access is controlled by Dataiku's security model

## Troubleshooting

### Common Issues

1. **API Calls Failing**: Check that the backend is running and API endpoints are properly configured
2. **Static Files Not Loading**: Verify that `resources/out/` contains the correct build files
3. **CORS Issues**: Ensure API proxy is properly configured in the web app template

### Logs
Check Dataiku logs for backend errors and browser console for frontend errors.

## Support

For issues or questions:
1. Check the Dataiku community forums
2. Review the Next.js and Dataiku documentation
3. Contact the plugin maintainer
