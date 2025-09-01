# This file is the actual code for the Python backend of your webapp tes1_t123

import dataiku
import pandas as pd
import json
import os
from flask import request, jsonify
from dataiku import pandasutils as pdu

# Import your custom Python modules
from tes1.crossplot import *
from tes1.data_processing import *
from tes1.depth_matching import *
from tes1.dgsa import *
from tes1.dns_dnsv import *
from tes1.dns_dnsv_plot import *
from tes1.fill_missing import *
from tes1.folder_nav_service import *
from tes1.gsa import *
from tes1.histogram import *
from tes1.iqual import *
from tes1.las_plotting_service import *
from tes1.module1_service import *
from tes1.ngsa import *
from tes1.plotting_service import *
from tes1.porosity import *
from tes1.qc_service import *
from tes1.rgbe_rpbe import *
from tes1.rgsa import *
from tes1.rt_r0 import *
from tes1.rt_r0_plot import *
from tes1.rwa import *
from tes1.splicing import *
from tes1.structures_service import *
from tes1.sw import *
from tes1.swgrad import *
from tes1.swgrad_plot import *
from tes1.trim_data import *
from tes1.vsh_calculation import *
from tes1.vsh_dn import *

@app.route('/list-wells')
def list_wells():
    """List all wells in the configured directory"""
    try:
        # Get wells directory from request or use default
        wells_dir = request.args.get('full_path', '/path/to/wells')

        # Use your folder navigation service
        wells = get_well_folders(wells_dir)
        return jsonify(wells)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/well-folder-files/<field>/<structure>/<well>')
def well_folder_files(field, structure, well):
    """Get files in a specific well folder"""
    try:
        # Use your folder navigation service
        files = get_well_files(field, structure, well)
        return jsonify({"csv_files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-well-columns')
def get_well_columns():
    """Get columns from well files"""
    try:
        file_paths = request.json.get('file_paths', [])
        columns = get_columns_from_files(file_paths)
        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-qc')
def run_qc():
    """Run quality control on uploaded files"""
    try:
        files = request.json.get('files', [])
        result = perform_qc(files)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-interval-normalization')
def run_interval_normalization():
    """Run interval normalization"""
    try:
        params = request.json.get('params', {})
        file_paths = request.json.get('file_paths', [])
        selected_wells = request.json.get('selected_wells', [])
        selected_intervals = request.json.get('selected_intervals', [])

        result = perform_interval_normalization(params, file_paths, selected_wells, selected_intervals)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-smoothing')
def run_smoothing():
    """Run smoothing on well data"""
    try:
        params = request.json.get('params', {})
        file_paths = request.json.get('file_paths', [])
        selected_wells = request.json.get('selected_wells', [])
        selected_intervals = request.json.get('selected_intervals', [])

        result = perform_smoothing(params, file_paths, selected_wells, selected_intervals)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-vsh-calculation')
def run_vsh_calculation():
    """Run VSH calculation"""
    try:
        params = request.json.get('params', {})
        full_path = request.json.get('full_path', '')
        selected_wells = request.json.get('selected_wells', [])
        selected_intervals = request.json.get('selected_intervals', [])
        selected_zones = request.json.get('selected_zones', [])

        result = perform_vsh_calculation(params, full_path, selected_wells, selected_intervals, selected_zones)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-porosity-calculation')
def run_porosity_calculation():
    """Run porosity calculation"""
    try:
        params = request.json.get('params', {})
        full_path = request.json.get('full_path', '')
        selected_wells = request.json.get('selected_wells', [])
        selected_intervals = request.json.get('selected_intervals', [])
        selected_zones = request.json.get('selected_zones', [])

        result = perform_porosity_calculation(params, full_path, selected_wells, selected_intervals, selected_zones)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-sw-calculation')
def run_sw_calculation():
    """Run SW calculation"""
    try:
        params = request.json.get('params', {})
        full_path = request.json.get('full_path', '')
        selected_wells = request.json.get('selected_wells', [])
        selected_intervals = request.json.get('selected_intervals', [])
        selected_zones = request.json.get('selected_zones', [])

        result = perform_sw_calculation(params, full_path, selected_wells, selected_intervals, selected_zones)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-crossplot')
def get_crossplot():
    """Generate crossplot data"""
    try:
        selected_wells = request.json.get('selected_wells', [])
        x_col = request.json.get('x_col', '')
        y_col = request.json.get('y_col', '')
        selected_intervals = request.json.get('selected_intervals', [])

        result = generate_crossplot(selected_wells, x_col, y_col, selected_intervals)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-histogram-plot')
def get_histogram_plot():
    """Generate histogram plot"""
    try:
        selected_wells = request.json.get('selected_wells', [])
        log_column = request.json.get('log_column', '')
        selected_intervals = request.json.get('selected_intervals', [])

        result = generate_histogram_plot(selected_wells, log_column, selected_intervals)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-log-percentiles')
def get_log_percentiles():
    """Get log percentiles for normalization"""
    try:
        file_paths = request.json.get('file_paths', [])
        selected_wells = request.json.get('selected_wells', [])
        selected_intervals = request.json.get('selected_intervals', [])
        log_column = request.json.get('log_column', '')

        result = calculate_log_percentiles(file_paths, selected_wells, selected_intervals, log_column)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/structure-folders/<field>/<structure>')
def structure_folders(field, structure):
    """Get structure folders"""
    try:
        result = get_structure_folders(field, structure)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-depth-matching')
def run_depth_matching():
    """Run depth matching"""
    try:
        well_name = request.json.get('well_name', '')
        result = perform_depth_matching(well_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-splicing')
def run_splicing():
    """Run well splicing"""
    try:
        params = request.json.get('params', {})
        selected_wells = request.json.get('selected_wells', [])
        run1_well = request.json.get('run1_well', '')
        run2_well = request.json.get('run2_well', '')
        run1_file_path = request.json.get('run1_file_path', '')
        run2_file_path = request.json.get('run2_file_path', '')

        result = perform_splicing(params, selected_wells, run1_well, run2_well, run1_file_path, run2_file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-splicing-plot')
def get_splicing_plot():
    """Get splicing plot"""
    try:
        file_path = request.json.get('file_path', '')
        result = generate_splicing_plot(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-gwd')
def run_gwd():
    """Run GWD calculation"""
    try:
        result = perform_gwd_calculation()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run-trim-data')
def run_trim_data():
    """Run data trimming"""
    try:
        params = request.json.get('params', {})
        file_paths = request.json.get('file_paths', [])
        selected_wells = request.json.get('selected_wells', [])
        selected_intervals = request.json.get('selected_intervals', [])

        result = perform_trim_data(params, file_paths, selected_wells, selected_intervals)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/flag-missing')
def flag_missing():
    """Flag missing data"""
    try:
        logs_to_check = request.json.get('logs_to_check', [])
        full_path = request.json.get('full_path', '')
        file_paths = request.json.get('file_paths', [])
        selected_wells = request.json.get('selected_wells', [])

        result = flag_missing_data(logs_to_check, full_path, file_paths, selected_wells)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/fill-flagged-missing')
def fill_flagged_missing():
    """Fill flagged missing data"""
    try:
        logs_to_fill = request.json.get('logs_to_fill', [])
        max_consecutive_nan = request.json.get('max_consecutive_nan', 3)
        full_path = request.json.get('full_path', '')
        file_paths = request.json.get('file_paths', [])
        selected_wells = request.json.get('selected_wells', [])

        result = fill_missing_data(logs_to_fill, max_consecutive_nan, full_path, file_paths, selected_wells)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add more API endpoints as needed based on your frontend requirements
