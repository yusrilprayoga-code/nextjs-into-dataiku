# Trimming data
import pandas as pd


def trim_well_log(df, bottom_depth=None, top_depth=None, required_columns=None, mode='CUSTOM'):
    """
    Trim data well log berdasarkan validitas log dan (opsional) batas DEPTH dari user.

    Parameters:
    - df: DataFrame sumur
    - bottom_depth: batas bawah DEPTH (opsional)
    - top_depth: batas atas DEPTH (opsional)
    - required_columns: list kolom yang dicek validitasnya (default ['GR', 'RT', 'NPHI', 'RHOB'])

    Returns:
    - DataFrame yang sudah di-trim
    """
    if required_columns is None:
        required_columns = ['GR', 'RT', 'NPHI', 'RHOB']

    # Validasi data berdasarkan required_columns
    valid_index = {}
    for col in required_columns:
        valid_index[col] = df[(df[col] != -999.0) & (df[col].isna())].index

    depth_min = df['DEPTH'].min()
    depth_max = df['DEPTH'].max()

    # Konversi input ke float jika diberikan
    try:
        if bottom_depth is not None:
            bottom_depth = float(bottom_depth)
        if top_depth is not None:
            top_depth = float(top_depth)
    except Exception as e:
        raise ValueError(f"Gagal mengonversi DEPTH input ke float: {e}")

    if mode == 'DEPTH_ABOVE':
        if bottom_depth is None:
            raise ValueError("DEPTH harus diisi pada mode DEPTH_ABOVE':")
        df = df[~((df['DEPTH'] >= depth_min) & (
            df['DEPTH'] <= bottom_depth))].copy()

    elif mode == 'DEPTH_BELOW':
        if top_depth is None:
            raise ValueError("DEPTH harus diisi pada mode DEPTH_BELOW")
        df = df[~((df['DEPTH'] >= top_depth) & (
            df['DEPTH'] <= depth_max))].copy()

    elif mode == 'CUSTOM':
        if top_depth is None or bottom_depth is None:
            raise ValueError(
                "TOP_DEPTH dan BOTTOM_DEPTH harus diisi untuk mode CUSTOM")
        if top_depth > bottom_depth:
            raise ValueError(
                "TOP_DEPTH tidak boleh lebih besar dari BOTTOM_DEPTH")
        df = df[~((df['DEPTH'] >= top_depth) & (
            df['DEPTH'] <= bottom_depth))].copy()

    else:
        raise ValueError(f"Mode tidak dikenali: {mode}")

    return df
