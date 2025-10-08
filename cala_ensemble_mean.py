#!/usr/bin/env python3
"""
程式說明：計算WRF系集資料的ensemble mean
輸入一個包含member維度的NetCDF檔案，計算所有變數在member軸上的平均值，
輸出檔案保留原始檔案結構，但member維度變為1（保留維度但僅有一個index）

Author: CYC (YakultSmoothie)
Create date: 2025-10-08
"""

import argparse
import xarray as xr
import numpy as np
import os
from pathlib import Path

# ============================================================================
# 參數設定區
# ============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Calculate ensemble mean from WRF output with member dimension',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run sample:
    python3 cal_ensemble_mean.py -i eth.nc
    python3 cal_ensemble_mean.py -i eth.nc -o eth_mean.nc
    python3 cal_ensemble_mean.py -i ./data/tk.nc -o ./output/tk_ensemble_mean.nc
    python3 cal_ensemble_mean.py -i eth.nc -c 5 
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input NetCDF file with member dimension')
    
    parser.add_argument('-o', '--output', default=None,
                       help='Output NetCDF file (default: input_ensemble_mean.nc)')
    
    parser.add_argument('-c', '--compress', type=int, default=0, choices=range(0, 10),
                       help='Compression level (0-9, 0=no compression, default=1)')
    
    return parser.parse_args()

# ============================================================================
# 主程式
# ============================================================================
if __name__ == '__main__':
    args = parse_arguments()
    
    # 處理輸入檔案路徑
    input_file = args.input
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # 處理輸出檔案路徑
    if args.output is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_ensemble_mean{input_path.suffix}"
    else:
        output_file = args.output
    
    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print("="*70)
    print("== Ensemble Mean along Member Axis Calculator ==")
    print("="*70)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Compression level: {args.compress}")
    print("")
    
    # ============================================================================
    # OPEN - 讀取檔案
    # ============================================================================
    print("Opening NetCDF file...")
    ds = xr.open_dataset(input_file)
    
    # ============================================================================
    # 讀檔 information
    # ============================================================================
    print(f"Dataset information:")
    print(f"    Dimensions: {dict(ds.sizes)}")
    print(f"    Variables: {list(ds.data_vars)}")
    print(f"    Coordinates: {list(ds.coords)}")
    
    # 檢查是否有member維度
    if 'member' not in ds.sizes:
        raise ValueError("Input file does not contain 'member' dimension")
    
    n_members = ds.sizes['member']
    print(f"    Number of ensemble members: {n_members}")
    print("")
    
    # ============================================================================
    # DEFINE - 計算系集平均
    # ============================================================================
    print("Calculating ensemble mean...")
    
    # 複製原始dataset以保留所有屬性和座標
    ds_mean = ds.copy(deep=True)
    
    # 找出所有包含member維度的變數
    vars_with_member = [var for var in ds.data_vars if 'member' in ds[var].sizes]
    print(f"    Variables with member dimension: {vars_with_member}")
    
    # 建立新的dataset，先處理member座標
    # 計算每個變數的ensemble mean並建立新的data_vars字典
    data_vars_mean = {}
    
    for var in vars_with_member:
        print(f"    Processing variable: {var}")
        print(f"        Original shape: {ds[var].shape}")
        
        # 計算member軸的平均值，保持維度
        mean_data = ds[var].mean(dim='member', keepdims=True)
        
        print(f"        Mean shape: {mean_data.shape}")
        print(f"        Mean data range: [{float(mean_data.min().values):.4f}, {float(mean_data.max().values):.4f}]")
        
        data_vars_mean[var] = mean_data
    
    # 處理不包含member維度的變數（直接複製）
    vars_without_member = [var for var in ds.data_vars if 'member' not in ds[var].dims]
    for var in vars_without_member:
        data_vars_mean[var] = ds[var]
    
    # 處理座標變數
    coords_mean = {}
    for coord in ds.coords:
        if coord == 'member':
            # member座標只保留第一個值
            coords_mean[coord] = ds[coord].isel(member=[0])
        elif 'member' in ds[coord].dims:
            # 其他包含member維度的座標變數，取第一個member
            coords_mean[coord] = ds[coord].isel(member=0)
        else:
            # 不包含member維度的座標直接複製
            coords_mean[coord] = ds[coord]
    
    # 建立新的dataset
    ds_mean = xr.Dataset(
        data_vars=data_vars_mean,
        coords=coords_mean,
        attrs=ds.attrs
    )
    
    print("")
    print(f"Output dataset information:")
    print(f"    Dimensions: {dict(ds_mean.sizes)}")
    print(f"    Member dimension size: {ds_mean.sizes['member']}")
    print("")
    
    # ============================================================================
    # WRITE AND SAVE - 寫出檔案
    # ============================================================================
    print(f"Writing ensemble mean to: {output_file}")
    
    # 設定壓縮參數
    encoding = {}
    if args.compress == 0:
        # 不壓縮
        for var in ds_mean.variables:
            encoding[var] = {'zlib': False}
        print(f"    Compression: None")
    else:
        # 使用指定的壓縮層級
        for var in ds_mean.variables:
            encoding[var] = {'zlib': True, 'complevel': args.compress}
        print(f"    Compression: zlib level {args.compress}")
    
    ds_mean.to_netcdf(output_file, encoding=encoding)
    
    # 關閉檔案
    ds.close()
    ds_mean.close()
    
    # 顯示輸出檔案資訊
    output_size = os.path.getsize(output_file) / (1024**2)  # MB
    print(f"    Output file size: {output_size:.2f} MB")
    print("")
    print("="*70)
    print("== Ensemble mean calculation completed successfully! ==")
    print("="*70)
