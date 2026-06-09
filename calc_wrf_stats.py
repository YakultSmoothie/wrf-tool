#!/usr/bin/env python3
# =============================================================================================
# ==== INFOMATION ========
# ========================
# 檔名: calc_wrf_stats.py
# 功能: 計算多個WRF輸出檔案的統計量，並輸出至新的netCDF檔案
# 作者: CYC
# create: 2025-03-17 at JET
# update: 2025-03-18 - CYC
#   - 移除--axis和--notime選項
#   - 確保每個檔案只處理一個時間點
#   - 修正TIME變數處理，使用第一個檔案的時間
#   - 修正 xarray FutureWarning: 使用 Dataset.sizes 替代 Dataset.dims 作為字典
#   - 調整錯誤處理和提示訊息格式
#
# Description:
#   此程式讀取一或多個WRF輸出檔案(netCDF格式)，計算指定的統計量
#   (均值、標準差、最小值、最大值、四分位數等)，並將結果輸出
#   至新的netCDF檔案。可處理任意數量的輸入檔案。
#   適用於計算WRF系集模擬的統計結果或時間統計量。
# ============================================================================================

import sys
import os
import argparse
import numpy as np
import xarray as xr
from datetime import datetime
import time
import warnings

#args_str = ' '.join(sys.argv[0:])    # 擷取輸入元素
#print(f"\n======= RUN: {args_str} =========\n")    # 顯示輸入元素
print(f"\n======= RUN: sys.argv[0] =========\n")    # 顯示輸入元素

#------------------------------------
def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='計算多個WRF輸出檔案的統計量，並輸出至新的netCDF檔案',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 1. 計算多個檔案的均值
     python3 calc_wrf_stats.py -i wrfout_d01_1 wrfout_d01_2 wrfout_d01_3 -o output_mean.nc --function mean

  # 2. 計算多個檔案的標準差
     python3 calc_wrf_stats.py -i wrfout_d01_* -o output_std.nc -f std

  # 3. 計算多個統計量(均值和標準差)
     python3 calc_wrf_stats.py -i wrfout_d01_* -o output_stats.nc -f mean,std

  # 4. 計算多個統計量並分別輸出
     python3 calc_wrf_stats.py -i wrfout_d01_* -o /path/output_{function}.nc -f mean,std,min,max

  # 5. 計算特定變數的統計量
     python3 calc_wrf_stats.py -i wrfout_d01_* -o output_stats.nc -f mean --vars T2,PSFC,U10,V10

  # 6. 不使用dask進行處理(適用於無dask環境)
     python3 calc_wrf_stats.py -i wrfout_d01_* -o output_mean.nc -f mean --no-dask

注意:
  - 輸入檔案必須是WRF netCDF格式
  - 所有輸入檔案必須有相同的網格設定
  - 時間設定: 每個檔案只處理第一個時間點
  - function參數可以使用逗號分隔多個統計量
  - 若輸出檔名中含有{function}，則會替換為對應的統計量名稱

支援的統計量:
  - mean: 平均值, sum: 總和,
  - std: 標準差,  - var: 變異數,
  - min: 最小值,  - max: 最大值,
  - q1: 第一四分位數(25%), - q2: 中位數(50%), - q3: 第三四分位數(75%)

A, V, D: CYC, v1.1, 2025-03-23
        """)

    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='wrfout檔案路徑 (可指定多個)')

    parser.add_argument('-o', '--output', required=True,
                       help='輸出檔案路徑')

    parser.add_argument('-f', '--function', default='mean',
                       help='統計量函數 (預設: mean)，可用逗號分隔指定多個')

    parser.add_argument('-V', '--vars',
                       help='要計算的變數，用逗號分隔 (預設: 全部變數)')

    parser.add_argument('--chunks', type=int, default=None,
                       help='讀取資料時的區塊大小 (bytes)，用於大檔案 (預設: None)')

    parser.add_argument('--no-dask', action='store_true',
                       help='不使用dask進行處理，適用於無dask環境')

    parser.add_argument('-info', '--information', action='store_true',
                       help='顯示詳細資訊')
    
    return parser.parse_args()

#------------------------------------
def manual_read_files(file_paths, variables=None):
    """手動讀取並合併多個檔案，不使用dask
       確保每個檔案只處理第一個時間點

    Args:
        file_paths: 檔案路徑列表
        variables: 要保留的變數列表(可選)

    Returns:
        合併後的xarray Dataset和第一個檔案的時間變數
    """
    datasets = []
    first_time_var = None  # 保存第一個檔案的時間變數
    time_var_name = None   # 時間變數的名稱
    
    for i, file_path in enumerate(file_paths):
        print(f"  讀取檔案 {i+1}/{len(file_paths)}: {file_path}")
        ds = xr.open_dataset(file_path)
        
        # 確保只處理第一個時間點
        if 'Time' in ds.sizes and ds.sizes['Time'] > 1:
            print(f"  注意: 檔案 {file_path} 有多個時間點 ({ds.sizes['Time']})，只使用第一個時間點")
            ds = ds.isel(Time=0)
        
        # 保存第一個檔案的時間變數
        if i == 0:
            # 檢查時間變數名稱 (WRF通常使用'Times'或'XTIME')
            if 'Times' in ds.variables:
                time_var_name = 'Times'
                first_time_var = ds['Times'].copy(deep=True)
            elif 'XTIME' in ds.variables:
                time_var_name = 'XTIME'
                first_time_var = ds['XTIME'].copy(deep=True)
            elif 'Time' in ds.coords:
                time_var_name = 'Time'
                first_time_var = ds['Time'].copy(deep=True)
            
            print(f"  保存第一個檔案的時間變數: {time_var_name}")
        
        # 如果指定了變數子集，僅保留這些變數
        if variables:
            # 檢查請求的變數是否都存在
            available_vars = [v for v in variables if v in ds.data_vars]
            ds = ds[available_vars]
        
        # 移除時間相關變數以避免計算錯誤
        for time_name in ['Times', 'XTIME']:
            if time_name in ds.variables and time_name != time_var_name:
                ds = ds.drop_vars(time_name)
        
        # 添加檔案維度
        ds = ds.expand_dims(dim={"file": [i]})
        datasets.append(ds)

    # 合併所有數據集
    combined_ds = xr.concat(datasets, dim="file")
    
    return combined_ds, first_time_var, time_var_name

#------------------------------------
def get_statistics(dataset, func_name):
    """計算指定的統計量

    Args:
        dataset: xarray Dataset
        func_name: 統計量函數名稱

    Returns:
        計算結果的 xarray Dataset
    """
    # 檢查是否存在時間相關變數
    time_vars = []
    for var_name, var in dataset.variables.items():
        if var.dtype.kind in 'SU':  # 字符串類型變數
            if var_name in ['Times', 'XTIME'] or 'time' in var_name.lower():
                time_vars.append(var_name)
    
    # 從計算中排除時間變數
    if time_vars:
        print(f"  從統計計算中排除時間變數: {time_vars}")
        calc_ds = dataset.drop_vars(time_vars)
    else:
        calc_ds = dataset
    
    # 計算統計量
    if func_name == 'mean':
        result = calc_ds.mean(dim='file')
    elif func_name == 'std':
        result = calc_ds.std(dim='file')
    elif func_name == 'var':
        result = calc_ds.var(dim='file')
    elif func_name == 'min':
        result = calc_ds.min(dim='file')
    elif func_name == 'max':
        result = calc_ds.max(dim='file')
    elif func_name == 'sum':
        result = calc_ds.sum(dim='file')
    elif func_name == 'q1':
        result = calc_ds.quantile(0.25, dim='file')
    elif func_name == 'q2':
        result = calc_ds.quantile(0.5, dim='file')
    elif func_name == 'q3':
        result = calc_ds.quantile(0.75, dim='file')
    else:
        raise ValueError(f"不支援的統計量函數: {func_name}")
    
    return result

#------------------------------------
def main():
    """主程序"""
    # 解析命令列參數
    args = parse_arguments()

    # 顯示基本資訊
    print(f"處理 {len(args.input)} 個輸入檔案")
    for i, file in enumerate(args.input):
        print(f"  {i+1:3d}: {file}")
    print(f"輸出檔案: {args.output}")
    print(f"統計量函數: {args.function}")
    print(f"使用dask: {not args.no_dask}")

    # 解析統計量函數
    functions = args.function.split(',')

    # 解析需要處理的變數
    variables = None
    if args.vars:
        variables = args.vars.split(',')
        print(f"僅計算以下變數: {variables}")

    # 讀取所有輸入檔案
    try:
        print(f"正在讀取輸入檔案...")
        
        # 保存第一個檔案的時間變數
        first_time_var = None
        time_var_name = None

        if args.no_dask:
            # 使用手動方法讀取並合併檔案，不依賴dask
            print("使用手動方法讀取檔案(不使用dask)...")
            ds, first_time_var, time_var_name = manual_read_files(args.input, variables)
        else:
            try:
                # 嘗試使用dask
                print("嘗試使用dask讀取檔案...")
                
                # 先讀取第一個檔案以保存時間變數
                first_ds = xr.open_dataset(args.input[0])
                
                # 保存第一個檔案的時間變數
                if 'Times' in first_ds.variables:
                    time_var_name = 'Times'
                    first_time_var = first_ds['Times'].copy(deep=True)
                elif 'XTIME' in first_ds.variables:
                    time_var_name = 'XTIME'
                    first_time_var = first_ds['XTIME'].copy(deep=True)
                elif 'Time' in first_ds.coords:
                    time_var_name = 'Time'
                    first_time_var = first_ds['Time'].copy(deep=True)
                
                # 確保只取第一個時間點
                if 'Time' in first_ds.sizes and first_ds.sizes['Time'] > 1:
                    first_ds = first_ds.isel(Time=0)
                    if 'Time' in first_time_var.dims:
                        first_time_var = first_time_var.isel(Time=0)
                
                print(f"  保存第一個檔案的時間變數: {time_var_name}")
                
                # 讀取所有檔案
                if args.chunks:
                    ds = xr.open_mfdataset(args.input, concat_dim='file', combine='nested',
                                         chunks={'file': 1, 'Time': args.chunks})
                else:
                    ds = xr.open_mfdataset(args.input, concat_dim='file', combine='nested')
                
                # 確保只處理第一個時間點
                if 'Time' in ds.sizes and ds.sizes['Time'] > 1:
                    print(f"  注意: 檔案有多個時間點 ({ds.sizes['Time']})，只使用第一個時間點")
                    ds = ds.isel(Time=0)
                
            except (ImportError, ValueError) as e:
                # 如果dask不可用，回退到手動方法
                print(f"警告: 無法使用dask({str(e)})，改用手動方法讀取檔案...")
                ds, first_time_var, time_var_name = manual_read_files(args.input, variables)

        # 如果指定了變數子集但還沒篩選(dask方法下)，僅保留這些變數
        if variables and not args.no_dask:
            # 檢查請求的變數是否都存在
            missing_vars = [v for v in variables if v not in ds.data_vars]
            if missing_vars:
                print(f"警告: 以下請求的變數在輸入檔案中不存在: {missing_vars}")
                variables = [v for v in variables if v in ds.data_vars]

            # 僅保留指定的變數
            ds = ds[variables]

        print(f"\n數據維度: {ds.sizes}")
        print(f"資料形狀範例: {next(iter(ds.data_vars.values())).shape}")
        if args.information:
            # 顯示讀取的資料基本資訊
            print(f"可用變數: {list(ds.data_vars)}")

        # 計算所有請求的統計量
        for func_name in functions:
            print(f"\n正在計算 {func_name}...")

            # 計算統計量
            result_ds = get_statistics(ds, func_name)
            
            # 將第一個檔案的時間變數加回結果中
            if first_time_var is not None and time_var_name is not None:
                print(f"  將第一個檔案的時間變數 {time_var_name} 加入結果")
                result_ds[time_var_name] = first_time_var

            # 添加全域屬性
            result_ds.attrs['title'] = f'WRF {func_name} statistics'
            result_ds.attrs['description'] = f'Statistics calculated from {len(args.input)} WRF files'
            result_ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            result_ds.attrs['input_files'] = ', '.join([os.path.basename(f) for f in args.input])
            result_ds.attrs['statistic_function'] = func_name
            result_ds.attrs['time_info'] = f'Time from first file: {args.input[0]}'

            # 決定輸出檔名
            if '{function}' in args.output:
                output_file = args.output.replace('{function}', func_name)
            else:
                # 如果有多個函數但輸出檔案名稱中沒有{function}，附加函數名
                if len(functions) > 1:
                    base, ext = os.path.splitext(args.output)
                    output_file = f"{base}_{func_name}{ext}"
                else:
                    output_file = args.output

            # 建立輸出目錄
            output_dir = os.path.dirname(os.path.abspath(output_file))
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 儲存結果
            print(f"正在儲存結果至 {output_file}...")
            print(f"處理的變數數量: {len(result_ds.data_vars)}")

            # 列出即將處理的變數和基本統計資訊
            if args.information:
                print("  變數摘要:")
                for i, (var_name, var_data) in enumerate(result_ds.data_vars.items(), 1):
                    # 獲取變數的基本統計資訊
                    try:
                        if var_name != time_var_name:  # 跳過時間變數
                            var_np = var_data.values
                            var_min = float(np.nanmin(var_np)) if np.any(~np.isnan(var_np)) else "N/A"
                            var_max = float(np.nanmax(var_np)) if np.any(~np.isnan(var_np)) else "N/A"
                            var_mean = float(np.nanmean(var_np)) if np.any(~np.isnan(var_np)) else "N/A"
                            var_shape = str(var_data.shape)
    
                            # 顯示變數訊息
                            print(f"  {i:3d}. {var_name:<15} | 形狀: {var_shape:<15} | 範圍: {var_min:.4g} to {var_max:.4g} | 平均: {var_mean:.4g}")
                        else:
                            print(f"  {i:3d}. {var_name:<15} | 時間變數，使用第一個檔案的值")
                    except Exception as e:
                        # 如果計算統計量出錯，只顯示基本信息
                        print(f"  {i:3d}. {var_name:<15} | 形狀: {str(var_data.shape):<15} | 統計訊息計算錯誤: {str(e)}")

            # 設定每個變數的壓縮選項
            #encoding = {}
            #for var in result_ds.data_vars:
            #    if var != time_var_name:  # 不壓縮時間變數
            #        encoding[var] = {'zlib': True, 'complevel': 4}
            
            # 保存結果
            # print("to_netcdf....")
            # print(f"encoding = {encoding}")
            # result_ds.to_netcdf(output_file, encoding=encoding)
            # result_ds.to_netcdf(output_file)
            # print(f"已成功儲存結果: {output_file}")

            # 逐變數寫入 netCDF 檔案
            print(f"正在逐變數寫入到 {output_file}...")
            
            # 如果檔案已存在，先刪除它
            if os.path.exists(output_file):
                print(f"  刪除現有檔案: {output_file}")
                os.remove(output_file)
            
            # 首先儲存座標資訊和全域屬性
            coords_ds = xr.Dataset(coords=result_ds.coords, attrs=result_ds.attrs)
            print(f"1. 儲存座標資訊和全域屬性...")
            coords_ds.to_netcdf(output_file, mode='w')
            print(f"  完成座標資訊寫入")
            
            # 逐個變數寫入
            total_vars = len(result_ds.data_vars)
            for i, (var_name, var_data) in enumerate(result_ds.data_vars.items(), 1):
                print(f"2.{i}/{total_vars} 寫入變數: {var_name}...")
                
                # 檢查是否為時間變數
                is_time_var = False
                if var_name == time_var_name:
                    is_time_var = True
                    print(f"  此為時間變數: {var_name}")
                
                # 創建只包含單一變數的資料集
                try:
                    var_ds = xr.Dataset({var_name: var_data})
                    
                    # 根據變數類型設定壓縮選項
                    if not is_time_var:  # 時間變數通常不壓縮
                        encoding = {var_name: {'zlib': True, 'complevel': 4}}
                        var_ds.to_netcdf(output_file, mode='a', encoding=encoding)
                    else:
                        var_ds.to_netcdf(output_file, mode='a')
                    
                    print(f"  成功寫入變數: {var_name}")
                except Exception as e:
                    print(f"  寫入變數 {var_name} 時發生錯誤: {str(e)}")
                    # 繼續處理其他變數，而不是直接中止
                    continue
            
            print(f"已成功儲存結果: {output_file}")

        # 關閉資料集
        ds.close()

    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#---------------------------------------
if __name__ == "__main__":
    main()

#===========================================================================================================
#print(f"\n======= RUN END: {args_str} =========\n")
print(f"\n======= RUN END: sys.argv[0] =========\n")    # 顯示輸入元素
