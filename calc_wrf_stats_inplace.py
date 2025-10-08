#!/usr/bin/env python3
# =============================================================================================
# ==== INFOMATION ========
# ========================
# 檔名: calc_wrf_stats_inplace.py
# 功能: 計算多個WRF輸出檔案的統計量，並替換第一個檔案中的變數值
# 作者: CYC
# create: 2025-03-19 at JET
#
# Description:
#   此程式讀取一或多個WRF輸出檔案(netCDF格式)，計算指定的統計量
#   (均值、標準差、最小值、最大值、四分位數等)，並將結果寫入到
#   第一個輸入檔案的複製版本中。保留原始檔案的結構和屬性，但
#   變數值被替換為統計結果。
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
import shutil

print(f"\n======= RUN: {os.path.basename(sys.argv[0])} =========\n")

#------------------------------------
def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='計算多個WRF輸出檔案的統計量，並替換第一個檔案中的變數值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 1. 計算多個檔案的均值，替換第一個檔案中的變數值
     python3 calc_wrf_stats_inplace.py -i wrfout_d01_1 wrfout_d01_2 wrfout_d01_3 -o output_mean.nc --function mean

  # 2. 計算多個檔案的標準差
     python3 calc_wrf_stats_inplace.py -i wrfout_d01_* -o output_std.nc -f std

  # 3. 計算特定變數的均值
     python3 calc_wrf_stats_inplace.py -i wrfout_d01_* -o output_mean.nc -f mean --vars T2,PSFC,U10,V10

  # 4. 不使用dask進行處理(適用於無dask環境)
     python3 calc_wrf_stats_inplace.py -i wrfout_d01_* -o output_mean.nc -f mean --no-dask

注意:
  - 輸入檔案必須是WRF netCDF格式
  - 所有輸入檔案必須有相同的網格設定
  - 時間設定: 每個檔案只處理第一個時間點
  - 程式會創建第一個檔案的複製，然後用統計結果替換其中的變數值
  - 輸出檔案會保留原始檔案的所有維度和屬性，但變數值會被替換

支援的統計量:
  - mean: 平均值, sum: 總和,
  - std: 標準差,  - var: 變異數,
  - min: 最小值,  - max: 最大值,
  - q1: 第一四分位數(25%), - q2: 中位數(50%), - q3: 第三四分位數(75%)

A, V, D: CYC, v1.1, 2025-03-25
        """)

    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='wrfout檔案路徑 (可指定多個)')

    parser.add_argument('-o', '--output', required=True,
                       help='輸出檔案路徑')

    parser.add_argument('-f', '--function', default='mean',
                       help='統計量函數 (預設: mean)')

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
        合併後的xarray Dataset和第一個檔案的原始Dataset
    """
    datasets = []
    first_ds = None

    for i, file_path in enumerate(file_paths):
        print(f"  讀取檔案 {i+1}/{len(file_paths)}: {file_path}")
        ds = xr.open_dataset(file_path)

        # 確保只處理第一個時間點
        if 'Time' in ds.sizes and ds.sizes['Time'] > 1:
            print(f"  注意: 檔案 {file_path} 有多個時間點 ({ds.sizes['Time']})，只使用第一個時間點")
            ds = ds.isel(Time=0)

        # 保存第一個檔案的完整Dataset
        if i == 0:
            first_ds = ds.copy(deep=True)
            print(f"  已保存第一個檔案作為模板")

        # 如果指定了變數子集，僅保留這些變數
        if variables:
            # 檢查請求的變數是否都存在
            available_vars = [v for v in variables if v in ds.data_vars]
            if i == 0:
                print(f"  僅處理以下變數: {available_vars}")
            ds = ds[available_vars]

        # 添加檔案維度
        ds = ds.expand_dims(dim={"file": [i]})
        datasets.append(ds)

    # 合併所有數據集
    combined_ds = xr.concat(datasets, dim="file")

    return combined_ds, first_ds

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
def replace_variables(template_ds, stats_ds):
    """用統計結果替換模板中的變數值，並保留原始的資料類型

    Args:
        template_ds: 模板Dataset (第一個檔案)
        stats_ds: 含有統計結果的Dataset

    Returns:
        更新後的Dataset
    """
    result_ds = template_ds.copy(deep=True)

    # 獲取可能要替換的變數列表
    replace_vars = [var for var in stats_ds.data_vars if var in template_ds.data_vars]
    print(f"  將替換以下 {len(replace_vars)} 個變數: {replace_vars[:5]}...")
    if len(replace_vars) > 5:
        print(f"  ...以及 {len(replace_vars)-5} 個更多變數")

    # 收集需要特殊處理的整數變數
    integer_vars = []

    # 逐個替換變數
    for var_name in replace_vars:
        try:
            # 檢查形狀是否匹配
            if template_ds[var_name].shape != stats_ds[var_name].shape:
                print(f"  警告: 變數 {var_name} 的形狀不匹配，無法替換")
                print(f"    模板形狀: {template_ds[var_name].shape}, 統計結果形狀: {stats_ds[var_name].shape}")
                continue

            # 檢查原始變數是否為整數類型
            original_dtype = template_ds[var_name].dtype
            if np.issubdtype(original_dtype, np.integer):
                integer_vars.append(var_name)
                # 對整數類型變數，將統計結果四捨五入並轉換為整數
                result_ds[var_name].values = np.round(stats_ds[var_name].values).astype(original_dtype)
                print(f"  已替換整數變數: {var_name} (四捨五入並轉換為 {original_dtype})")
            else:
                # 替換浮點數變數值
                result_ds[var_name].values = stats_ds[var_name].values
                print(f"  已替換變數: {var_name}")
        except Exception as e:
            print(f"  替換變數 {var_name} 時出錯: {str(e)}")

    if integer_vars:
        print(f"\n共處理 {len(integer_vars)} 個整數類型變數")

    return result_ds

#------------------------------------
def prepare_encoding(template_ds):
    """從模板Dataset準備編碼設定

    Args:
        template_ds: 模板Dataset (第一個檔案)

    Returns:
        包含編碼設定的字典
    """
    encoding = {}

    # 遍歷所有變數
    for var_name, var in template_ds.variables.items():
        # 初始化此變數的編碼設定
        var_encoding = {}

        # 從原始變數的編碼中獲取關鍵設定
        original_encoding = var.encoding

        # 複製關鍵設定
        for key in ['_FillValue', 'dtype', 'scale_factor', 'add_offset', 'zlib', 'complevel']:
            if key in original_encoding:
                var_encoding[key] = original_encoding[key]

        # 確保整數類型變數有適當的_FillValue
        if np.issubdtype(var.dtype, np.integer) and '_FillValue' not in var_encoding:
            # 如果是整數類型但沒有_FillValue，添加一個適當的預設值
            if var.dtype == np.int8:
                var_encoding['_FillValue'] = np.int8(-127)
            elif var.dtype == np.int16:
                var_encoding['_FillValue'] = np.int16(-32767)
            elif var.dtype == np.int32:
                var_encoding['_FillValue'] = np.int32(-2147483647)
            elif var.dtype == np.int64:
                var_encoding['_FillValue'] = np.int64(-9223372036854775807)
            elif var.dtype == np.uint8:
                var_encoding['_FillValue'] = np.uint8(255)
            elif var.dtype == np.uint16:
                var_encoding['_FillValue'] = np.uint16(65535)
            elif var.dtype == np.uint32:
                var_encoding['_FillValue'] = np.uint32(4294967295)
            elif var.dtype == np.uint64:
                var_encoding['_FillValue'] = np.uint64(18446744073709551615)

        # 如果原始數據已壓縮，保持壓縮
        if 'zlib' not in var_encoding:
            var_encoding['zlib'] = True
            var_encoding['complevel'] = 4

        # 保存此變數的編碼設定
        if var_encoding:
            encoding[var_name] = var_encoding

    return encoding

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

    # 解析需要處理的變數
    variables = None
    if args.vars:
        variables = args.vars.split(',')
        print(f"僅計算以下變數: {variables}")

    # 確保輸出目錄存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 讀取所有輸入檔案
    try:
        print(f"正在讀取輸入檔案...")

        # 保存第一個檔案的完整Dataset
        template_ds = None

        if args.no_dask:
            # 使用手動方法讀取並合併檔案，不依賴dask
            print("使用手動方法讀取檔案(不使用dask)...")
            ds, template_ds = manual_read_files(args.input, variables)
        else:
            try:
                # 嘗試使用dask
                print("嘗試使用dask讀取檔案...")

                # 先讀取第一個檔案作為模板
                template_ds = xr.open_dataset(args.input[0])
                print(f"  template file: {args.input[0]}")

                # 確保模板只保留第一個時間點
                if 'Time' in template_ds.sizes and template_ds.sizes['Time'] > 1:
                    print(f"  注意: 模板檔案有多個時間點 ({template_ds.sizes['Time']})，只使用第一個時間點")
                    template_ds = template_ds.isel(Time=0)

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
                ds, template_ds = manual_read_files(args.input, variables)

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
        print(f"模板維度: {template_ds.sizes}")

        if args.information:
            # 顯示讀取的資料基本資訊
            print(f"可用變數: {list(ds.data_vars)}")

        # 計算指定的統計量
        print(f"\n正在計算 {args.function}...")
        stats_ds = get_statistics(ds, args.function)

        # 用統計結果替換模板中的變數值
        print(f"\n正在用 {args.function} 結果替換模板中的變數值...")
        result_ds = replace_variables(template_ds, stats_ds)

        # 添加全域屬性
        result_ds.attrs['title'] = template_ds.attrs.get('title', 'WRF files') + f' - {args.function} statistics'
        result_ds.attrs['description'] = f'Statistics ({args.function}) calculated from {len(args.input)} WRF files'
        result_ds.attrs['modified_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_ds.attrs['input_files'] = ', '.join([os.path.basename(f) for f in args.input])
        result_ds.attrs['statistic_function'] = args.function
        result_ds.attrs['template_file'] = os.path.basename(args.input[0])

        # 列出統計結果的基本訊息
        if args.information:
            print("\n變數統計資訊:")
            for i, (var_name, var_data) in enumerate(result_ds.data_vars.items(), 1):
                # 獲取變數的基本統計資訊
                try:
                    if var_name in stats_ds.data_vars:  # 被替換的變數
                        var_np = var_data.values
                        var_min = float(np.nanmin(var_np)) if np.any(~np.isnan(var_np)) else "N/A"
                        var_max = float(np.nanmax(var_np)) if np.any(~np.isnan(var_np)) else "N/A"
                        var_mean = float(np.nanmean(var_np)) if np.any(~np.isnan(var_np)) else "N/A"
                        var_shape = str(var_data.shape)

                        # 顯示被替換的變數訊息
                        print(f"  {i:3d}. {var_name:<15} | 已替換 | 形狀: {var_shape:<15} | 範圍: {var_min:.4g} to {var_max:.4g} | 平均: {var_mean:.4g}")
                    else:
                        # 顯示未替換的變數訊息
                        print(f"  {i:3d}. {var_name:<15} | 未替換 | 形狀: {str(var_data.shape):<15}")
                except Exception as e:
                    # 如果計算統計量出錯，只顯示基本信息
                    print(f"  {i:3d}. {var_name:<15} | 形狀: {str(var_data.shape):<15} | 統計訊息計算錯誤: {str(e)}")

        # 準備變數的編碼設定
        print(f"\n正在準備變數的編碼設定...")
        encoding = prepare_encoding(template_ds)
        print(f"已準備 {len(encoding)} 個變數的編碼設定")

        # 儲存結果
        print(f"\n正在儲存結果至 {args.output}...")
        result_ds.to_netcdf(args.output, encoding=encoding)
        print(f"已成功儲存結果: {args.output}")

        # 關閉資料集
        ds.close()
        template_ds.close()
        result_ds.close()

    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#---------------------------------------
if __name__ == "__main__":
    main()

#===========================================================================================================
print(f"\n======= RUN END: {os.path.basename(sys.argv[0])} =========\n")
