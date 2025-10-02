#!/usr/bin/env python3
# =============================================================================================
# ==== INFORMATION ========
# ========================
# 檔名: extract_wrf_to_nc.py
# 功能: 從WRF輸出檔案中提取指定變數並轉存為NetCDF檔案 (支援網格插值)
# 作者: CYC
# 建立日期: 2025-06-11
# 更新日期: 2025-06-14 - 新增網格插值功能 + WRF全域屬性提取功能(全屬性版本) + 系集/時間平均功能
# 更新日期: 2025-06-15 - 優化提取迴圈in wrfdata_sel
# 更新日期: 2025-06-22 - 增加平均之後保留的屬性
#
# Description:
#   此程式可以從WRF系集輸出中提取指定domain、系集成員、時間點、氣壓層的資料，
#   並可選擇性地將結果插值至規則經緯度網格，最後保存為NetCDF檔案。
#   支援3D大氣變數和2D地面變數，提供可選的資料壓縮功能以節省存儲空間。
#   v1.2版本新增了網格插值功能(-LL參數)，可將WRF曲線網格插值至規則經緯度網格，
#   - 並保留了從原始WRF檔案提取完整全域屬性的功能，
#   - 以及系集平均(--Emean)和時間平均(--Tmean)功能。
#   v1.3版本優化提取迴圈由MVT（Member→Variable→Time）改為 MTV（Member→Time→Variable）
#   - 新增def wrfdata_sel_optimized
#   - 暫時保留def wrfdata_sel
# ============================================================================================

import os
import sys
import argparse
import netCDF4
import xarray as xr
import numpy as np
import wrf
from netCDF4 import Dataset
import pandas as pd
from scipy.interpolate import griddata  # 新增：用於網格插值

def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='從WRF輸出檔案中提取指定變數並轉存為NetCDF檔案（支援多變數與網格插值）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 分解多變數為單獨變數
  python extract_wrf_multi_vars.py -i /path/to/wrf -o extract_wrf_vars_p -V z,uvmet -L 850 -T "2006-06-09_00:00:00" -E 1 --decompose_multi 

  # 提取單層(地面層)資料
  python extract_wrf_multi_vars.py -i /path/to/wrf -o extract_wrf_vars_s -V slp,Q2,uvmet10 -L surface [or sfc or -9999] -T "2006-06-09_00:00:00" -E 1 --decompose_multi

  # 網格插值使用範例
  python extract_wrf_multi_vars.py -i /path/to/wrf -V z,QVAPOR -L 850 -T "2006-06-09_00:00:00" -E 1 -LL 110,130,15,30,0.1 -o interpolated_grid --decompose_multi

  # 系集平均
  python extract_wrf_multi_vars.py -i /path/to/wrf -V z,QVAPOR -L 850 -T "2006-06-09_00:00:00" -E 1,2,3,4,5 --Emean -o ensemble_mean

  # 時間平均  
  python extract_wrf_multi_vars.py -i /path/to/wrf -V z,QVAPOR -L 850 -T "2006-06-09_00:00:00,2006-06-09_03:00:00,2006-06-09_06:00:00" -E 1 --Tmean -o time_mean

  # 系集和時間雙重平均 + 網格插值
  python extract_wrf_multi_vars.py -i /path/to/wrf -V z,QVAPOR -L 850 -T "2006-06-09_00:00:00,2006-06-09_03:00:00" -E 1,2,3 --Emean --Tmean -LL 119,123,22,26,0.2 -o ensemble_time_mean_interp

作者: CYC
Update: 2025-06-22 [v1.4 - Added grid interpolation functionality + comprehensive WRF global attributes extraction + Ensemble/Time averaging + wrfdata_sel_optimized]
        """)

    # 必要參數
    parser.add_argument('-i', '--input_dir', required=True,
                       help='WRF檔案根目錄路徑')

    # 選用參數
    parser.add_argument('-V', '--variable', default='z',
                       help='變數名稱，可用逗號分隔多個變數，例如: z 或 z,tk,ua,va,uvmet (預設: z)')
    parser.add_argument('-L', '--levels', default='850',
                       help='氣壓層（hPa），以逗號分隔，例如: 850 或 850,500,200 (預設: 850)')
    parser.add_argument('-E', '--ensemble', default='1',
                       help='系集成員編號，以逗號分隔，例如: 1 或 1,2,3 (預設: 1)')
    parser.add_argument('-T', '--times', default=None,
                       help='時間點，以逗號分隔，格式: "YYYY-MM-DD_HH:MM:SS" (預設: 自動偵測第一個可用時間)')
    parser.add_argument('-D', '--domain', default='d02',
                       help='Domain名稱 (預設: d02)')
    
    # 變數處理選項
    parser.add_argument('--decompose_multi', action='store_true',
                       help='將多分量變數分解為單獨的變數 (例如: uvmet -> uvmet_u, uvmet_v) (預設: False)')
    
    # 網格插值參數
    parser.add_argument('-LL', '--lonlat_grid', default=None,
                       help='目標經緯度網格範圍，格式: "lon1,lon2,lat1,lat2,resolution" 或 "lon1,lon2,lat1,lat2,dlon,dlat" (預設: 不進行插值)')
    
    # 平均處理選項
    parser.add_argument('--Emean', action='store_true',
                       help='對系集成員維度進行平均 (保留member維度但只有一個值) (預設: False)')
    parser.add_argument('--Tmean', action='store_true',
                       help='對時間維度進行平均 (保留Time維度但只有一個值) (預設: False)')
    
    # 輸出相關參數
    parser.add_argument('-n', '--output_dir', default='./output-w2nc',
                       help='輸出目錄路徑 (預設: ./output-w2nc)')
    parser.add_argument('-o', '--output_file', default='extract_wrf_multi_vars',
                       help='輸出NetCDF檔案名稱 (預設: extract_wrf_multi_vars)')
    parser.add_argument('-c', '--compression', type=int, default=0,
                       help='壓縮等級 0-9，0表示不壓縮 (預設: 0)')

    return parser.parse_args()


def ensure_output_directory(output_dir):
    """確保輸出目錄存在"""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
            return "./"
    return output_dir

def detect_available_times(wrf_dir, domain, ensemble_members):
    """自動偵測可用的時間點"""
    print("Auto-detecting available time points...")
    
    # 使用第一個系集成員來偵測時間點
    first_member = ensemble_members[0]
    member_dir = f"member{first_member:03d}"
    member_path = os.path.join(wrf_dir, member_dir)
    
    if not os.path.exists(member_path):
        raise FileNotFoundError(f"Member directory not found: {member_path}")
    
    # 尋找該domain的檔案
    import glob
    pattern = os.path.join(member_path, f"wrfout_{domain}_*")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No WRF files found for domain {domain} in {member_path}")
    
    # 提取時間字串
    time_points = []
    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        # 提取時間部分: wrfout_d02_2006-06-09_00:00:00
        time_part = filename.split(f'wrfout_{domain}_')[1]
        time_points.append(time_part)
    
    print(f"    Found {len(time_points)} time points")
    print(f"    Time range: {time_points[0]} to {time_points[-1]}")
    
    return time_points

def parse_ll_parameter(ll_string):
    """
    解析經緯度網格參數字串
    
    Parameters:
    -----------
    ll_string : str - 格式: "lon1,lon2,lat1,lat2,resolution" 或 "lon1,lon2,lat1,lat2,dlon,dlat"
    
    Returns:
    --------
    dict - 包含網格參數的字典
    """
    try:
        parts = [float(x.strip()) for x in ll_string.split(',')]
        
        if len(parts) == 5:
            # 格式: lon1,lon2,lat1,lat2,resolution (正方形網格)
            lon1, lon2, lat1, lat2, resolution = parts
            return {
                'lon_min': lon1,
                'lon_max': lon2, 
                'lat_min': lat1,
                'lat_max': lat2,
                'dlon': resolution,
                'dlat': resolution
            }
        elif len(parts) == 6:
            # 格式: lon1,lon2,lat1,lat2,dlon,dlat (矩形網格)
            lon1, lon2, lat1, lat2, dlon, dlat = parts
            return {
                'lon_min': lon1,
                'lon_max': lon2,
                'lat_min': lat1, 
                'lat_max': lat2,
                'dlon': dlon,
                'dlat': dlat
            }
        else:
            raise ValueError(f"LL parameter should have 5 or 6 values, got {len(parts)}")
            
    except Exception as e:
        raise ValueError(f"Invalid LL parameter format: {ll_string}. Expected format: 'lon1,lon2,lat1,lat2,resolution' or 'lon1,lon2,lat1,lat2,dlon,dlat'. Error: {e}")

def decompose_multi_component_variable(var_data, var_name, component_dim):
    """
    將多分量變數分解為單獨的變數，並完整保留原始變數屬性
    
    Parameters:
    -----------
    var_data : xarray.DataArray - 多分量變數數據
    var_name : str - 原始變數名稱
    component_dim : str - 分量維度名稱 ('u_v' 或 'wspd_wdir')
    
    Returns:
    --------
    dict - {分解後變數名: 對應的DataArray}
    """
    decomposed_vars = {}
    
    # 獲取分量坐標
    component_coords = var_data.coords[component_dim].values
    
    print(f"            Decomposing {var_name} into {len(component_coords)} components: {component_coords}")
    
    for i, comp_name in enumerate(component_coords):
        # 提取單一分量並移除分量維度
        comp_data = var_data.isel({component_dim: i}).drop_vars(component_dim)
        
        # 創建新的變數名稱
        decomposed_var_name = f"{var_name}_{comp_name}"
        
        # 設定變數名稱和屬性
        comp_data.name = decomposed_var_name

        # *** 修改：完整複製原始屬性並更新特定資訊 ***
        # 安全地處理屬性，防止 attrs 為 None 的情況
        if hasattr(var_data, 'attrs') and var_data.attrs is not None:
            comp_attrs = dict(var_data.attrs)
        else:
            comp_attrs = {}
            print(f"                Warning: {var_name} has no attributes or attrs is None") 
        
        # 保留所有原始屬性，特別是units、FieldType、MemoryOrder等
        if 'long_name' in comp_attrs:
            comp_attrs['long_name'] = f"{comp_attrs['long_name']} - {comp_name} component"
        elif 'description' in comp_attrs:
            comp_attrs['long_name'] = f"{comp_attrs['description']} - {comp_name} component"
        else:
            comp_attrs['long_name'] = f"{var_name} - {comp_name} component"
        
        # 添加分解相關資訊
        comp_attrs['decomposed_from'] = var_name
        comp_attrs['component'] = comp_name
        
        # 如果原始變數有units，確保保留
        if 'units' in comp_attrs:
            print(f"                -> {decomposed_var_name}: 保留單位 {comp_attrs['units']}")
        
        comp_data.attrs = comp_attrs

        decomposed_vars[decomposed_var_name] = comp_data
        print(f"                -> {decomposed_var_name}")
    
    return decomposed_vars

def get_multi_component_variables():
    """
    返回已知的多分量變數列表及其對應的分量維度
    
    Returns:
    --------
    dict - {變數名: 分量維度名}
    """
    return {
        'uvmet': 'u_v',
        'uvmet10': 'u_v', 
        'wspd_wdir': 'wspd_wdir',
        'wspd_wdir10': 'wspd_wdir',
        'uvmet_wspd_wdir': 'wspd_wdir',
        'uvmet10_wspd_wdir': 'wspd_wdir'
    }


def apply_ensemble_mean(dataset):
    """
    對數據集進行系集平均處理，保持維度一致性，並完整保留原始變數屬性
    
    Parameters:
    -----------
    dataset : xarray.Dataset - 輸入數據集
    
    Returns:
    --------
    xarray.Dataset - 系集平均後的數據集
    """
    print(f"\n    Applying ensemble mean (--Emean)...")
    print(f"        Averaging over 'member' dimension...")
    
    # 對每個變數進行系集平均，保持維度
    averaged_vars = {}
    for var_name in dataset.data_vars:
        var_data = dataset[var_name]
        if 'member' in var_data.dims:
            # 計算平均並保持維度
            var_mean = var_data.mean(dim='member', keepdims=True, skipna=True)
            
            # 更新member座標為表示平均值
            var_mean = var_mean.assign_coords(member=['ensemble_mean'])
            
            # *** 修改：完整保留原始變數屬性 ***
            var_attrs = dict(var_data.attrs)  # 先複製所有原始屬性
            
            # 添加系集平均相關的屬性（保留原始屬性的基礎上添加）
            var_attrs['ensemble_averaged'] = 'True'
            var_attrs['original_ensemble_size'] = len(var_data.member)
            var_attrs['ensemble_members_used'] = str(var_data.member.values.tolist())
            
            # 如果原始有units屬性，確保保留
            if 'units' in var_data.attrs:
                print(f"            保留 {var_name} 的單位: {var_data.attrs['units']}")
            
            # 應用完整的屬性字典
            var_mean.attrs = var_attrs
            
            averaged_vars[var_name] = var_mean
            print(f"            {var_name}: {dict(var_data.sizes)} -> {dict(var_mean.sizes)}")
        else:
            print(f"            Warning: {var_name} does not have 'member' dimension, skipping")
            averaged_vars[var_name] = var_data
    
    # 重建數據集並更新全域屬性
    result_dataset = xr.Dataset(averaged_vars, attrs=dataset.attrs)
    global_attrs = dict(result_dataset.attrs)
    global_attrs['processing_ensemble_mean'] = 'True'
    global_attrs['processing_ensemble_mean_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    result_dataset.attrs = global_attrs
    
    print(f"        Ensemble averaging completed.")
    return result_dataset

def apply_time_mean(dataset):
    """
    對數據集進行時間平均處理，保持維度一致性，並完整保留原始變數屬性
    
    Parameters:
    -----------
    dataset : xarray.Dataset - 輸入數據集
    
    Returns:
    --------
    xarray.Dataset - 時間平均後的數據集
    """
    print(f"\n    Applying time mean (--Tmean)...")
    print(f"        Averaging over 'Time' dimension...")
    
    # 對每個變數進行時間平均，保持維度
    averaged_vars = {}
    for var_name in dataset.data_vars:
        var_data = dataset[var_name]
        if 'Time' in var_data.dims:
            # 計算平均並保持維度
            var_mean = var_data.mean(dim='Time', keepdims=True, skipna=True)
            
            # 創建時間平均的座標標籤
            original_times = var_data.Time.values
            if len(original_times) > 1:
                start_time = pd.to_datetime(original_times[0])
                end_time = pd.to_datetime(original_times[-1])
                time_label = f"mean_{start_time.strftime('%Y%m%d%H')}_{end_time.strftime('%Y%m%d%H')}"
            else:
                time_label = f"mean_{pd.to_datetime(original_times[0]).strftime('%Y%m%d%H')}"
            
            # 更新Time座標為表示平均值
            # 使用中間時間點作為代表
            middle_time = original_times[len(original_times)//2]
            var_mean = var_mean.assign_coords(Time=[middle_time])
            
            # *** 修改：完整保留原始變數屬性 ***
            var_attrs = dict(var_data.attrs)  # 先複製所有原始屬性
            
            # 添加時間平均相關的屬性（保留原始屬性的基礎上添加）
            var_attrs['time_averaged'] = 'True'
            var_attrs['original_time_points'] = len(original_times)
            var_attrs['time_range_start'] = str(original_times[0])
            var_attrs['time_range_end'] = str(original_times[-1])
            var_attrs['time_average_label'] = time_label
            
            # 如果原始有units屬性，確保保留
            if 'units' in var_data.attrs:
                print(f"            保留 {var_name} 的單位: {var_data.attrs['units']}")
            
            # 應用完整的屬性字典
            var_mean.attrs = var_attrs
            
            averaged_vars[var_name] = var_mean
            print(f"            {var_name}: {dict(var_data.sizes)} -> {dict(var_mean.sizes)}")
        else:
            print(f"            Warning: {var_name} does not have 'Time' dimension, skipping")
            averaged_vars[var_name] = var_data
    
    # 重建數據集並更新全域屬性
    result_dataset = xr.Dataset(averaged_vars, attrs=dataset.attrs)
    global_attrs = dict(result_dataset.attrs)
    global_attrs['processing_time_mean'] = 'True'
    global_attrs['processing_time_mean_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    result_dataset.attrs = global_attrs
    
    print(f"        Time averaging completed.")
    return result_dataset

def extract_wrf_global_attributes(base_dir, domain, ensemble_members, time_points):
    """
    從原始WRF檔案提取全域屬性
    
    Parameters:
    -----------
    base_dir : str - WRF檔案根目錄
    domain : str - Domain名稱
    ensemble_members : list - 系集成員列表
    time_points : list - 時間點列表
    
    Returns:
    --------
    dict - 提取的全域屬性字典
    """
    print(f"\n    Extracting WRF global attributes from original files...")
    print(f"        Using comprehensive extraction (all available global attributes)")
    
    extracted_attrs = {}
    
    try:
        # 嘗試從第一個系集成員的第一個時間點讀取屬性
        first_member = ensemble_members[0]
        first_time = time_points[0]
        
        member_dir = f"member{first_member:03d}"
        filename = f"wrfout_{domain}_{first_time}"
        sample_file_path = os.path.join(base_dir, member_dir, filename)
        
        print(f"        Reading global attributes from: {sample_file_path}")
        
        if not os.path.exists(sample_file_path):
            print(f"        Warning: Sample file not found: {sample_file_path}")
            print(f"        Will attempt to find any available WRF file...")
            
            # 嘗試尋找任何可用的WRF檔案
            import glob
            pattern = os.path.join(base_dir, f"member{first_member:03d}", f"wrfout_{domain}_*")
            available_files = glob.glob(pattern)
            
            if available_files:
                sample_file_path = available_files[0]
                print(f"        Using alternative file: {sample_file_path}")
            else:
                print(f"        Error: No WRF files found for global attribute extraction")
                return extracted_attrs
        
        # 打開樣本檔案提取全域屬性
        with Dataset(sample_file_path, 'r') as sample_ncfile:
            print(f"        Extracting all available WRF global attributes...")
            
            # 獲取所有全域屬性名稱
            all_global_attrs = sample_ncfile.ncattrs()
            print(f"        Found {len(all_global_attrs)} total global attributes in WRF file")
            
            successful_attrs = []
            failed_attrs = []
            
            # 遍歷所有全域屬性
            for attr_name in all_global_attrs:
                try:
                    attr_value = getattr(sample_ncfile, attr_name)
                    
                    # 確保屬性值是可序列化的
                    if isinstance(attr_value, (str, int, float, np.number)):
                        extracted_attrs[f"WRF_{attr_name}"] = attr_value
                        successful_attrs.append(attr_name)
                    elif isinstance(attr_value, (list, tuple, np.ndarray)):
                        # 處理陣列類型的屬性
                        if len(attr_value) == 1:
                            extracted_attrs[f"WRF_{attr_name}"] = attr_value[0]
                        else:
                            extracted_attrs[f"WRF_{attr_name}"] = str(attr_value)
                        successful_attrs.append(attr_name)
                    else:
                        # 其他類型轉換為字串
                        try:
                            extracted_attrs[f"WRF_{attr_name}"] = str(attr_value)
                            successful_attrs.append(attr_name)
                        except:
                            # 如果連字串轉換都失敗，跳過該屬性
                            failed_attrs.append(f"{attr_name}: cannot convert to string")
                            continue
                            
                except Exception as attr_err:
                    failed_attrs.append(f"{attr_name}: {str(attr_err)}")
                    continue
            
            print(f"        Successfully extracted {len(successful_attrs)} global attributes:")
            
            # 按屬性名稱模式分類顯示（基於常見的WRF屬性模式）
            categories = {
                'Grid/Dimensions': [attr for attr in successful_attrs if any(x in attr for x in ['GRID', 'DX', 'DY', 'WEST', 'SOUTH', 'BOTTOM', 'PATCH'])],
                'Physics Schemes': [attr for attr in successful_attrs if any(x in attr for x in ['PHYSICS', 'OPT', 'SCHEME'])],
                'Projection/Geography': [attr for attr in successful_attrs if any(x in attr for x in ['LAT', 'LON', 'PROJ', 'MAP', 'CEN', 'POLE', 'STAND'])],
                'Time/Date': [attr for attr in successful_attrs if any(x in attr for x in ['TIME', 'DATE', 'GMT', 'JUL'])],
                'Nesting/Domain': [attr for attr in successful_attrs if any(x in attr for x in ['PARENT', 'RATIO', 'START', 'GRID_ID'])],
                'Land Use/Surface': [attr for attr in successful_attrs if any(x in attr for x in ['MMINLU', 'ISWATER', 'ISLAKE', 'ISICE', 'ISURBAN', 'LAND', 'SURFACE'])],
                'Model Settings': [attr for attr in successful_attrs if any(x in attr for x in ['DT', 'RADT', 'DAMP', 'DIFF', 'KM_', 'BLDT', 'CUDT'])],
                'Simulation Info': [attr for attr in successful_attrs if any(x in attr for x in ['SIMULATION', 'TITLE', 'START_DATE', 'IDEAL', 'REAL'])]
            }
            
            # 顯示分類結果
            categorized_attrs = set()
            for category, attr_list in categories.items():
                if attr_list:
                    print(f"            {category}: {len(attr_list)} attributes")
                    categorized_attrs.update(attr_list)
            
            # 顯示未分類的屬性
            uncategorized = [attr for attr in successful_attrs if attr not in categorized_attrs]
            if uncategorized:
                print(f"            Other attributes: {len(uncategorized)} attributes")
                # 顯示前幾個未分類的屬性作為範例
                if len(uncategorized) <= 10:
                    print(f"                Examples: {uncategorized}")
                else:
                    print(f"                Examples: {uncategorized[:10]} ... (and {len(uncategorized)-10} more)")
            
            if failed_attrs:
                print(f"        Failed to extract {len(failed_attrs)} attributes:")
                for failed in failed_attrs[:5]:  # 只顯示前5個失敗的
                    print(f"            {failed}")
                if len(failed_attrs) > 5:
                    print(f"            ... and {len(failed_attrs) - 5} more")
        
        # 添加額外的處理資訊
        extracted_attrs['WRF_extraction_source'] = os.path.basename(sample_file_path)
        extracted_attrs['WRF_extraction_time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        extracted_attrs['WRF_domain_extracted'] = domain
        extracted_attrs['WRF_total_extracted_attrs'] = len(successful_attrs)
        extracted_attrs['WRF_total_available_attrs'] = len(all_global_attrs)
        
        print(f"        Global attribute extraction completed:")
        print(f"            Successfully extracted: {len(successful_attrs)} attributes")
        print(f"            Total available in file: {len(all_global_attrs)} attributes") 
        print(f"            Extraction rate: {100*len(successful_attrs)/len(all_global_attrs):.1f}%")
        print(f"            Final output attributes: {len(extracted_attrs)} (includes metadata)")
        
    except Exception as e:
        print(f"        Error during global attribute extraction: {str(e)}")
        print(f"        Continuing without WRF global attributes...")
        return {}
    
    return extracted_attrs

def save_to_netcdf(data, output_path, compression_level=0, 
                   wrf_base_dir=None, wrf_domain=None, wrf_ensemble_members=None, wrf_time_points=None):
    """
    將xarray數據保存為NetCDF檔案
    
    Parameters:
    -----------
    data : xarray.DataArray - 要保存的數據
    output_path : str - 輸出檔案路徑
    compression_level : int - 壓縮等級 (0-9)
    wrf_base_dir : str - WRF檔案根目錄 (用於提取全域屬性)
    wrf_domain : str - WRF domain名稱
    wrf_ensemble_members : list - 系集成員列表
    wrf_time_points : list - 時間點列表
    """
    print(f"\nSaving data to NetCDF file...")
    print(f"    Output file: {output_path}")
    print(f"    Compression level: {compression_level}")
    
    # 轉換為Dataset以便保存
    if isinstance(data, xr.DataArray):
        # 確保DataArray有名稱
        if data.name is None:
            data.name = 'extracted_variable'
        dataset = data.to_dataset()
    else:
        dataset = data
    
    # -----------------
    # 清理編碼相關的屬性衝突
    # -----------------
    print(f"    Cleaning encoding conflicts...")
    
    # 需要從變數屬性中移除的編碼相關鍵值
    encoding_attrs_to_remove = ['_FillValue', 'missing_value', 'scale_factor', 'add_offset', 
                               'zlib', 'complevel', 'shuffle', 'chunksizes', 'fletcher32']
    
    # 清理數據變數的編碼屬性
    for var_name in dataset.data_vars:
        cleaned_attrs = {}
        removed_attrs = []
        
        for attr_name, attr_value in dataset[var_name].attrs.items():
            if attr_name in encoding_attrs_to_remove:
                removed_attrs.append(attr_name)
            else:
                cleaned_attrs[attr_name] = attr_value
        
        if removed_attrs:
            print(f"        Removed encoding attributes from {var_name}: {removed_attrs}")
        
        dataset[var_name].attrs = cleaned_attrs
    
    # 清理座標變數的編碼屬性
    for coord_name in dataset.coords:
        if hasattr(dataset.coords[coord_name], 'attrs'):
            cleaned_attrs = {}
            removed_attrs = []
            
            for attr_name, attr_value in dataset.coords[coord_name].attrs.items():
                if attr_name in encoding_attrs_to_remove:
                    removed_attrs.append(attr_name)
                else:
                    cleaned_attrs[attr_name] = attr_value
            
            if removed_attrs:
                print(f"        Removed encoding attributes from coordinate {coord_name}: {removed_attrs}")
            
            dataset.coords[coord_name].attrs = cleaned_attrs
    
    # -----------------
    # 設定編碼選項
    # -----------------
    encoding = {}
    
    if compression_level > 0:
        # 為數據變數設定壓縮
        for var_name in dataset.data_vars:
            encoding[var_name] = {
                'zlib': True,
                'complevel': compression_level,
                'shuffle': True,
                '_FillValue': np.nan
            }
        
        # 為座標變數設定壓縮（但需要特殊處理某些座標）
        for coord_name in dataset.coords:
            coord_data = dataset.coords[coord_name]
            
            # 特殊處理problematic的座標
            if coord_name == 'interp_level':
                # 檢查interp_level的數據類型和內容
                if coord_data.dtype.kind in ['U', 'S', 'O']:  # 字符串類型
                    print(f"        Skipping compression for string coordinate: {coord_name}")
                    continue
                elif len(coord_data) == 1:  # 只有一個值的座標
                    print(f"        Skipping compression for single-value coordinate: {coord_name}")
                    continue
            elif coord_name == 'Time':
                # Time座標特殊處理
                encoding[coord_name] = {
                    'zlib': True,
                    'complevel': min(0, compression_level),
                    'shuffle': False  # Time座標不使用shuffle
                }
                continue
            elif coord_name in ['XLONG', 'XLAT']:
                # 經緯度座標特殊處理
                encoding[coord_name] = {
                    'zlib': True,
                    'complevel': min(0, compression_level),
                    'shuffle': True
                }
                continue
            
            # 其他座標的一般壓縮設定
            if coord_name not in ['XTIME', 'lon', 'lat']:  # 跳過可能有問題的座標
                encoding[coord_name] = {
                    'zlib': True,
                    'complevel': min(0, compression_level),
                    'shuffle': True
                }
    
    print(f"    Applied compression encoding to: {list(encoding.keys())}")
    
    try:
        # -----------------
        # 提取並保存投影資訊
        # -----------------
        print(f"    Extracting and preserving projection information...")
        
        projection_info = {}  # 存儲提取的投影資訊
        
        # 檢查所有變數和座標的投影屬性
        all_items = list(dataset.data_vars.items()) + list(dataset.coords.items())
        
        for item_name, item_data in all_items:
            if hasattr(item_data, 'attrs'):
                for attr_name, attr_value in item_data.attrs.items():
                    if 'projection' in attr_name.lower() or 'crs' in attr_name.lower():
                        # 檢查是否為投影對象
                        if hasattr(attr_value, '__dict__') or 'Lambert' in str(type(attr_value)):
                            print(f"        Found projection object in {item_name}.{attr_name}")
                            
                            # 提取LambertConformal投影參數
                            if 'LambertConformal' in str(type(attr_value)):
                                try:
                                    proj_params = {
                                        'projection_type': 'LambertConformal',
                                        'standard_longitude': getattr(attr_value, 'stand_lon', 'unknown'),
                                        'center_latitude': getattr(attr_value, 'moad_cen_lat', 'unknown'),
                                        'true_latitude_1': getattr(attr_value, 'truelat1', 'unknown'),
                                        'true_latitude_2': getattr(attr_value, 'truelat2', 'unknown'),
                                        'pole_latitude': getattr(attr_value, 'pole_lat', 'unknown'),
                                        'pole_longitude': getattr(attr_value, 'pole_lon', 'unknown')
                                    }
                                    
                                    # 將投影參數加入全域資訊
                                    for param_name, param_value in proj_params.items():
                                        projection_info[param_name] = param_value
                                    
                                    """
                                    print(f"            Extracted Lambert Conformal parameters:")
                                    for param_name, param_value in proj_params.items():
                                        print(f"                {param_name}: {param_value}")
                                    """
                                        
                                except Exception as ex:
                                    print(f"            Warning: Could not extract projection parameters: {ex}")
                                    # 至少保存字符串表示
                                    projection_info['projection_string'] = str(attr_value)
                            else:
                                # 對於其他類型的投影對象，保存字符串表示
                                projection_info['projection_string'] = str(attr_value)
        
        # -----------------
        # *** 新增：提取WRF全域屬性 ***
        # -----------------
        wrf_global_attrs = {}
        if (wrf_base_dir and wrf_domain and wrf_ensemble_members and wrf_time_points):
            print(f"    Extracting WRF global attributes from original files...")
            wrf_global_attrs = extract_wrf_global_attributes(
                wrf_base_dir, wrf_domain, wrf_ensemble_members, wrf_time_points
            )
            if wrf_global_attrs:
                print(f"        Successfully extracted {len(wrf_global_attrs)} WRF global attributes")
            else:
                print(f"        No WRF global attributes were extracted")
        else:
            print(f"    Skipping WRF global attribute extraction (missing parameters)")
        
        # -----------------
        # 清理不相容的屬性
        # -----------------
        print(f"    Cleaning incompatible attributes for NetCDF serialization...")
        
        # 處理數據變數的屬性
        for var_name in dataset.data_vars:
            var_attrs = dict(dataset[var_name].attrs)  # 複製屬性字典
            cleaned_attrs = {}
            
            for attr_name, attr_value in var_attrs.items():
                # 檢查屬性值是否可以序列化到NetCDF
                if isinstance(attr_value, (str, int, float, np.number, list, tuple, np.ndarray)):
                    # 跳過編碼相關的屬性，這些已經在前面處理過了
                    if attr_name not in ['_FillValue', 'missing_value', 'scale_factor', 'add_offset']:
                        cleaned_attrs[attr_name] = attr_value
                elif hasattr(attr_value, '__str__'):
                    # 對於投影對象等複雜對象，已經提取過資訊，現在移除
                    if 'projection' in attr_name.lower() or 'crs' in attr_name.lower():
                        print(f"        Removing projection object '{attr_name}' (info preserved in global attributes)")
                    else:
                        print(f"        Removing incompatible attribute '{attr_name}' (type: {type(attr_value)})")
                else:
                    print(f"        Removing incompatible attribute '{attr_name}' (type: {type(attr_value)})")
            
            # 應用清理後的屬性
            dataset[var_name].attrs = cleaned_attrs
        
        # 處理座標變數的屬性  
        for coord_name in dataset.coords:
            if hasattr(dataset.coords[coord_name], 'attrs'):
                coord_attrs = dict(dataset.coords[coord_name].attrs)
                cleaned_attrs = {}
                
                for attr_name, attr_value in coord_attrs.items():
                    if isinstance(attr_value, (str, int, float, np.number, list, tuple, np.ndarray)):
                        # 跳過編碼相關的屬性
                        if attr_name not in ['_FillValue', 'missing_value', 'scale_factor', 'add_offset']:
                            cleaned_attrs[attr_name] = attr_value
                    elif hasattr(attr_value, '__str__'):
                        if 'projection' in attr_name.lower() or 'crs' in attr_name.lower():
                            print(f"        Removing coordinate projection object '{attr_name}' (info preserved in global attributes)")
                        else:
                            print(f"        Removing incompatible coordinate attribute '{attr_name}' (type: {type(attr_value)})")
                    else:
                        print(f"        Removing incompatible coordinate attribute '{attr_name}' (type: {type(attr_value)})")
                
                dataset.coords[coord_name].attrs = cleaned_attrs
        
        # 處理全域屬性
        dataset_attrs = dict(dataset.attrs)
        cleaned_global_attrs = {}
        
        for attr_name, attr_value in dataset_attrs.items():
            if isinstance(attr_value, (str, int, float, np.number, list, tuple, np.ndarray)):
                cleaned_global_attrs[attr_name] = attr_value
            elif hasattr(attr_value, '__str__'):
                if 'projection' in attr_name.lower() or 'crs' in attr_name.lower():
                    print(f"        Removing global projection object '{attr_name}' (info preserved in specific attributes)")
                else:
                    print(f"        Removing incompatible global attribute '{attr_name}' (type: {type(attr_value)})")
            else:
                print(f"        Removing incompatible global attribute '{attr_name}' (type: {type(attr_value)})")
        
        # -----------------
        # 添加投影資訊到全域屬性
        # -----------------
        if projection_info:
            print(f"    Adding projection information to global attributes...")
            
            # 將投影參數加入全域屬性
            for param_name, param_value in projection_info.items():
                attr_name = f"projection_{param_name}"
                cleaned_global_attrs[attr_name] = param_value
                print(f"        Added: {attr_name} = {param_value}")
            
            # 創建投影的文字描述
            if 'projection_type' in projection_info and projection_info['projection_type'] == 'LambertConformal':
                proj_description = (
                    f"Lambert Conformal Conic projection: "
                    f"Standard longitude={projection_info.get('standard_longitude', 'N/A')}°, "
                    f"Center latitude={projection_info.get('center_latitude', 'N/A')}°, "
                    f"True latitudes={projection_info.get('true_latitude_1', 'N/A')}° and {projection_info.get('true_latitude_2', 'N/A')}°"
                )
                cleaned_global_attrs['projection_description'] = proj_description
                print(f"        Added projection description: {proj_description}")
        else:
            print(f"        No projection information found to preserve")
        
        # -----------------
        # *** 新增：添加WRF全域屬性 ***
        # -----------------
        if wrf_global_attrs:
            print(f"    Adding WRF global attributes to output file...")
            
            # 將WRF全域屬性加入到輸出檔案的全域屬性中
            for attr_name, attr_value in wrf_global_attrs.items():
                cleaned_global_attrs[attr_name] = attr_value
            
            print(f"        Successfully added {len(wrf_global_attrs)} WRF global attributes")
            
            # 顯示一些重要的WRF屬性
            important_wrf_attrs = [
                'WRF_DX', 'WRF_DY', 'WRF_MP_PHYSICS', 'WRF_CU_PHYSICS', 
                'WRF_BL_PBL_PHYSICS', 'WRF_GRID_ID', 'WRF_MAP_PROJ_CHAR'
            ]
            
            found_important = [attr for attr in important_wrf_attrs if attr in wrf_global_attrs]
            if found_important:
                print(f"        Key WRF attributes preserved: {found_important}")
        
        # 添加其他自定義全域屬性
        cleaned_global_attrs.update({
            'title': 'Extracted WRF data with comprehensive metadata and grid interpolation',
            'source': 'extract_wrf_to_nc.py',
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': f'Extracted variables with full WRF global attributes preserved and optional grid interpolation',
            'conventions': 'CF-1.6'
        })
        
        dataset.attrs = cleaned_global_attrs
        
        print(f"    Attribute processing completed")
        print(f"    Total global attributes: {len(cleaned_global_attrs)}")
        
        # 顯示屬性統計
        wrf_attr_count = len([k for k in cleaned_global_attrs.keys() if k.startswith('WRF_')])
        proj_attr_count = len([k for k in cleaned_global_attrs.keys() if k.startswith('projection_')])
        interp_attr_count = len([k for k in cleaned_global_attrs.keys() if k.startswith('interpolation_')])
        other_attr_count = len(cleaned_global_attrs) - wrf_attr_count - proj_attr_count - interp_attr_count
        
        #print(f"        WRF attributes: {wrf_attr_count}")
        #print(f"        Projection attributes: {proj_attr_count}")
        #print(f"        Interpolation attributes: {interp_attr_count}")
        #print(f"        Other attributes: {other_attr_count}")
        
        # -----------------
        # 保存檔案
        # -----------------
        print(f"    Attempting to save NetCDF file...")
        
        # 在保存前再次檢查encoding設定
        valid_encoding = {}
        for var_name, enc_dict in encoding.items():
            if var_name in dataset.data_vars or var_name in dataset.coords:
                valid_encoding[var_name] = enc_dict
            else:
                print(f"        Warning: Skipping encoding for non-existent variable: {var_name}")
        
        try:
            # 嘗試保存
            dataset.to_netcdf(output_path, encoding=valid_encoding)
            print(f"        File saved successfully with encoding")
            
        except Exception as save_error:
            print(f"        Warning: Save with encoding failed: {save_error}")
            print(f"        Attempting to save without compression...")
            
            # 如果帶編碼保存失敗，嘗試不使用壓縮保存
            try:
                dataset.to_netcdf(output_path)
                print(f"        File saved successfully without compression")
                compression_level = 0  # 更新壓縮等級標記
                
            except Exception as fallback_error:
                print(f"        Error: Could not save file even without compression: {fallback_error}")
                raise fallback_error
        
        # 檢查檔案大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"    File processing completed:")
        print(f"        File size: {file_size:.2f} MB")
        
        if compression_level > 0:
            print(f"        Compression applied: level {compression_level}")
        else:
            print(f"        No compression applied")
            
        # 驗證保存的檔案
        print(f"    Verifying saved file...")
        try:
            test_ds = xr.open_dataset(output_path)
            print(f"        File verification successful")
            print(f"        Saved dimensions: {list(test_ds.sizes.keys())}")
            print(f"        Saved variables: {list(test_ds.data_vars.keys())}")

            # *** 詳細的維度和形狀資訊 ***
            print(f"        Dataset shape information:")
            
            # 顯示每個維度的大小
            print(f"            Dimensions: {dict(test_ds.sizes)}")
            
            # 顯示第一個變數的完整形狀作為範例
            if test_ds.data_vars:
                first_var_name = list(test_ds.data_vars.keys())[0]
                first_var = test_ds[first_var_name]
                print(f"            Shape (example from '{first_var_name}'): {first_var.shape}")
                print(f"            Dimensions: {first_var.dims}")
            
            # *** 氣壓層資訊 ***
            if 'interp_level' in test_ds.coords:
                pressure_levels = test_ds.coords['interp_level'].values
                if len(pressure_levels) == 1 and pressure_levels[0] == 'surface':
                    print(f"            Pressure levels: Surface level only")
                else:
                    try:
                        # 嘗試轉換為數值並顯示
                        numeric_levels = [float(p) for p in pressure_levels if p != 'surface']
                        if numeric_levels:
                            print(f"            Pressure levels: {numeric_levels} hPa")
                        else:
                            print(f"            Pressure levels: {pressure_levels}")
                    except (ValueError, TypeError):
                        print(f"            Pressure levels: {pressure_levels}")
            
            # *** 時間範圍資訊 ***
            if 'Time' in test_ds.coords:
                time_values = test_ds.coords['Time'].values
                if len(time_values) == 1:
                    time_str = pd.to_datetime(time_values[0]).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"            Time: {time_str}")
                elif len(time_values) > 1:
                    start_time = pd.to_datetime(time_values[0]).strftime('%Y-%m-%d %H:%M:%S')
                    end_time = pd.to_datetime(time_values[-1]).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"            Time range: {start_time} to {end_time}")
                    print(f"            Number of time steps: {len(time_values)}")
            
            # *** 系集成員資訊 ***
            if 'member' in test_ds.coords:
                member_values = test_ds.coords['member'].values
                if len(member_values) == 1:
                    if member_values[0] == 'ensemble_mean':
                        print(f"            Members: Ensemble mean")
                    else:
                        print(f"            Members: {member_values[0]}")
                else:
                    print(f"            Members: {list(member_values)} ({len(member_values)} total)")

            # *** 空間網格資訊 ***
            spatial_dims = []
            spatial_info = []
            
            # 檢查不同可能的空間維度名稱
            if 'south_north' in test_ds.sizes and 'west_east' in test_ds.sizes:
                spatial_dims = ['south_north', 'west_east']
                spatial_info.append(f"WRF grid: {test_ds.sizes['south_north']} × {test_ds.sizes['west_east']}")
            elif 'lat' in test_ds.sizes and 'lon' in test_ds.sizes:
                spatial_dims = ['lat', 'lon']
                spatial_info.append(f"Regular grid: {test_ds.sizes['lat']} × {test_ds.sizes['lon']}")
                
                # 如果是規則網格，顯示經緯度範圍
                if 'lat' in test_ds.coords and 'lon' in test_ds.coords:
                    lat_range = test_ds.coords['lat'].values
                    lon_range = test_ds.coords['lon'].values
                    spatial_info.append(f"Latitude: {lat_range.min():.3f}° to {lat_range.max():.3f}°")
                    spatial_info.append(f"Longitude: {lon_range.min():.3f}° to {lon_range.max():.3f}°")
            
            if spatial_info:
                print(f"            Spatial grid:")
                for info in spatial_info:
                    print(f"                {info}")
            
            # 顯示保存的投影資訊
            projection_attrs = {k: v for k, v in test_ds.attrs.items() if k.startswith('projection_')}
            if projection_attrs:
                print(f"        Preserved projection attributes:")
                for attr_name, attr_value in projection_attrs.items():
                    print(f"            {attr_name}: {attr_value}")
            
            # 顯示保存的插值資訊
            interp_attrs = {k: v for k, v in test_ds.attrs.items() if k.startswith('interpolation_')}
            if interp_attrs:
                print(f"        Preserved interpolation attributes:")
                for attr_name, attr_value in interp_attrs.items():
                    print(f"            {attr_name}: {attr_value}")
            
            # 顯示保存的WRF屬性統計
            wrf_attrs = {k: v for k, v in test_ds.attrs.items() if k.startswith('WRF_')}
            if wrf_attrs:
                print(f"        Preserved WRF attributes: {len(wrf_attrs)} total")
                """
                # 顯示一些關鍵的WRF屬性
                key_wrf_attrs = ['WRF_DX', 'WRF_DY', 'WRF_MP_PHYSICS', 'WRF_CU_PHYSICS']
                for key_attr in key_wrf_attrs:
                    if key_attr in wrf_attrs:
                        print(f"            {key_attr}: {wrf_attrs[key_attr]}")
                """
            
            test_ds.close()
            
        except Exception as verify_error:
            print(f"        Warning: File verification failed: {verify_error}")
            print(f"        But file was saved, you may still be able to use it")
            
    except Exception as e:
        print(f"    Error during save process: {str(e)}")
        
        # 提供更詳細的錯誤診斷
        if '_FillValue' in str(e):
            print(f"    This appears to be a _FillValue encoding conflict.")
            print(f"    The variable attributes contained encoding information that conflicts with xarray's encoding.")
            
        elif 'filter' in str(e).lower():
            print(f"    This appears to be a compression filter issue.")
            print(f"    Some NetCDF libraries may not support certain compression settings for string coordinates.")
            
        elif 'projection' in str(e).lower() or 'lambertconformal' in str(e).lower():
            print(f"    This appears to be a projection attribute issue.")
            print(f"    WRF projection objects cannot be directly serialized to NetCDF.")
            
        # 嘗試提供修復建議
        print(f"    Suggested solutions:")
        print(f"        1. Try running with -c 0 to disable compression")
        print(f"        2. Try a different variable")
        print(f"        3. Check that all WRF files are properly formatted")
        
        raise

    print(f"    Final output file: {output_path}")

def rename_lonlat_variables(dataset):
    """
    重新命名數據集中的lon/lat變數名稱

    Parameters:
    -----------
    dataset : xarray.Dataset - 輸入數據集

    Returns:
    --------
    xarray.Dataset - 重新命名後的數據集
    """
    print(f"\nChecking for lon/lat variable names to rename...")

    # 檢查需要重新命名的變數
    rename_dict = {}

    if 'lon' in dataset.data_vars:
        rename_dict['lon'] = 'xlon'
        print(f"    Found variable 'lon' -> will rename to 'xlon'")

    if 'lat' in dataset.data_vars:
        rename_dict['lat'] = 'xlat'
        print(f"    Found variable 'lat' -> will rename to 'xlat'")

    # 執行重新命名
    if rename_dict:
        dataset = dataset.rename(rename_dict)
        print(f"    Successfully renamed {len(rename_dict)} variables: {list(rename_dict.keys())} -> {list(rename_dict.values())}")

        # 更新重新命名變數的屬性
        for old_name, new_name in rename_dict.items():
            if hasattr(dataset[new_name], 'attrs'):
                var_attrs = dict(dataset[new_name].attrs)
                var_attrs['original_variable_name'] = old_name
                var_attrs['renamed_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                dataset[new_name].attrs = var_attrs
                print(f"            Added rename metadata to '{new_name}'")
    else:
        print(f"    No lon/lat variables found to rename")

    return dataset

def batch_griddata_interpolation(data, src_lon, src_lat, target_lon, target_lat, original_dataset=None):
    """
    循環插值五維數據的每個X-Y平面，使用scipy.griddata並保留原始NaN區域
    
    Parameters:
    -----------
    data : numpy.ndarray - 形狀為 (member, time, level, south_north, west_east)
    src_lon : numpy.ndarray - 源網格經度，形狀為 (south_north, west_east)
    src_lat : numpy.ndarray - 源網格緯度，形狀為 (south_north, west_east)
    target_lon : numpy.ndarray - 目標網格經度，形狀為 (new_south_north, new_west_east)
    target_lat : numpy.ndarray - 目標網格緯度，形狀為 (new_south_north, new_west_east)
    original_dataset : xarray.Dataset - 原始數據集，用於提取_FillValue資訊
    
    Returns:
    --------
    numpy.ndarray - 插值後的數據，形狀為 (member, time, level, new_south_north, new_west_east)
    """
    print(f"        Performing loop-based griddata interpolation with NaN mask preservation...")
    
    m, t, l, sn, we = data.shape
    new_sn, new_we = target_lon.shape
    
    print(f"            Input data shape: {data.shape}")
    print(f"            Target grid shape: {target_lon.shape}")
    print(f"            Total X-Y planes to interpolate: {m * t * l}")
    
    # -----------------
    # *** 新增：讀取原始數據的_FillValue資訊 ***
    # -----------------
    print(f"            Analyzing _FillValue and missing value markers...")
    
    fill_values = []
    if original_dataset is not None:
        # 從原始xarray數據集中提取_FillValue
        for var_name in original_dataset.data_vars:
            var_data = original_dataset[var_name]
            if hasattr(var_data, 'attrs'):
                if '_FillValue' in var_data.attrs:
                    fill_val = var_data.attrs['_FillValue']
                    fill_values.append(fill_val)
                    print(f"                Found _FillValue for {var_name}: {fill_val} (type: {type(fill_val)})")
                if 'missing_value' in var_data.attrs:
                    miss_val = var_data.attrs['missing_value']
                    fill_values.append(miss_val)
                    print(f"                Found missing_value for {var_name}: {miss_val} (type: {type(miss_val)})")
    
    # 添加常見的缺失值標記
    common_fill_values = [-9999, -999, 9999, 1e20, -1e20, 999999, -999999]
    fill_values.extend(common_fill_values)
    
    # 去除重複值並轉換為numpy array以便比較
    unique_fill_values = []
    for fv in fill_values:
        if np.isnan(fv) if isinstance(fv, (float, np.floating)) else False:
            continue  # 跳過NaN，因為會用isnan()單獨處理
        if fv not in unique_fill_values:
            unique_fill_values.append(fv)
    
    print(f"                Total fill values to check: {len(unique_fill_values)} + NaN")
    if unique_fill_values:
        print(f"                Fill values: {unique_fill_values}")
    
    # -----------------
    # *** 詳細診斷原始數據的缺失值情況 ***
    # -----------------
    print(f"            Diagnosing missing values in original data...")
    
    # 檢查不同類型的缺失值
    data_nan_count = np.isnan(data).sum()
    data_inf_count = np.isinf(data).sum()
    data_finite_count = np.isfinite(data).sum()
    data_total = data.size
    
    print(f"                NaN values: {data_nan_count}")
    print(f"                Inf values: {data_inf_count}")
    print(f"                Finite values: {data_finite_count}")
    print(f"                Total values: {data_total}")
    print(f"                Finite ratio: {100*data_finite_count/data_total:.2f}%")
    
    # 檢查是否有特殊的缺失值標記
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    print(f"                Value range: {data_min:.6f} to {data_max:.6f}")
    
    # 檢查_FillValue和其他缺失值標記
    total_fill_value_count = 0
    for fill_val in unique_fill_values:
        try:
            if np.isfinite(fill_val):
                count = np.sum(np.abs(data - fill_val) < 1e-6)
                if count > 0:
                    print(f"                Found {count} values equal to {fill_val} (_FillValue/missing_value)")
                    total_fill_value_count += count
        except:
            continue
    
    print(f"                Total _FillValue/missing_value markers: {total_fill_value_count}")
    
    # 建立源座標點 (經度, 緯度)
    src_points = np.column_stack((src_lon.flatten(), src_lat.flatten()))
    print(f"            Source points shape: {src_points.shape}")
    
    # 檢查是否有無效的座標
    valid_coord_mask = np.isfinite(src_points).all(axis=1)
    print(f"            Valid coordinate points: {valid_coord_mask.sum()}/{len(valid_coord_mask)}")
    
    if valid_coord_mask.sum() < 4:
        raise ValueError(f"Not enough valid coordinate points: {valid_coord_mask.sum()}. Need at least 4 points.")
    
    # -----------------
    # *** 修改：正確處理_FillValue的有效區域mask創建 ***
    # -----------------
    print(f"            Creating _FillValue-aware validity mask...")
    
    # 檢查所有場中哪些空間點至少有一個有效值
    data_2d_reshaped = data.reshape(m * t * l, sn, we)  # (場數, south_north, west_east)
    
    # 1. 基本的有限性檢查
    finite_mask = np.isfinite(data_2d_reshaped)
    
    # 2. *** 新增：_FillValue檢查 ***
    fill_value_mask = np.ones_like(data_2d_reshaped, dtype=bool)
    for fill_val in unique_fill_values:
        try:
            if np.isfinite(fill_val):
                is_fill_value = np.abs(data_2d_reshaped - fill_val) < 1e-6
                fill_value_mask &= ~is_fill_value
                fill_count = np.sum(is_fill_value)
                if fill_count > 0:
                    print(f"                Marked {fill_count} values equal to {fill_val} as invalid")
        except:
            continue
    
    # 3. 統計異常值檢查（保留，但降低影響）
    reasonable_range_mask = np.ones_like(data_2d_reshaped, dtype=bool)
    data_std = np.nanstd(data_2d_reshaped)
    data_mean = np.nanmean(data_2d_reshaped)
    
    if data_std > 0:
        outlier_threshold = 20 * data_std  # 放寬到20個標準差（降低誤判）
        lower_bound = data_mean - outlier_threshold
        upper_bound = data_mean + outlier_threshold
        
        statistical_outliers = ((data_2d_reshaped < lower_bound) | 
                               (data_2d_reshaped > upper_bound))
        # 只有在沒有明確_FillValue的情況下才使用統計方法
        if total_fill_value_count == 0:
            reasonable_range_mask = ~statistical_outliers
            outlier_count = np.sum(statistical_outliers)
            if outlier_count > 0:
                print(f"                Found {outlier_count} statistical outliers (used as backup method)")
        else:
            print(f"                Skipping statistical outlier detection (explicit _FillValue found)")
    
    # 綜合所有條件創建有效性mask
    global_valid_mask_3d = finite_mask & fill_value_mask & reasonable_range_mask
    global_valid_mask_2d = global_valid_mask_3d.any(axis=0)  # (south_north, west_east)
    
    valid_spatial_points = global_valid_mask_2d.sum()
    total_spatial_points = global_valid_mask_2d.size
    
    print(f"            _FillValue-aware validity analysis:")
    print(f"                Valid spatial points: {valid_spatial_points}/{total_spatial_points}")
    print(f"                Valid spatial ratio: {100*valid_spatial_points/total_spatial_points:.1f}%")
    
    # 如果大部分點都有效，檢查是否可能遺漏了什麼
    if valid_spatial_points / total_spatial_points > 0.95:
        print(f"                Note: High valid ratio detected.")
        if total_fill_value_count == 0:
            print(f"                This might indicate all data is genuinely valid,")
            print(f"                or there might be undetected missing value encoding.")
        else:
            print(f"                _FillValue markers were found and processed.")
    
    # 將有效mask轉為浮點數用於插值
    mask_for_interp = global_valid_mask_2d.astype(float).flatten()
    
    # 插值有效區域mask到目標網格
    print(f"            Interpolating validity mask...")
    
    mask_valid_points = src_points[valid_coord_mask]
    mask_valid_data = mask_for_interp[valid_coord_mask]
    
    try:
        # 遮照內插
        interpolated_mask = griddata(
            mask_valid_points, 
            mask_valid_data, 
            (target_lon, target_lat), 
            method='nearest',
            fill_value=0.0
        )
        print(f"            Mask interpolation completed (method='linear')")
        
    except Exception as e:
        print(f"            Linear mask interpolation failed: {e}")
        print(f"            Using nearest neighbor for mask interpolation...")
        
        interpolated_mask = griddata(
            mask_valid_points, 
            mask_valid_data, 
            (target_lon, target_lat), 
            method='nearest'
        )
        interpolated_mask[interpolated_mask < 0.5] = 0.0
    
    # *** 動態調整mask閾值 ***
    mask_threshold = 0.7 if total_fill_value_count > 0 else 0.5  # 有_FillValue時更嚴格
    final_mask = interpolated_mask >= mask_threshold
    
    print(f"            Mask statistics:")
    print(f"                Interpolated mask range: {interpolated_mask.min():.3f} to {interpolated_mask.max():.3f}")
    print(f"                Mask threshold: {mask_threshold}")
    print(f"                Valid target points (mask >= {mask_threshold}): {final_mask.sum()}/{final_mask.size}")
    print(f"                Valid ratio after masking: {100*final_mask.sum()/final_mask.size:.1f}%")
    
    # -----------------
    # 初始化結果數組
    # -----------------
    result = np.full((m, t, l, new_sn, new_we), np.nan, dtype=data.dtype)
    
    # -----------------
    # 循環處理每個X-Y平面
    # -----------------
    print(f"            Starting loop interpolation...")
    
    plane_count = 0
    successful_planes = 0
    
    for member_idx in range(m):
        for time_idx in range(t):
            for level_idx in range(l):
                plane_count += 1
                
                # 提取當前X-Y平面
                current_plane = data[member_idx, time_idx, level_idx, :, :]  # (south_north, west_east)
                
                # 展平當前平面
                current_data = current_plane.flatten()
                
                # *** 修改：針對當前平面應用相同的_FillValue檢查 ***
                plane_finite_mask = np.isfinite(current_data)
                
                plane_fill_value_mask = np.ones_like(current_data, dtype=bool)
                for fill_val in unique_fill_values:
                    try:
                        if np.isfinite(fill_val):
                            plane_fill_value_mask &= ~(np.abs(current_data - fill_val) < 1e-6)
                    except:
                        continue
                
                plane_valid_mask = plane_finite_mask & plane_fill_value_mask
                combined_valid_mask = valid_coord_mask & plane_valid_mask
                
                valid_points_current = src_points[combined_valid_mask]
                valid_data_current = current_data[combined_valid_mask]
                
                if len(valid_points_current) < 4:
                    if plane_count <= 5:  # 只顯示前幾個的詳細信息
                        print(f"                Plane {plane_count}/{m*t*l} (m{member_idx+1},t{time_idx+1},l{level_idx+1}): "
                              f"Not enough valid points ({len(valid_points_current)}), skipping")
                    continue
                
                # 執行插值
                try:
                    interp_plane = griddata(
                        valid_points_current, 
                        valid_data_current, 
                        (target_lon, target_lat), 
                        method='linear',
                        fill_value=np.nan
                    )
                    
                    # 應用mask
                    interp_plane[~final_mask] = np.nan
                    
                    # 存儲結果
                    result[member_idx, time_idx, level_idx, :, :] = interp_plane
                    successful_planes += 1
                    
                    if plane_count % 5 == 0 or plane_count <= 5:  # 顯示前幾個和每5個
                        valid_count = np.isfinite(interp_plane).sum()
                        total_count = interp_plane.size
                        value_range = f"{np.nanmin(interp_plane):.3f} to {np.nanmax(interp_plane):.3f}" if valid_count > 0 else "no valid data"
                        print(f"                Plane {plane_count}/{m*t*l} (m{member_idx+1},t{time_idx+1},l{level_idx+1}): "
                              f"Success, {valid_count}/{total_count} valid points, range: {value_range}")
                
                except Exception as e:
                    if plane_count <= 5:
                        print(f"                Plane {plane_count}/{m*t*l} (m{member_idx+1},t{time_idx+1},l{level_idx+1}): "
                              f"Linear interpolation failed: {e}")
                    
                    # 嘗試最近鄰插值
                    try:
                        interp_plane = griddata(
                            valid_points_current, 
                            valid_data_current, 
                            (target_lon, target_lat), 
                            method='nearest'
                        )
                        
                        # 應用mask
                        interp_plane[~final_mask] = np.nan
                        
                        # 存儲結果
                        result[member_idx, time_idx, level_idx, :, :] = interp_plane
                        successful_planes += 1
                        
                        if plane_count <= 5:
                            print(f"                    -> Nearest neighbor fallback successful")
                        
                    except Exception as e2:
                        if plane_count <= 5:
                            print(f"                    -> Nearest neighbor also failed: {e2}")
                        continue
    
    print(f"            Loop interpolation completed:")
    print(f"                Total planes processed: {plane_count}")
    print(f"                Successful interpolations: {successful_planes}")
    print(f"                Success rate: {100*successful_planes/plane_count:.1f}%")
    print(f"            Final result shape: {result.shape}")
    print(f"            Result value range: {np.nanmin(result):.3f} to {np.nanmax(result):.3f}")
    print(f"            Final valid data ratio: {100*np.isfinite(result).sum()/result.size:.1f}%")
    
    if successful_planes == 0:
        raise ValueError("No planes were successfully interpolated")

    return result

def interpolate_to_regular_grid(dataset, ll_params):
    """
    將WRF數據插值到規則經緯度網格（使用scipy.griddata方法）
    
    Parameters:
    -----------
    dataset : xarray.Dataset - 要插值的WRF數據集
    ll_params : dict - 經緯度網格參數
    
    Returns:
    --------
    xarray.Dataset - 插值後的數據集
    """
    print(f"\n{'-'*30}")
    print(f"-- Grid Interpolation (scipy.griddata method) --")
    print(f"{'-'*30}")
    
    # 檢查輸入數據的經緯度坐標
    print(f"    Analyzing input dataset grid...")
    
    # 尋找經緯度坐標變數
    lon_var = None
    lat_var = None
    
    # 常見的經緯度變數名稱
    possible_lon_names = ['XLONG', 'lon', 'longitude', 'long']
    possible_lat_names = ['XLAT', 'lat', 'latitude']
    
    # 在坐標和數據變數中尋找
    all_vars = list(dataset.coords.keys()) + list(dataset.data_vars.keys())
    
    for var_name in all_vars:
        if var_name in possible_lon_names:
            lon_var = var_name
            print(f"        Found longitude variable: {lon_var}")
        elif var_name in possible_lat_names:
            lat_var = var_name
            print(f"        Found latitude variable: {lat_var}")
    
    if lon_var is None or lat_var is None:
        # 嘗試從第一個數據變數的坐標中查找
        first_var_name = list(dataset.data_vars.keys())[0]
        first_var = dataset[first_var_name]
        
        print(f"        Coordinates in {first_var_name}: {list(first_var.coords.keys())}")
        
        # 如果沒找到，拋出錯誤並提供診斷資訊
        available_coords = list(dataset.coords.keys())
        available_vars = list(dataset.data_vars.keys())
        
        print(f"        Available coordinates: {available_coords}")
        print(f"        Available data variables: {available_vars}")
        
        raise ValueError(f"Could not find longitude/latitude coordinates. "
                        f"Expected one of {possible_lon_names} for longitude and {possible_lat_names} for latitude.")
    
    # 獲取原始網格資訊
    orig_lon = dataset[lon_var]
    orig_lat = dataset[lat_var]
    
    # 提取二維座標陣列（參考成功案例的做法）
    print(f"    Extracting 2D coordinate arrays...")
    
    # 檢查座標的維度結構
    print(f"        Original longitude shape: {orig_lon.shape}")
    print(f"        Original latitude shape: {orig_lat.shape}")
    print(f"        Longitude dimensions: {orig_lon.dims}")
    print(f"        Latitude dimensions: {orig_lat.dims}")
    
    # 提取2D座標數組（模仿成功案例: XLAT[0, :, :], XLONG[0, :, :]）
    if len(orig_lon.shape) == 3:
        # 假設是 (Time, south_north, west_east) 或類似結構
        src_lon = orig_lon.isel({orig_lon.dims[0]: 0}).values  # 取第一個時間
        src_lat = orig_lat.isel({orig_lat.dims[0]: 0}).values
        print(f"        Extracted 2D coordinates from 3D arrays (using first time step)")
    elif len(orig_lon.shape) == 2:
        # 已經是2D
        src_lon = orig_lon.values
        src_lat = orig_lat.values
        print(f"        Using existing 2D coordinate arrays")
    else:
        # 嘗試其他可能的維度結構
        if len(orig_lon.shape) > 3:
            # 可能是 (member, Time, south_north, west_east) 等
            # 取第一個member和第一個time
            indices = {dim: 0 for dim in orig_lon.dims[:-2]}  # 除了最後兩個維度外都取第一個
            src_lon = orig_lon.isel(indices).values
            src_lat = orig_lat.isel(indices).values
            print(f"        Extracted 2D coordinates from high-dimensional arrays")
        else:
            raise ValueError(f"Unsupported coordinate shape: {orig_lon.shape}")
    
    print(f"    Original grid information:")
    print(f"        Longitude range: {src_lon.min():.3f}° to {src_lon.max():.3f}°")
    print(f"        Latitude range: {src_lat.min():.3f}° to {src_lat.max():.3f}°")
    print(f"        Grid shape: {src_lon.shape}")
    print(f"        Grid type: Curvilinear (WRF native)")
    
    # 創建目標規則網格
    print(f"    Creating target regular grid...")
    print(f"        Longitude: {ll_params['lon_min']}° to {ll_params['lon_max']}° (step: {ll_params['dlon']}°)")
    print(f"        Latitude: {ll_params['lat_min']}° to {ll_params['lat_max']}° (step: {ll_params['dlat']}°)")
    
    # 計算網格點數量
    n_lon = int((ll_params['lon_max'] - ll_params['lon_min']) / ll_params['dlon']) + 1
    n_lat = int((ll_params['lat_max'] - ll_params['lat_min']) / ll_params['dlat']) + 1
    
    print(f"        Target grid size: {n_lon} × {n_lat} = {n_lon * n_lat} points")
    
    # 創建目標座標
    target_lon_1d = np.linspace(ll_params['lon_min'], ll_params['lon_max'], n_lon)
    target_lat_1d = np.linspace(ll_params['lat_min'], ll_params['lat_max'], n_lat)
    
    # 創建2D網格
    target_lon, target_lat = np.meshgrid(target_lon_1d, target_lat_1d)
    
    # 檢查網格範圍是否合理
    orig_lon_range = [src_lon.min(), src_lon.max()]
    orig_lat_range = [src_lat.min(), src_lat.max()]
    target_lon_range = [ll_params['lon_min'], ll_params['lon_max']]
    target_lat_range = [ll_params['lat_min'], ll_params['lat_max']]
    
    # 檢查是否需要外插
    lon_extrapolation = (target_lon_range[0] < orig_lon_range[0] or target_lon_range[1] > orig_lon_range[1])
    lat_extrapolation = (target_lat_range[0] < orig_lat_range[0] or target_lat_range[1] > orig_lat_range[1])
    
    if lon_extrapolation or lat_extrapolation:
        print(f"    Warning: Target grid extends beyond original data domain:")
        if lon_extrapolation:
            print(f"        Longitude extrapolation detected")
            print(f"            Original: {orig_lon_range[0]:.3f}° to {orig_lon_range[1]:.3f}°")
            print(f"            Target:   {target_lon_range[0]:.3f}° to {target_lon_range[1]:.3f}°")
        if lat_extrapolation:
            print(f"        Latitude extrapolation detected")
            print(f"            Original: {orig_lat_range[0]:.3f}° to {orig_lat_range[1]:.3f}°")
            print(f"            Target:   {target_lat_range[0]:.3f}° to {target_lat_range[1]:.3f}°")
        print(f"        Extrapolated regions may have reduced accuracy")
    else:
        print(f"    Grid interpolation (no extrapolation needed)")
    
    # 執行插值
    print(f"\n    Performing batch interpolation using scipy.griddata...")
    
    try:
        # 為插值準備數據變數字典
        interpolated_vars = {}
        
        # 對每個變數進行插值
        for var_name in dataset.data_vars:
            print(f"        Processing variable: {var_name}")
            
            var_data = dataset[var_name]
            
            # 檢查變數是否有空間維度
            spatial_dims = ['south_north', 'west_east']
            has_spatial_dims = all(dim in var_data.dims for dim in spatial_dims)
            
            if not has_spatial_dims:
                print(f"            -> No spatial dimensions, copying as-is")
                interpolated_vars[var_name] = var_data
                continue
            
            print(f"            Variable shape: {var_data.shape}")
            print(f"            Variable dimensions: {var_data.dims}")
            
            # 確保變數是5維：['member', 'Time', 'interp_level', 'south_north', 'west_east']
            expected_dims = ['member', 'Time', 'interp_level', 'south_north', 'west_east']
            if list(var_data.dims) != expected_dims:
                print(f"            Warning: Variable dimensions {list(var_data.dims)} != expected {expected_dims}")
                print(f"            Attempting to reorder dimensions...")
                
                # 嘗試重新排列維度
                try:
                    var_data = var_data.transpose(*expected_dims)
                    print(f"            Successfully reordered dimensions")
                except Exception as e:
                    print(f"            Failed to reorder dimensions: {e}")
                    print(f"            Skipping variable {var_name}")
                    continue
            
            # 轉換為numpy數組進行插值
            data_array = var_data.values
            
            # 使用批量插值函數
            try:
                interpolated_array = batch_griddata_interpolation(
                    data_array, src_lon, src_lat, target_lon, target_lat
                )

                # 創建新的xarray DataArray
                new_dims = ['member', 'Time', 'interp_level', 'lat', 'lon']
                new_coords = {
                    'member': var_data.coords['member'],
                    'Time': var_data.coords['Time'],
                    'interp_level': var_data.coords['interp_level'],
                    'lat': target_lat_1d,
                    'lon': target_lon_1d
                }
                
                interpolated_var = xr.DataArray(
                    interpolated_array,
                    dims=new_dims,
                    coords=new_coords,
                    name=var_name,
                    attrs=dict(var_data.attrs)
                )
                
                # 添加插值相關的屬性
                interpolated_var.attrs.update({
                    'interpolated_to_regular_grid': 'True',
                    'interpolation_method': 'scipy_griddata_linear',
                    'original_grid_shape': str(src_lon.shape),
                    'target_grid_shape': str(target_lon.shape)
                })

                interpolated_vars[var_name] = interpolated_var
                print(f"            -> Success: {data_array.shape} -> {interpolated_array.shape}")
                
            except Exception as var_error:
                print(f"            -> Error in var_error: {str(var_error)}")
                print(f"            -> Skipping variable {var_name}")
                continue
        
        if not interpolated_vars:
            raise ValueError("No variables could be interpolated successfully")
        
        # 創建插值後的數據集
        interpolated_dataset = xr.Dataset(interpolated_vars)
        
        # 複製非空間坐標
        print(f"    Copying non-spatial coordinates...")
        
        for coord_name in dataset.coords:
            if coord_name not in [lon_var, lat_var, 'lon', 'lat']:
                # 檢查坐標是否已經在插值後的數據集中
                if coord_name not in interpolated_dataset.coords:
                    # 檢查坐標是否依賴於空間維度
                    coord_data = dataset.coords[coord_name]
                    spatial_dims_in_coord = [dim for dim in coord_data.dims if dim in ['south_north', 'west_east']]
                    
                    if len(spatial_dims_in_coord) == 0:
                        # 非空間坐標，直接複製
                        interpolated_dataset = interpolated_dataset.assign_coords({coord_name: coord_data})
                        print(f"        Copied coordinate: {coord_name}")
                    else:
                        print(f"        Skipped spatial coordinate: {coord_name} (replaced with regular grid)")
        
        # 複製全域屬性
        interpolated_dataset.attrs = dict(dataset.attrs)
        
        # 添加插值相關的全域屬性
        interpolated_dataset.attrs.update({
            'interpolation_method': 'scipy_griddata_linear_batch',
            'interpolation_target_grid': f"lon={ll_params['lon_min']}:{ll_params['dlon']}:{ll_params['lon_max']}, lat={ll_params['lat_min']}:{ll_params['dlat']}:{ll_params['lat_max']}",
            'interpolation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_grid_type': 'curvilinear_wrf',
            'target_grid_type': 'regular_latlon',
            'interpolation_extrapolation_used': str(lon_extrapolation or lat_extrapolation),
            'interpolation_batch_processing': 'True'
        })
        
        print(f"    Interpolation completed successfully:")
        print(f"        Original grid: {src_lon.shape}")
        print(f"        Target grid: {target_lon.shape}")
        print(f"        Variables interpolated: {list(interpolated_dataset.data_vars.keys())}")
        print(f"        Interpolation method: scipy.griddata (batch processing)")
        
        return interpolated_dataset
        
    except Exception as e:
        print(f"    Error during interpolation: {str(e)}")

def wrfdata_sel_optimized(I, D, L, T, E, V_list, decompose_multi=False):
    """
    優化版本：使用 MTV (Member-Time-Variable) 迴圈結構
    減少檔案 I/O 操作次數以提升性能
    """
    
    print(f"\nWRF multi-variable data extraction (OPTIMIZED MTV structure):")
    print(f"    Base directory: {I}")
    print(f"    Domain: {D}")
    print(f"    Pressure levels: {L} hPa")
    print(f"    Time points: {len(T)} times")
    print(f"    Ensemble members: {E} ({len(E)} members)")
    print(f"    Variables: {V_list} ({len(V_list)} variables)")
    print(f"    Expected file operations: {len(E) * len(T)} files (vs {len(E) * len(V_list) * len(T)} in old version)")

    # 驗證基礎目錄
    if not os.path.exists(I):
        raise FileNotFoundError(f"Base directory does not exist: {I}")

    # 獲取多分量變數資訊
    multi_component_vars = get_multi_component_variables()
    
    # 初始化數據存儲結構
    all_variables_data = {var_name: [] for var_name in V_list}
    if decompose_multi:
        # 預先計算可能的分解變數名稱
        for var_name in V_list:
            if var_name in multi_component_vars:
                component_dim = multi_component_vars[var_name]
                if component_dim == 'u_v':
                    all_variables_data[f"{var_name}_u"] = []
                    all_variables_data[f"{var_name}_v"] = []
                elif component_dim == 'wspd_wdir':
                    all_variables_data[f"{var_name}_wspd"] = []
                    all_variables_data[f"{var_name}_wdir"] = []

    successful_members = []
    total_files_processed = 0
    total_variables_processed = 0

    # -----------------
    # 優化的 MTV 迴圈結構
    # -----------------
    for e_idx, e in enumerate(E):
        print(f"    Processing member {e:03d}...")
        
        member_success = True
        member_variables_data = {}
        
        # 初始化該成員的所有變數數據列表
        for var_name in all_variables_data.keys():
            member_variables_data[var_name] = []

        # -----------------
        # T 迴圈：時間點（第二層）
        # -----------------
        for t_idx, time_str in enumerate(T):
            # 構建檔案路徑
            member_dir = f"member{e:03d}"
            filename = f"wrfout_{D}_{time_str}"
            file_path = os.path.join(I, member_dir, filename)

            # 檢查檔案是否存在
            if not os.path.exists(file_path):
                print(f"        Warning: File not found: {file_path}")
                member_success = False
                break

            print(f"        Reading: {member_dir}/{filename}")
            
            try:
                # *** 關鍵優化：每個時間檔案只打開一次 ***
                ncfile = Dataset(file_path)
                total_files_processed += 1
                
                # 存儲該時間點所有變數的數據
                time_variables_data = {}
                
                # -----------------
                # V 迴圈：變數（第三層，最內層）
                # -----------------
                for v_idx, V in enumerate(V_list):
                    try:
                        # 提取變數（檔案已經打開，直接使用）
                        var_data = wrf.getvar(ncfile, V, timeidx=0)
                        total_variables_processed += 1
                        print(f"            Variables: {V_list[v_idx]}")
                        
                        # 檢查變數維度特性
                        has_vertical_dim = any(dim in var_data.sizes for dim in ['bottom_top', 'bottom_top_stag'])
                        has_component_dim = any(dim in var_data.sizes for dim in ['u_v', 'wspd_wdir'])

                        # 處理3D/2D變數和插值
                        if has_vertical_dim:
                            if has_component_dim:
                                # 多分量3D變數的特殊處理
                                component_dim = None
                                for dim in ['u_v', 'wspd_wdir']:
                                    if dim in var_data.sizes:
                                        component_dim = dim
                                        break
                                
                                if component_dim:
                                    component_list = []
                                    n_components = var_data.sizes[component_dim]
                                    
                                    for comp_idx in range(n_components):
                                        component_data = var_data.isel({component_dim: comp_idx})
                                        component_interp = wrf.vinterp(ncfile, component_data, "pressure", L,
                                                                     extrapolate=False, timeidx=0)
                                        component_interp = component_interp.expand_dims(component_dim)
                                        component_list.append(component_interp)
                                    
                                    var_interp = xr.concat(component_list, dim=component_dim)
                                    
                                    if component_dim == 'u_v':
                                        var_interp = var_interp.assign_coords({component_dim: ['u', 'v']})
                                    elif component_dim == 'wspd_wdir':
                                        var_interp = var_interp.assign_coords({component_dim: ['wspd', 'wdir']})
                                else:
                                    var_interp = wrf.vinterp(ncfile, var_data, "pressure", L,
                                                           extrapolate=False, timeidx=0)
                            else:
                                # 單一3D變數
                                var_interp = wrf.vinterp(ncfile, var_data, "pressure", L,
                                                       extrapolate=False, timeidx=0)
                        else:
                            # 2D變數
                            var_interp = var_data.expand_dims('interp_level')
                            var_interp = var_interp.assign_coords(interp_level=['surface'])

                        # 添加時間維度
                        time_dt = pd.to_datetime(time_str, format='%Y-%m-%d_%H:%M:%S')
                        var_interp = var_interp.expand_dims('Time')
                        var_interp = var_interp.assign_coords(Time=[time_dt])
                        var_interp.name = V

                        # 存儲該變數數據
                        time_variables_data[V] = var_interp

                    except Exception as var_ex:
                        print(f"            Error processing variable {V}: {var_ex}")
                        member_success = False
                        break

                # *** 關鍵優化：檔案處理完畢後立即關閉 ***
                ncfile.close()
                
                if not member_success:
                    break

                # -----------------
                # 處理該時間點的所有變數數據（包括分解）
                # -----------------
                for V in V_list:
                    if V in time_variables_data:
                        var_data = time_variables_data[V]
                        
                        # 檢查是否需要分解多分量變數
                        if decompose_multi and V in multi_component_vars:
                            component_dim = multi_component_vars[V]
                            if component_dim in var_data.sizes:
                                # 分解變數
                                decomposed_vars = decompose_multi_component_variable(var_data, V, component_dim)
                                for decomposed_var_name, decomposed_var_data in decomposed_vars.items():
                                    if decomposed_var_name in member_variables_data:
                                        member_variables_data[decomposed_var_name].append(decomposed_var_data)
                            else:
                                # 未找到分量維度，按原樣處理
                                member_variables_data[V].append(var_data)
                        else:
                            # 不分解或非多分量變數
                            member_variables_data[V].append(var_data)

            except Exception as file_ex:
                print(f"        Error processing file {file_path}: {file_ex}")
                member_success = False
                break

        # -----------------
        # 合併該成員的時間數據
        # -----------------
        if member_success:
            print(f"        Member {e:03d} completed: processing {len([k for k, v in member_variables_data.items() if v])} variables")
            
            # 為每個變數合併時間維度並添加成員維度
            for var_name in member_variables_data:
                if member_variables_data[var_name]:  # 確保該變數有數據
                    # 沿時間維度合併
                    var_combined = xr.concat(member_variables_data[var_name], dim='Time')
                    
                    # 添加成員維度
                    var_combined = var_combined.expand_dims('member')
                    var_combined = var_combined.assign_coords(member=[e])
                    
                    # 添加到總數據中
                    if var_name not in all_variables_data:
                        all_variables_data[var_name] = []
                    all_variables_data[var_name].append(var_combined)

            successful_members.append(e)
        else:
            print(f"        Warning: Member {e:03d} failed")

    # -----------------
    # 合併所有成員數據
    # -----------------
    if not any(all_variables_data.values()):
        raise ValueError("No data was successfully loaded!")

    print(f"    Performance summary:")
    print(f"        Total files opened: {total_files_processed} (vs {len(E) * len(V_list) * len(T)} in old version)")
    print(f"        Total variables extracted: {total_variables_processed}")
    #print(f"        File I/O reduction: {100 * (1 - total_files_processed / (len(E) * len(V_list) * len(T))):.1f}%")

    # 創建最終數據集
    final_dataset_dict = {}
    
    for var_name in all_variables_data:
        if all_variables_data[var_name]:
            var_combined = xr.concat(all_variables_data[var_name], dim='member')
            var_combined = var_combined.assign_coords(member=successful_members)
            final_dataset_dict[var_name] = var_combined

    final_dataset = xr.Dataset(final_dataset_dict)

    print(f"\nOptimized multi-variable extraction completed:")
    print(f"    Variables: {list(final_dataset.data_vars.keys())}")
    print(f"    Dimensions: {list(final_dataset.sizes.keys())}")
    print(f"    Successful members: {successful_members}")

    return final_dataset


#===============================================================
def main():
    """主程序"""
    # 解析命令列參數
    args = parse_arguments()
    
    # 確保輸出目錄存在
    output_dir = ensure_output_directory(args.output_dir)
    
    try:
        # -----------------
        # 參數解析和格式化
        # -----------------
        print(f"\n{'='*99}")
        print(f"== WRF Multi-Variable Data Extraction Tool with Grid Interpolation + Averaging ==")
        print(f"{'='*99}")
        
        # 解析變數參數
        variables = [v.strip() for v in args.variable.split(',')]
        #print(f"    Variables to extract: {variables}")
        
        # 解析氣壓層參數
        if args.levels.lower() in ['surface', 'sfc', '2d']:
            levels = [-9999]  # 特殊值表示地面變數
        else:
            levels = [float(l.strip()) for l in args.levels.split(',')]
        
        # 解析系集成員參數
        ensemble_members = [int(e.strip()) for e in args.ensemble.split(',')]
        
        # 解析時間參數
        if args.times is None:
            # 自動偵測可用時間點
            time_points = detect_available_times(args.input_dir, args.domain, ensemble_members)
            # 使用第一個時間點
            times = [time_points[0]]
            print(f"    Using first available time: {times[0]}")
        else:
            times = [t.strip().strip('"') for t in args.times.split(',')]
        
        # -----------------
        # 提取WRF數據
        # -----------------
        print(f"\n{'='*50}")
        print(f"== Extracte Dataset Using wrf.getvar ==")
        print(f"{'='*50}")

        #**** v1.3 update  ****
        extracted_dataset = wrfdata_sel_optimized(  # 使用優化版本
            I=args.input_dir,
            D=args.domain, 
            L=levels,
            T=times,
            E=ensemble_members,
            V_list=variables,
            decompose_multi=args.decompose_multi
        )

        # -----------------
        # 重新命名lon/lat變數
        # -----------------
        extracted_dataset = rename_lonlat_variables(extracted_dataset)
        
        # -----------------
        # *** 新增：網格插值處理 ***
        # -----------------
        if args.lonlat_grid:
            print(f"\n{'='*50}")
            print(f"== Grid Interpolation ==")
            print(f"{'='*50}")
            
            # 解析經緯度網格參數
            try:
                ll_params = parse_ll_parameter(args.lonlat_grid)
                print(f"    Target grid parameters:")
                print(f"        Longitude: {ll_params['lon_min']}° to {ll_params['lon_max']}° (step: {ll_params['dlon']}°)")
                print(f"        Latitude: {ll_params['lat_min']}° to {ll_params['lat_max']}° (step: {ll_params['dlat']}°)")
                
                # 執行網格插值
                extracted_dataset = interpolate_to_regular_grid(extracted_dataset, ll_params)
                
            except Exception as interp_error:
                print(f"    Error during grid interpolation: {str(interp_error)}")
                print(f"    Continuing without interpolation...")
                print(f"    Pleace check:")
                print(f"    Please verify the following:")
                print(f"        1. The format of the -LL parameter is correct.")
                print(f"        2. The WRF data includes valid latitude and longitude coordinates.")
                print(f"        3. The target grid range is within the bounds of the original dataset.")
        
        # -----------------
        # *** 修改：平均處理 (現在在插值之後) ***
        # -----------------
        if args.Emean or args.Tmean:
            print(f"\n{'='*50}")
            print(f"== Applying Averaging Operations ==")
            print(f"{'='*50}")
            
            # 顯示原始數據維度
            print(f"    Original data dimensions:")
            for var_name in extracted_dataset.data_vars:
                var_data = extracted_dataset[var_name]
                print(f"        {var_name}: {dict(var_data.sizes)}")
            
            # 應用系集平均
            if args.Emean:
                extracted_dataset = apply_ensemble_mean(extracted_dataset)
            
            # 應用時間平均
            if args.Tmean:
                extracted_dataset = apply_time_mean(extracted_dataset)
            
            # 顯示最終數據維度
            print(f"\n    Final data dimensions after averaging:")
            for var_name in extracted_dataset.data_vars:
                var_data = extracted_dataset[var_name]
                print(f"        {var_name}: {dict(var_data.sizes)}")
                
                # 顯示平均相關資訊
                if hasattr(var_data, 'attrs'):
                    if var_data.attrs.get('ensemble_averaged') == 'True':
                        print(f"            -> Ensemble averaged from {var_data.attrs.get('original_ensemble_size')} members")
                    if var_data.attrs.get('time_averaged') == 'True':
                        print(f"            -> Time averaged from {var_data.attrs.get('original_time_points')} time points")
            
        
        # -----------------
        # 保存數據（如果指定輸出檔案）- *** 修改的部分 ***
        # -----------------
        
        if args.output_file:
            print(f"\n{'='*50}")
            print(f"== Save to NC File ==")
            print(f"{'='*50}")

            output_path = os.path.join(output_dir, args.output_file)
            
            # 根據處理操作自動調整檔名
            if args.Emean or args.Tmean or args.lonlat_grid:
                base_name, ext = os.path.splitext(args.output_file)
                suffix = ""
                if args.lonlat_grid:
                    # 從網格參數生成簡短的標識
                    ll_parts = args.lonlat_grid.split(',')
                    if len(ll_parts) >= 5:
                        grid_id = f"grid{ll_parts[4]}"  # 使用解析度作為標識
                        suffix += f"_{grid_id}"
                if args.Emean:
                    # 系集平均
                    suffix += "_Emean"
                if args.Tmean:
                    # 時間平均
                    suffix += "_Tmean"
                if levels[0] == -9999:
                    # 地面層
                    suffix += "_surface"
                output_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
                print(f"    Output filename adjusted for processing: {os.path.basename(output_path)}")
            
            # 確保檔案有.nc副檔名
            if not output_path.endswith('.nc'):
                output_path += '.nc'
            
            # *** 新增WRF全域屬性相關參數 ***
            save_to_netcdf(
                data=extracted_dataset, 
                output_path=output_path, 
                compression_level=args.compression,
                wrf_base_dir=args.input_dir,           # 新增：WRF檔案根目錄
                wrf_domain=args.domain,                # 新增：Domain名稱
                wrf_ensemble_members=ensemble_members, # 新增：系集成員列表
                wrf_time_points=times                  # 新增：時間點列表
            )
        else:
            print(f"\nNo output file specified. Data extraction completed but not saved.")
        
        print(f"\n{'='*99}")
        print(f"== Multi-variable extraction with grid interpolation and averaging completed done ==")
        print(f"{'='*99}")
        
        return extracted_dataset
        
    except Exception as e:
        print(f"\nError during extraction: {str(e)}")
        sys.exit(1)
#===============================================================

# -----------------
# 使用範例和測試
# -----------------
if __name__ == "__main__":
    extracted_data = main()
