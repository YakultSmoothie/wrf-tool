#!/usr/bin/env python3
# ===========================================================================================
# NetCDF轉CTL檔案生成器 - 針對WRF輸出檔案
# 使用Python和xarray讀取NetCDF檔案，自動提取維度、座標、屬性資訊並生成GrADS CTL檔案
# ===========================================================================================
# 檔名: nc2ctl_for_extract_wrf_to_nc.py
# 功能: 使用xarray讀取NetCDF檔案並自動生成對應的GrADS CTL檔案，支援WRF投影轉換
# 作者: CYC, create: 2025-06-22
# 執行範例: python3 nc_to_ctl.py -i input.nc
# create: 2025-06-22
# update (v1.5): 2025-10-02 (修正XTIME純量處理問題)
# ===========================================================================================

import xarray as xr
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import pandas as pd

# -----------------
# 參數設定區
# -----------------
default_input_file = "./sfc_Emean_surface.nc"     # 預設輸入檔名
default_output_file = ""                         # 預設輸出檔名(空字串表示自動產生)
default_output_dir = "./"                        # 預設輸出目錄
default_quiet = False                            # 預設非安靜模式

# -----------------
# 命令列參數解析
# -----------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="NetCDF to CTL file converter for WRF output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
執行範例 (Run examples):
  python nc_to_ctl.py -i sfc_Emean_surface.nc
  python nc_to_ctl.py -i input.nc -o custom_output.ctl
  python nc_to_ctl.py -i ./WRF/m001/input.nc -n ./WRF
        """)

    parser.add_argument('-i', '--input', default=default_input_file,
                        help=f'輸入NetCDF檔名 (預設: {default_input_file})')
    parser.add_argument('-o', '--output', default=default_output_file,
                        help='輸出CTL檔名 (預設: 自動產生)')
    parser.add_argument('-n', '--output_dir', default=None,
                        help=f'輸出目錄路徑 (預設: {default_output_dir})')

    args = parser.parse_args()

    # 如果沒有指定輸出目錄，使用輸入檔案的所在路徑
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input))
        if args.output_dir == "":  # 如果檔案在當前目錄
            args.output_dir = "./"
        elif not args.output_dir.endswith("/"):  # 確保路徑以 / 結尾
            args.output_dir += "/"

    return args

# -----------------
# 月份名稱轉換函數
# -----------------
def month_to_abbrev(month_num):
    """將月份數字轉換為月份縮寫"""
    month_map = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
        7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
    }
    return month_map.get(month_num, "jan")

# -----------------
# 投影參數處理函數
# -----------------
def get_projection_info(ds):
    """提取WRF投影相關資訊"""
    proj_info = {}

    # 提取WRF相關屬性
    wrf_attrs = ['CEN_LAT', 'CEN_LON', 'TRUELAT1', 'TRUELAT2', 'STAND_LON',
                 'DX', 'DY', 'MAP_PROJ', 'MAP_PROJ_CHAR']

    for attr in wrf_attrs:
        # 嘗試不同的屬性名稱格式
        attr_names = [attr, f'WRF_{attr}']
        for attr_name in attr_names:
            if attr_name in ds.attrs:
                proj_info[attr] = ds.attrs[attr_name]
                break

        # 如果找不到，給預設值
        if attr not in proj_info:
            if attr in ['CEN_LAT', 'TRUELAT1', 'TRUELAT2']:
                proj_info[attr] = 1  # 預設緯度
            elif attr in ['CEN_LON', 'STAND_LON']:
                proj_info[attr] = 1  # 預設經度
            elif attr in ['DX', 'DY']:
                proj_info[attr] = 110  # 預設格點數
            elif attr == 'MAP_PROJ':
                proj_info[attr] = 1  # 預設Lambert投影
            elif attr == 'MAP_PROJ_CHAR':
                proj_info[attr] = "Lambert Conformal"

    return proj_info

# -----------------
# 時間資訊處理函數
# -----------------
def get_time_info(ds):
    """提取時間資訊"""
    import pandas as pd
    from datetime import datetime

    time_info = {}

    # 檢查是否有時間維度
    if 'Time' in ds:
        time_info['time_dim'] = ds.sizes['Time']

        # 嘗試從 Time.units 獲取起始時間
        if hasattr(ds['Time'], 'units') and 'since' in ds['Time'].units:
            try:
                date_part = ds['Time'].units.split('since')[1].strip()
                dt = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
                time_info['start_year'] = dt.year
                time_info['start_month'] = dt.month
                time_info['start_day'] = dt.day
                time_info['start_hour'] = dt.hour
                print(f"    從Time.units解析起始時間: {dt}")
            except:
                print(f"    警告: 無法解析Time.units格式")

        # 嘗試從 datetime64 格式獲取起始時間
        elif ds['Time'].dtype.kind == 'M':
            try:
                start_time = pd.to_datetime(ds['Time'].values[0])
                time_info['start_year'] = start_time.year
                time_info['start_month'] = start_time.month
                time_info['start_day'] = start_time.day
                time_info['start_hour'] = start_time.hour
                print(f"    從datetime64格式解析起始時間: {start_time}")
            except:
                print(f"    警告: 無法解析datetime64格式")

        # 嘗試從 XTIME 計算時間間隔
        if 'XTIME' in ds:
            try:
                xtime_var = ds['XTIME']
                
                # 檢查 XTIME 是否為陣列且有多於一個元素
                if xtime_var.ndim > 0 and xtime_var.size > 1:
                    xtime_values = xtime_var.values
                    print(f"    XTIME的時間: {xtime_values}")
                    time_interval = int(xtime_values[1] - xtime_values[0])
                    time_info['time_interval'] = time_interval
                    time_info['time_unit'] = "mn"
                    print(f"    從XTIME計算時間間隔: {time_interval} mn (xtime_values[1] - xtime_values[0])")
                else:
                    print(f"    XTIME為純量或單一值，無法計算時間間隔")
                    print(f"    XTIME值: {xtime_var.values}")
            except Exception as e:
                print(f"    警告: 無法從XTIME計算間隔 - {e}")

    # 檢查並補齊缺少的必要元素
    required_keys = {
        'time_dim': 1,
        'start_year': 9999,
        'start_month': 1,
        'start_day': 1,
        'start_hour': 0,
        'time_interval': 1,
        'time_unit': "yr"
    }

    for key, default_value in required_keys.items():
        if key not in time_info:
            time_info[key] = default_value
            print(f"    使用預設值: {key} = {default_value}")

    return time_info

# -----------------
# 主程式
# -----------------
def main():

    # 解析命令列參數
    args = parse_arguments()

    # 檢查輸入檔案是否存在
    if not os.path.exists(args.input):
        print(f"錯誤: 找不到輸入檔案 '{args.input}'")
        sys.exit(1)

    # 確保輸出目錄存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"建立輸出目錄: {args.output_dir}")

    # 決定輸出檔案名稱
    if args.output:
        output_file = os.path.join(args.output_dir, args.output)
    else:
        # 自動產生輸出檔名，將.nc替換為.ctl
        base_name = os.path.basename(args.input)
        if base_name.endswith('.nc'):
            ctl_name = base_name[:-3] + '.ctl'
        else:
            ctl_name = base_name + '.ctl'
        output_file = os.path.join(args.output_dir, ctl_name)

    print(f"處理檔案: {args.input}")
    print(f"輸出CTL檔: {output_file}")

    # -----------------
    # OPEN - 讀取NetCDF檔案
    # -----------------
    try:
        print("開始讀取NetCDF檔案...")
        ds = xr.open_dataset(args.input)
        print(f"    成功開啟檔案")

        # -----------------
        # 讀檔information - 顯示檔案基本資訊
        # -----------------
        print(f"NetCDF檔案資訊:")
        print(f"    維度: {dict(ds.sizes)}")
        print(f"    變數數量: {len(ds.data_vars)}")
        print(f"    座標: {list(ds.coords.keys())}")
        print(f"    全域屬性數量: {len(ds.attrs)}")

    except Exception as e:
        print(f"讀取NetCDF檔案時發生錯誤: {e}")
        sys.exit(1)

    # -----------------
    # DEFINE處理資料 - 提取各種資訊
    # -----------------
    print("提取檔案元資料...")

    # 提取維度資訊
    west_east = ds.sizes.get('west_east', 0)
    south_north = ds.sizes.get('south_north', 0)
    time_dim_size = ds.sizes.get('Time', 1)
    member_dim = ds.sizes.get('member', 0)

    print(f"    網格大小: {west_east} x {south_north}")
    print(f"    時間步數: {time_dim_size}")
    print(f"    系集成員數: {member_dim if member_dim > 0 else 1}")

    # 提取投影資訊
    proj_info = get_projection_info(ds)
    print(f"    地圖投影: {proj_info.get('MAP_PROJ_CHAR', 'Lambert Conformal')} (代碼: {proj_info.get('MAP_PROJ', 1)})")
    print(f"    中心點: {proj_info.get('CEN_LAT', 99.9)}°N, {proj_info.get('CEN_LON', 999.9)}°E")
    print(f"    格距: {proj_info.get('DX', 99999)}m x {proj_info.get('DY', 99999)}m")

    # 提取時間資訊
    time_info = get_time_info(ds)
    print(f"    起始時間: {time_info['start_hour']:02d}z{time_info['start_day']:02d}{month_to_abbrev(time_info['start_month'])}{time_info['start_year']}")
    print(f"    時間間隔: {time_info['time_interval']}{time_info['time_unit']}")

    # 提取垂直層資訊
    if 'interp_level' in ds.coords:
        interp_levels = ds.coords['interp_level'].values
        interp_dim = len(interp_levels)
        print(f"    垂直層數: {interp_dim}")
        print(f"    層次: {interp_levels}")

        # 檢查並替換 'surface' 為 -9999
        if 'surface' in interp_levels:
            interp_levels = [-9999 if level == 'surface' else level for level in interp_levels]
            print(f"    已將 'surface' 替換為 -9999")
            print(f"    修改後層次: {interp_levels}")

    else:
        interp_levels = [99999]  # 預設值
        interp_dim = 1
        print(f"    警告: 找不到'interp_level'座標，使用預設值: {interp_levels}")

    # 計算投影參數
    iref = west_east / 2 + 0.5
    jref = south_north / 2 + 0.5

    # 格距計算（從投影參數）
    dx_deg = proj_info.get('DX', 15000) / 111000  # 粗略轉換公尺為度
    dy_deg = proj_info.get('DY', 15000) / 111000

    # 從實際座標變數計算經緯度範圍和格點數
    if 'XLONG' in ds and 'XLAT' in ds:
        # 讀取經緯度座標
        xlong = ds['XLONG'].values
        xlat = ds['XLAT'].values

        # 取時間第一個的座標（如果有時間維度）
        if xlong.ndim > 2:
            xlong = xlong[0]  # 取第一個時間
            xlat = xlat[0]

        # 經度 <= 0 時加上 360
        xlong[xlong <= 0] += 360

        # 計算經緯度範圍
        lon_min = np.min(xlong)
        lon_max = np.max(xlong)
        lat_min = np.min(xlat)
        lat_max = np.max(xlat)

        # 計算格點數：實際範圍除以格距，再加4格點
        xdef_points = int((lon_max - lon_min) / dx_deg) + 4
        ydef_points = int((lat_max - lat_min) / dy_deg) + 4

        # 起始點：最小值往前推2格點
        lon_start = lon_min - dx_deg * 2
        lat_start = lat_min - dy_deg * 2

        print(f"    從XLONG/XLAT計算:")
        print(f"        經度範圍: {lon_min:.4f}° 到 {lon_max:.4f}°")
        print(f"        緯度範圍: {lat_min:.4f}° 到 {lat_max:.4f}°")
        print(f"        投影格距: dx={dx_deg:.6f}°, dy={dy_deg:.6f}°")
        print(f"        XDEF格點數: {xdef_points}")
        print(f"        YDEF格點數: {ydef_points}")
        print(f"        起始點: lon={lon_start:.6f}°, lat={lat_start:.6f}°")

    else:
        # 備用方案：估算經緯度範圍（當找不到XLONG/XLAT時）
        print(f"    警告: 找不到XLONG/XLAT座標變數，使用估算方式")
        lon_span = west_east * dx_deg
        lat_span = south_north * dy_deg
        lon_start = proj_info.get('CEN_LON', 120.5) - lon_span / 2
        lat_start = proj_info.get('CEN_LAT', 23.5) - lat_span / 2
        xdef_points = west_east + 52  # 原始估算方式
        ydef_points = south_north + 22

    # 提取資料變數（過濾座標變數）
    skip_vars = {'XLONG', 'XLAT', 'XTIME', 'interp_level', 'Time', 'time'}
    data_vars = []

    for var_name in ds.data_vars:
        if var_name not in skip_vars:
            data_vars.append(var_name)

    print(f"    資料變數數量: {len(data_vars)}")
    print(f"    變數名稱: {data_vars}")

    # -----------------
    # WRITE AND SAVE - 生成CTL檔案
    # -----------------
    print("生成CTL檔案...")

    try:
        with open(output_file, 'w') as f:
            # 基本資訊
            f.write(f"DSET ^{os.path.basename(args.input)}\n")
            f.write("DTYPE netcdf\n")
            title = ds.attrs.get('title', ds.attrs.get('TITLE', 'WRF Output Data'))
            f.write(f"TITLE {title}\n")
            f.write("UNDEF -999.\n")

            # 投影定義 PDEF
            map_proj = proj_info.get('MAP_PROJ', 1)
            if map_proj == 1:  # Lambert Conformal
                f.write(f"PDEF  {west_east} {south_north} lccr  "
                       f"{proj_info.get('CEN_LAT', 99.9)} {proj_info.get('CEN_LON', 999.9)}    "
                       f"{iref:.1f} {jref:.1f}    "
                       f"{proj_info.get('TRUELAT2', 99.9)} {proj_info.get('TRUELAT1', 99.9)} "
                       f"{proj_info.get('STAND_LON', 999.9)}    "
                       f"{proj_info.get('DX', 99999)} {proj_info.get('DY', 99999)}\n")
            elif map_proj == 2:  # Polar Stereographic
                f.write(f"pdef  {west_east} {south_north} nps  "
                       f"{iref:.1f} {jref:.1f} {proj_info.get('STAND_LON', 999.9)} "
                       f"{proj_info.get('DX', 99999)/1000:.3f}\n")
            else:  # 預設為Lambert Conformal
                f.write(f"pdef  {west_east} {south_north} lccr  "
                       f"{proj_info.get('CEN_LAT', 99.9)} {proj_info.get('CEN_LON', 999.9)}    "
                       f"{iref:.1f} {jref:.1f}    "
                       f"{proj_info.get('TRUELAT2', 99.9)} {proj_info.get('TRUELAT1', 99.9)} "
                       f"{proj_info.get('STAND_LON', 999.9)}    "
                       f"{proj_info.get('DX', 99999)} {proj_info.get('DY', 99999)}\n")

            # 座標定義
            f.write(f"XDEF  {xdef_points}  LINEAR  {lon_start:.6f}  {abs(dx_deg):.6f}\n")
            f.write(f"YDEF  {ydef_points}  LINEAR  {lat_start:.6f}  {abs(dy_deg):.6f}\n")

            # 垂直層定義
            f.write(f"ZDEF {interp_dim} LEVELS")
            for level in interp_levels:
                f.write(f" {level}")
            f.write("\n")

            # 時間定義
            month_abbr = month_to_abbrev(time_info['start_month'])
            f.write(f"TDEF {time_info['time_dim']} LINEAR "
                   f"{time_info['start_hour']:02d}z{time_info['start_day']:02d}{month_abbr}{time_info['start_year']} "
                   f"{time_info['time_interval']}{time_info['time_unit']}\n")

            # 系集定義
            if member_dim > 0:
                f.write(f"EDEF {member_dim} NAMES")
                for i in range(1, member_dim + 1):
                    f.write(f" {i:03d}")
                f.write("\n")
            else:
                f.write("EDEF 1 NAMES 001\n")

            f.write("\n")

            # 風向量配對（網格風轉經緯風）
            f.write("# 網格風轉經緯風，必須是小寫\n")
            f.write("VECTORPAIRS  ua,va u10,v10\n")
            f.write("\n")

            # 變數定義, 補上兩個變數
            f.write(f"VARS {len(data_vars) + 2}\n")
            for var_name in data_vars:
                var_name_lower = var_name.lower()

                # 取得變數的單位，如果沒有則使用空字串
                try:
                    units = ds[var_name].attrs.get('units', '')
                    if units == '':
                        units = '1'  # 如果單位為空，使用 '1' 表示無量綱
                except:
                    units = '1'

                print(f"    變數 {var_name}: 單位 = '{units}'")
                f.write(f"    {var_name}=>{var_name_lower}  {interp_dim}  e,t,z,y,x [{units}]\n")
            f.write(f"    XLAT=>xlat 0  y,x [deg]\n")
            f.write(f"    XLONG=>xlon 0  y,x [deg]\n")
            f.write("ENDVARS\n")

        print(f"    CTL檔案成功寫入: {output_file}")

    except Exception as e:
        print(f"寫入CTL檔案時發生錯誤: {e}")
        sys.exit(1)

    # -----------------
    # 顯示完成資訊
    # -----------------
    print("\n=== 處理完成摘要 ===")
    print(f"關鍵參數:")
    print(f"  網格大小: {west_east} x {south_north}")
    print(f"  中心點: {proj_info.get('CEN_LAT', 99.9)}°N, {proj_info.get('CEN_LON', 999.9)}°E")
    print(f"  地圖投影: {proj_info.get('MAP_PROJ_CHAR', 'Lambert Conformal')} (代碼: {proj_info.get('MAP_PROJ', 1)})")
    print(f"  真實緯度: {proj_info.get('TRUELAT1', 99.9)}°, {proj_info.get('TRUELAT2', 99.9)}°")
    print(f"  標準經度: {proj_info.get('STAND_LON', 999.9)}°")
    print(f"  格距: {proj_info.get('DX', 99999)}m x {proj_info.get('DY', 99999)}m")
    print(f"  垂直層: {interp_levels}")
    print(f"  時間間隔: {time_info['time_interval']}{time_info['time_unit']}")
    print(f"  起始時間: {time_info['start_hour']:02d}z{time_info['start_day']:02d}{month_abbr}{time_info['start_year']}")
    print(f"  系集成員: {member_dim if member_dim > 0 else 1}")
    print(f"  變數數量: {len(data_vars)} + 2")
    print("")
    print(f"CTL檔案成功生成: {output_file}")

    # 關閉資料集
    ds.close()


if __name__ == "__main__":

    print(f"\n{'='*99}")
    print(f"== RUN {__file__} ==")
    print(f"{'='*99}\n")

    main()

    print(f"\n{'='*99}")
    print(f"== END {__file__} ==")
    print(f"{'='*99}\n")
