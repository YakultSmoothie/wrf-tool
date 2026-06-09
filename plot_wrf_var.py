#!/usr/bin/env python3
# =============================================================================================
# ==== INFOMATION ========
# ========================
'''
檔名: plot_var_LL.py
功能: 從WRF輸出檔案讀取並可視化指定變數至lon-lat水平剖面上
作者: YakultSmoothie

Create: 2025-03-07
  Update: 2025-03-08 - YakultSmoothie
    - 添加經緯度網格標籤控制選項(-glc)
    - 移除不使用的選項(-fr, -nb)
    - 修復時間字串格式化顯示問題
  Update: 2025-03-11 - 增加-ext功能
  Update: 2025-03-12 - 增加sigma層繪製功能
  Update: 2025-03-13 - 調整預設:  -V  slp; -gl 10,10
  Update: 2025-03-17 - CYC
    - 增加-L支持逗號分隔的多層次繪圖功能
    - 修改plot_wrf_variable函數返回Fig和Ax以支持多層次顯示
    - 增加自動輸出檔名功能
  Update: 2025-03-20 - CYC
    - 增加-ba參數實現基礎數值運算功能
    - 增加-vx和-vy參數以選取特定顯示範圍
  Update: 2025-03-21 - CYC - v1.4.0
    - 增加-ve參數實現風場向量繪製功能
    - 根據不同層類型自動選擇適當的風場數據來源
    - 支持風場數據的裁切和向量密度控制
    - -n 開新資料夾
    - 增強-cb功能，支持指定不同變數來繪製等值線
    - 增加-ci參數控制等值線間距
    - 增加-cc參數控制等值線顏色
  Update: 2025-03-21 - CYC - v1.4.1
    - 修正不給contour會出錯的問題
    - 修正VAR, VAR2沒轉換為掩碼陣列的問題
    - 調整軸上刻度的數量
  Update: 2025-03-25 - CYC
    - 新增 bm = get_basemap(VAR_2D) 用於獲取基礎地圖對象
    - 移除了 -nc/--no_coastline 參數，新增 -cp/--coastline_option 參數
    - 用zorder修改排序
  
Description:
    此程式讀取WRF輸出檔案，繪製指定變數在特定氣壓層的空間分佈圖。
    支援一次畫shaded, countor, vector 
 
Disclaimer:
    This software is provided "as is" without warranty of any kind.
    The authors are not liable for any damages arising from its use.
    Use at your own risk.
'''
# ============================================================================================
import sys
import os
import argparse
from datetime import datetime
from matplotlib import colormaps, cm
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from wrf import (to_np, getvar, interplevel, vinterp, smooth2d, latlon_coords, ALL_TIMES, get_proj_params, get_cartopy, get_basemap)
from mpl_toolkits.axes_grid1 import make_axes_locatable


args_str = ' '.join(sys.argv[0:])    # 擷取輸入元素
print(f"\n======= RUN: {args_str} =========\n")    # 顯示輸入元素

#------------------------------------
def wind_to_uv(wspd, wdir):
    """將風速風向轉換為U和V分量
    
    Args:
        wspd: 風速數組
        wdir: 風向數組(從北方順時針度數)
    
    Returns:
        tuple: (u, v) 風場的U和V分量
    """
#    print(wspd)
#    print(wdir)
    wspd_np = to_np(wspd)
    wdir_np = to_np(wdir)
    u = -wspd_np * np.sin(np.radians(wdir_np))
    v = -wspd_np * np.cos(np.radians(wdir_np))
    print(f"風場資料轉換完成: wspd=[{np.min(wspd_np):.2f}, {np.max(wspd_np):.2f}], " 
          f"wdir=[{np.min(wdir_np):.1f}, {np.max(wdir_np):.1f}]")
    return u, v

#------------------------------------
def parse_arguments(args=None):
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='從WRF輸出檔案讀取並可視化指定變數',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
 1. 繪製海平面氣壓圖
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V slp 
 2. 繪製850hPa等壓面位勢高度圖(地形下輸出為nan)
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V z -L 850 
 3. 啟用外插處理低於地形的數值(-V z 與 tk, 需選取對應的 -ext)
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V z -L 1000 -ex -ext z
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V tk -L 1000 -ex -ext tk 
 4. 繪製sigma層的變數(如溫度場在第0層)
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V tk -L 0 --sigma 
 5. 自定義顏色表和色階數
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V td2 -cm viridis -lnl 15
 6. 指定色階範圍
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V rh -L 850 -lcl 30,40,50,60,70,80,90
 7. 指定視圖範圍
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V slp -vx 10,50 -vy 5,30
 8. 繪製同層風場向量
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V slp -ve
 9. 在氣壓層上繪製等值線 (與主變數不同)
     python3 plot_wrf_var.py -i wrfout_d01_2006-06-09_00:00:00 -V rh -L 850 -cb z -ci 20

常用WRF變數代碼:
  slp: 海平面氣壓          z: 重力位高度            temp: 溫度             td: 露點溫度
  rh: 相對濕度             ua/va: 水平風場          pw: 可降水量           cape_2d: CAPE指數
  cloud*: 雲量相關         dbz: 雷達回波            th: 位溫               tk: 開氏溫度
  更多變數請參閱: https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.getvar.html

氣壓層內插方法:
  https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.vinterp.html

CYC, v1.5.0, @ JET

        """)

    parser.add_argument('-i', '--input',
                       default='/home/ox/pCloud/01-Backup/sample_data/wrfout/wrfout_d02_2006-06-09_00:00:00',
                       help='WRF輸出檔案路徑')
    parser.add_argument('-V', '--variable',
                       default='slp',
                       help='要繪製的變數代碼 (例如: slp, z, temp)')
    parser.add_argument('-L', '--level',
                       default='-9999',
                       help='氣壓層 (單位: hPa，地表變數為-9999)，可用逗號分隔指定多層 (例如: 850,700,500)')
    parser.add_argument('-o', '--output',
                       help='輸出圖檔路徑，不指定則顯示在畫面上；多層次時自動加入層次標識')
    parser.add_argument('-n', '--output_dir',
                       default='.',
                       help='輸出資料夾路徑，不存在時會自動創建')
    parser.add_argument('-cm', '--colormap',
                       default='turbo',
                       help='顏色表名稱 (預設: turbo)')
    parser.add_argument('-lnl', '--num_levels',
                       type=int,
                       default=10,
                       help='色階數量 (預設: 10)')
    parser.add_argument('-lcl', '--level_contours',
                       help='自定義色階範圍，用逗號分隔，例如: 10,20,50,100,200')
    
    # <<< 等值線相關參數 - 修改部分 START >>>
    parser.add_argument('-cb', '--contour',
                       default=None,
                       help='要繪製等值線的變數 (例如: slp, z, temp)，不指定則不繪製')
    parser.add_argument('-ci', '--contour_cint',
                       type=float, 
                       default=None,
                       help='等值線的間距 (預設: 自動計算，與主變數相同)')
    parser.add_argument('-cc', '--contour_color',
                       default='black',
                       help='等值線的顏色 (預設: black)')
    parser.add_argument('-clw', '--contour_linewidth',
                       type=float,
                       default=0.7,
                       help='等值線的線寬 (預設: 0.7)')
    parser.add_argument('-cext', '--contour_extrapolate_type',
                       default='none',
                       help='contour_field_type')
    # <<< 等值線相關參數 - 修改部分 END >>>

    # 經緯度格線相關參數
    parser.add_argument('-gl', '--gridlines',
                       default='10,10',
                       help='經緯度格線間距，用逗號分隔（經度,緯度），例如: 10,10')
    parser.add_argument('-glc', '--grid_labels',
                       type=lambda x: x.lower() not in ['false', 'f', 'no', 'n', '0'],
                       default=True,
                       help='是否顯示經緯度格線標籤 (True/False，預設: True)')

    parser.add_argument('-ex', '--extrapolate',
                       action='store_true',
                       help='是否外插低於地形的數值')
    parser.add_argument('-ext', '--extrapolate_type',
                       default='none',
                       help='field_type')
    parser.add_argument('-bs', '--base_size',
                       type=float,
                       default=5.0,
                       help='圖形基準尺寸 (預設: 5.0)')
    parser.add_argument('-t', '--time_idx',
                       type=int,
                       default=0,
                       help='時間索引 (預設: 0)')
  #  parser.add_argument('-nc', '--no_coastline',
  #                     action='store_true',
  #                     help='不繪製海岸線')
    parser.add_argument('-cp', '--coastline_option',
                       type=int,  # 明確指定類型為整數
                       default=2,
                       choices=[0, 1, 2],  # 限制可接受的值
                       help='type of coastline, it must be 0, 1 or 2')
    parser.add_argument('-na', '--no_annotation',
                       action='store_true',
                       help='不添加註釋信息')
    parser.add_argument('-sg', '--sigma',
                       action='store_true',
                       help='將-L參數視為sigma層索引而非氣壓層 (地表變數不適用)')
    parser.add_argument('-ba', '--basic_arithmetic', 
                       help='基礎數學運算 (例如: "+1", "*2", "-273.15")')
    parser.add_argument('-bu', '--basic_unit',
                       default=None,
                       help='基礎運算後的單位 (例如: "C" 代替 "K")(default: None)')
    parser.add_argument('-vx', '--view_x_range',
                       help='指定X軸顯示範圍，格式為"x1,x2"')
    parser.add_argument('-vy', '--view_y_range',
                       help='指定Y軸顯示範圍，格式為"y1,y2"')
                       
    # ------- ; 風場向量相關參數 ; --------
    parser.add_argument('-ve', '--vector',
                       action='store_true',
                       help='是否繪製風場向量')
    parser.add_argument('-vs', '--vector_skip',
                       type=int,
                       default=10,
                       help='風場向量間隔，數值越大間隔越大 (預設: 10)')
    parser.add_argument('-vc', '--vector_color',
                       default='blue',
                       help='風場向量顏色 (預設: blue)')
    parser.add_argument('-vw', '--vector_width',
                       type=float,
                       default=0.005,
                       help='風場向量線寬 (預設: 0.005)')
    parser.add_argument('-vl', '--vector_scale',
                       type=float,
                       default=-1,
                       help='vector比例 (預設: -1 [=max wind wpeed * 15]). 大scale值則箭頭短')
    parser.add_argument('-vf', '--vector_ref',
                       type=float,
                       default=-1,
                       help='圖標的參考風速 (預設: -1 [=max wind wpeed])')

    # ------- ^ 風場向量相關參數 ^ --------

    return parser.parse_args(args)

#------------------------------------
def plot_wrf_variable(ifn, var_plot='slp', level_p=-9999, output=None, colormap='jet', output_dir='.',
                     num_levels=10, level_contours=None, contour=None, gridlines='10,10',
                     grid_labels=True, extrapolate=False, extrapolate_type='none', base_size=5., time_idx=0,
                     coastline_option = 2, no_annotation=False, sigma=False,
                    # no_coastline=False,
                     basic_arithmetic=None, basic_unit=None,
                     view_x_range=None, view_y_range=None,
                     vector=False, vector_skip=10, vector_color='black',  
                     vector_width=0.003, vector_scale=15, vector_ref=-1,
                     contour_cint=None, contour_color='black', contour_linewidth=0.6, contour_extrapolate_type='none'):
    """
    繪製WRF變數
    返回:
        tuple: (fig, ax, output_path) - matplotlib圖形對象、軸對象和已使用的輸出路徑
    """
    print(f"INPUT FILE(s): {ifn}")
    print(f"VAR_shaded: {var_plot}")

    # 解析經緯度格線間距
    try:
        lon_interval, lat_interval = map(float, gridlines.split(','))
    except:
        print(f"警告: 無法解析格線間距 '{gridlines}'，使用預設值10,10")
        lon_interval, lat_interval = 10.0, 10.0

    # 打開WRF NetCDF檔案
    try:
        ncfile = Dataset(ifn)
    except Exception as e:
        print(f"錯誤: 無法打開檔案 {ifn}")
        print(f"錯誤信息: {str(e)}")
        sys.exit(1)

    # 從 WRF 檔案獲取投影參數
   # cart_proj = get_proj_params(ncfile)

    # 讀取主變數
    try:
        VAR_a = getvar(ncfile, var_plot, timeidx=time_idx)
    except Exception as e:
        print(f"錯誤: 無法讀取變數 {var_plot}")
        print(f"錯誤信息: {str(e)}")
        sys.exit(1)
        
    # 讀取等值線變數 (如果指定)
    VAR2_a = None
    if contour is not None:
        try:
            print(f"VAR_contour: {contour}")
            VAR2_a = getvar(ncfile, contour, timeidx=time_idx)
            var2_units = getattr(VAR2_a, 'units', 'Unknown')
            #print(var2_units)
        except Exception as e:
            print(f"警告: 無法讀取等值線變數 {contour}: {str(e)}")
            print("等值線將不會被繪製")
            contour = None

    # 初始化風場變數和等值線變數
    u = v = None
    VAR2 = None
    VAR2_2D = None

    # 處理氣壓層內插或sigma層選取
    if level_p == -9999:  # 地表變數
        level_title = "surface"
        VAR = VAR_a
        print(f"繪製地表變數: {var_plot}")
        
        # 處理等值線變數 (如果有)
        if VAR2_a is not None:
            VAR2 = VAR2_a
            print(f"繪製地表等值線變數: {contour}")
        
        # 讀取地表風場資料 (如果需要)
        if vector:
            try:
                print("讀取地表風場資料 (wspd_wdir10)")
                wind_data = getvar(ncfile, "wspd_wdir10", timeidx=time_idx)
                wspd = wind_data[0]  # 風速
                wdir = wind_data[1]  # 風向
                
                # 轉換為U和V分量
                u, v = wind_to_uv(wspd, wdir)
            except Exception as e:
                print(f"警告: 無法讀取地表風場資料: {str(e)}")
                print(f"風場向量將不會被繪製")
    elif sigma:  # sigma層
        try:
            # 確認level_p是整數
            level_idx = int(level_p)
            if level_idx < 0 or level_idx >= VAR_a.shape[0]:
                raise ValueError(f"Sigma層索引 {level_idx} 超出範圍 [0, {VAR_a.shape[0]-1}]")

            print(f"選取sigma層: {level_idx}")
            VAR = VAR_a[level_idx]  # 直接選取指定的sigma層
            level_title = f"Level {level_idx}"
            
            # 處理等值線變數 (如果有)
            if VAR2_a is not None:
                VAR2 = VAR2_a[level_idx]
                print(f"選取sigma層等值線變數: {contour}, 層級 {level_idx}")
            
            # 讀取sigma層風場資料 (如果需要)
            if vector:
                try:
                    print(f"讀取sigma層風場資料 (wspd_wdir): 層級 {level_idx}")
                    ua = getvar(ncfile, "ua", timeidx=time_idx)[level_idx]
                    va = getvar(ncfile, "va", timeidx=time_idx)[level_idx]
                    u = to_np(ua.squeeze()) 
                    v = to_np(va.squeeze()) 
                except Exception as e:
                    print(f"警告: 無法讀取sigma層風場資料: {str(e)}")
                    print(f"風場向量將不會被繪製")
        except Exception as e:
            print(f"錯誤: 無法選取sigma層 {level_p}")
            print(f"錯誤信息: {str(e)}")
            sys.exit(1)
    else:  # 氣壓層
        try:
            print(f"內插到氣壓層: {level_p} hPa")
            VAR = vinterp(ncfile, VAR_a, "pressure", [level_p], extrapolate=extrapolate, log_p=True, field_type=extrapolate_type)
            level_title = f"{level_p} hPa"
            print(f"外插設定: {extrapolate}, {extrapolate_type}")
            
            # 處理等值線變數 (如果有)
            if VAR2_a is not None:
                VAR2 = vinterp(ncfile, VAR2_a, "pressure", [level_p], extrapolate=extrapolate, log_p=True, field_type=contour_extrapolate_type)
                print(f"等值線變數到氣壓層: {contour}, {level_p} hPa")
                print(f"等值線外插設定: {extrapolate}, {contour_extrapolate_type}")
            
            # 讀取氣壓層風場資料 (如果需要)
            if vector:
                try:
                    print(f"讀取氣壓層風場資料: 層級 {level_p} hPa")
                    # 直接使用ua, va
                    ua = getvar(ncfile, "ua", timeidx=time_idx)
                    ua = vinterp(ncfile, ua, "pressure", [level_p], extrapolate=extrapolate, log_p=True, field_type='pressure')
                    u = to_np(ua.squeeze()) 
                    va = getvar(ncfile, "va", timeidx=time_idx)
                    va = vinterp(ncfile, va, "pressure", [level_p], extrapolate=extrapolate, log_p=True, field_type='pressure')
                    v = to_np(va.squeeze())
                except Exception as e:
                    print(f"警告: 無法讀取氣壓層風場資料: {str(e)}")
                    print(f"風場向量將不會被繪製")
        except Exception as e:
            print(f"錯誤: 無法內插到氣壓層 {level_p} hPa")
            print(f"錯誤信息: {str(e)}")
            sys.exit(1)

    # Get the cartopy projection object
    cart_proj = get_cartopy(VAR)

 
    # u, v to 掩碼陣列
    if vector:
        u = np.ma.array(u)
        v = np.ma.array(v)

    # 獲取地形遮罩（用於繪製海岸線）
    landmask = getvar(ncfile, "LANDMASK", timeidx=time_idx)

    # 獲取經緯度座標
    lats, lons = latlon_coords(VAR)

    # 獲取2D數據
    VAR_2D = VAR.squeeze()  # 移除長度為1的維度
    if VAR2 is not None:    # 處理等值線變數的2D數據 (如果有)
        VAR2_2D = VAR2.squeeze()

    # ------- 裁切矩陣 START -------
    # 處理視圖範圍   
    view_x = None
    if view_x_range:
        try:
            x1, x2 = map(int, view_x_range.split(','))
            if 0 <= x1 < x2 < VAR_2D.shape[1]:
                view_x = (x1, x2)
            else:
                print(f"警告: X軸範圍 {view_x_range} 超出有效範圍 [0, {VAR_2D.shape[1]-1}], 使用全部範圍")
        except Exception as e:
            print(f"警告: 無法解析X軸範圍 '{view_x_range}': {e}")
    
    view_y = None
    if view_y_range:
        try:
            y1, y2 = map(int, view_y_range.split(','))
            if 0 <= y1 < y2 < VAR_2D.shape[0]:
                view_y = (y1, y2)
            else:
                print(f"警告: Y軸範圍 {view_y_range} 超出有效範圍 [0, {VAR_2D.shape[0]-1}], 使用全部範圍")
        except Exception as e:
            print(f"警告: 無法解析Y軸範圍 '{view_y_range}': {e}")
    
    # 裁剪資料範圍
    print(f"ori VAR_2D shape, ndims: {VAR_2D.shape}, {VAR_2D.ndim}")
    orig_index = None    # 先設沒有裁切
    if view_x is not None or view_y is not None:
        x_start = view_x[0] if view_x is not None else 0
        x_end = view_x[1] if view_x is not None else VAR_2D.shape[1]
        y_start = view_y[0] if view_y is not None else 0
        y_end = view_y[1] if view_y is not None else VAR_2D.shape[0]
        
        # 裁剪主要變數資料
        VAR_2D = VAR_2D[y_start:y_end, x_start:x_end]
        lons = lons[y_start:y_end, x_start:x_end]
        lats = lats[y_start:y_end, x_start:x_end]
        landmask = landmask[y_start:y_end, x_start:x_end]
        if VAR2 is not None: 
            VAR2_2D = VAR2_2D[y_start:y_end, x_start:x_end]
        
        # 同時裁剪風場資料 (如果存在)
        if u is not None and v is not None:
            u = u[y_start:y_end, x_start:x_end]
            v = v[y_start:y_end, x_start:x_end]
        
        print(f"已裁剪資料範圍至: y={y_start}:{y_end}; x={x_start}:{x_end}")

        # 記錄原始索引範圍用於後續設置刻度
        orig_index = {
            'x_start': x_start,
            'x_end': x_end,
            'y_start': y_start,
            'y_end': y_end
        }
    # ------- 裁切矩陣 END -------
   
    # Get the basemap object
    bm = get_basemap(VAR_2D)
    print(bm)

    # 轉換為NumPy數組
    lats_np = to_np(lats)
    lons_np = to_np(lons)
    landmask_np = to_np(landmask)
    VAR_2D = np.ma.array(to_np(VAR_2D))   # 轉換為掩碼陣列
    if VAR2 is not None:
        VAR2_2D = np.ma.array(to_np(VAR2_2D))   # 轉換為掩碼陣列

    # 應用基礎運算, give new units
    original_units = VAR.units if hasattr(VAR, 'units') else 'Unknown'
    print(f"original_units = {original_units}")
    if basic_arithmetic:
        try:
            orig_shape = VAR_2D.shape   # 使用安全的eval方法進行數值運算
            x = VAR_2D.flatten()        # 創建臨時變數x來進行運算
            result = eval(f"x{basic_arithmetic}")    # 計算結果
            VAR_2D = result.reshape(orig_shape)      # 恢復原始形狀

            print(f"已應用基礎運算: VAR_2D{basic_arithmetic}")

            # 設定新的單位 (如果提供)
            if basic_unit:
                VAR_2D.units = basic_unit
                print(f"單位_shd已從 '{original_units}' 更新為 '{basic_unit}'")
        except Exception as e:
            print(f"警告: 應用基礎運算時出錯: {str(e)}")
    else:
        # use original units
       #print("VAR_2D 類型:", type(VAR_2D))
       #print("VAR_2D 形狀:", VAR_2D.shape if hasattr(VAR_2D, 'shape') else "無形狀")
       #print("VAR_2D 屬性:", dir(VAR_2D))
        VAR_2D.units = original_units
        print(f"單位_shd: ['{VAR_2D.units if hasattr(VAR_2D, 'units') else 'Unknown'}']")

    if VAR2 is not None:
        print(f"單位_cnt: ['{var2_units}']")
   

    # 計算合適的圖形大小 (基於數據網格尺寸)
    rows, cols = VAR_2D.shape
    aspect_ratio = cols / rows  # 計算長寬比

    # 設定合適的圖形尺寸，考慮到colorbar的空間
    max_size = 15.0  # 最大尺寸
    min_size = 3.0   # 最小尺寸

    if aspect_ratio > 1:
        # 寬大於高的情況
        width = min(base_size * aspect_ratio, max_size)
        height = width / aspect_ratio
        height = max(height, min_size)
    else:
        # 高大於寬的情況
        height = min(base_size / aspect_ratio, max_size)
        width = height * aspect_ratio
        width = max(width, min_size)

    width += base_size * 0.25
    figsize = (width, height)
    print(f"網格尺寸: {rows}x{cols}, 長寬比: {aspect_ratio:.2f}")
    print(f"圖形尺寸: {figsize[0]:.1f}x{figsize[1]:.1f} 英寸")

    # 計算統計量，處理NaN值
    VAR_valid = VAR_2D[~np.isnan(VAR_2D)]
    if len(VAR_valid) > 0:
        data_min = np.min(VAR_valid)
        data_max = np.max(VAR_valid)
        data_mean = np.mean(VAR_valid)
        data_std = np.std(VAR_valid)

        # 計算四分位數 Q1, Q2, Q3
        q1 = np.percentile(VAR_valid, 25)
        q2 = np.percentile(VAR_valid, 50)  # 同median
        q3 = np.percentile(VAR_valid, 75)

        # 改成三行輸出，排版更清晰
        print(f"[Info] VAR_2D statistics")
        print(f"    mean={data_mean:.2f}, std={data_std:.2f}")
        print(f"    min={data_min:.2f}, max={data_max:.2f}")
        print(f"    Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
    else:
        print("警告: 數據全為NaN，無法計算統計量")
        data_min = data_max = data_mean = data_std = q1 = q2 = q3 = np.nan
        
    # 如果有等值線變數，計算其統計量

    if VAR2_2D is not None:
        VAR2_valid = VAR2_2D[~np.isnan(VAR2_2D)]
        if len(VAR2_valid) > 0:
            data2_min = np.min(VAR2_valid)
            data2_max = np.max(VAR2_valid)
            data2_mean = np.mean(VAR2_valid)
            data2_std = np.std(VAR2_valid)
            data2_q1 = np.percentile(VAR2_valid, 25)
            data2_q2 = np.percentile(VAR2_valid, 50)  # 同median
            data2_q3 = np.percentile(VAR2_valid, 75)
            
            # 計算等值線的合適間距 (如果未指定)
            if contour_cint is None:
                # 利用數據範圍自動計算間距，使得有10-15條等值線
                data_range = data2_max - data2_min
                contour_cint = data_range / 10  # 默認10條等值線
                # 四捨五入到最接近的易懂數值
                magnitude = 10 ** np.floor(np.log10(contour_cint))
                contour_cint = np.round(contour_cint / magnitude) * magnitude
            
            print(f"[Info] VAR2_2D (等值線變數) statistics")
            print(f"    mean={data2_mean:.2f}, std={data2_std:.2f}")
            print(f"    min={data2_min:.2f}, max={data2_max:.2f}")
            print(f"    Q1={data2_q1:.2f}, Q2={data2_q2:.2f}, Q3={data2_q3:.2f}")
            print(f"    等值線間距: {contour_cint}")
        else:
            print("警告: 等值線數據全為NaN，無法計算統計量")
            contour = None  # 禁用等值線繪製

    # 處理自定義色階
    if level_contours:
        try:
            levels = [float(x) for x in level_contours.split(',')]
            print(f"使用自定義色階: {levels}")
        except:
            print(f"錯誤: 無法解析自定義色階 '{level_contours}'，使用預設色階")
            levels = num_levels
    else:
        levels = num_levels

    # 獲取時間信息
    time_var = getvar(ncfile, "times", timeidx=time_idx)
    time_str = str(time_var.values) if time_var is not None else ""

    # 格式化時間字串，只顯示年月日時分
    if time_str:
        try:
            # 修正時間格式化問題
            if isinstance(time_var.values, np.datetime64):
                time_str = str(time_var.values)[:16]  # 截取到分鐘部分
            else:
                dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            print(f"格式化時間: {time_str}")
        except Exception as e:
            print(f"警告: 時間格式化失敗: {str(e)}")
            print(f"使用原始時間字串: {time_str}")

    # 獲取文件名（不包含路徑）
    file_basename = os.path.basename(ifn)

   #print(cart_roj)
    # 創建圖形和軸
    background_color = 'gray'
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)  # 創建軸對象
    bm.ax = ax  # 確保Basemap知道要在哪個軸上繪圖
   #fig = plt.figure(figsize=figsize)
   #ax = fig.add_subplot(1, 1, 1, projection=cart_proj)
    fig.subplots_adjust(bottom=0.10, left=0.10, top=0.80) 
    ax.set_facecolor(background_color)  # 設置軸域的背景色
    print(f"創建了圖形: {fig}")

    # 添加經緯網格
    '''
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                     linewidth=0.3, color='gray', alpha=0.5, linestyle='--',
                     x_inline=False, y_inline=False)
    '''

    # Convert the lats and lons to x and y.  Make sure you convert the lats and
    # lons to numpy arrays via to_np, or basemap crashes with an undefined
    # RuntimeError.
    x_bm, y_bm = bm(lons_np, lats_np)
    #print(x_bm, y_bm)

    # 繪製經緯度格線
    if gridlines:
        try:
            # 計算經緯度格線的間隔
            lon_min = np.floor(np.min(lons_np) / lon_interval) * lon_interval
            lon_max = np.ceil(np.max(lons_np) / lon_interval) * lon_interval
            lat_min = np.floor(np.min(lats_np) / lat_interval) * lat_interval
            lat_max = np.ceil(np.max(lats_np) / lat_interval) * lat_interval

            lon_levels = np.arange(lon_min, lon_max + lon_interval, lon_interval)
            lat_levels = np.arange(lat_min, lat_max + lat_interval, lat_interval)

            # 繪製經緯度格線
            lon_contour = bm.contour(x_bm, y_bm, lons_np, levels=lon_levels,
                                    colors='gray', linewidths=0.4, alpha=0.8, zorder=-1)
            lat_contour = bm.contour(x_bm, y_bm, lats_np, levels=lat_levels,
                                    colors='gray', linewidths=0.4, alpha=0.8, zorder=-1)

            # 根據grid_labels參數決定是否添加經緯度標籤
            if grid_labels:
                ax.clabel(lon_contour, inline=1, fontsize=6, fmt='%g')
                ax.clabel(lat_contour, inline=1, fontsize=6, fmt='%g')
                print("已添加經緯度標籤")
            else:
                print("已隱藏經緯度標籤")

        except Exception as e:
            print(f"警告: 繪製網格線時出錯: {str(e)}")
    

    # 設置colormap
    print(f"colormap= {colormap}")
    cmap = colormaps.get_cmap(colormap)

    # 增加統計NaN值數量
    nan_count = np.count_nonzero(np.isnan(VAR_2D))
    nan_percent = nan_count / (rows * cols) * 100
    print(f"NaN值統計: {nan_count} ({nan_percent:.2f}% of total)")

    print(f"coastline_option = {coastline_option}")
    if coastline_option == 0:
        # 不繪製海岸線
        pass
    elif coastline_option == 1:
        # 使用landmask繪製海岸線
        try:
            if landmask_np is None:
                raise ValueError("使用landmask選項時必須提供landmask_np參數")
            ax.contour(x_bm, y_bm, landmask_np, levels=[0.9], colors='black', linewidths=2.0, zorder=-3)
            ax.contour(x_bm, y_bm, landmask_np, levels=[0.9], colors='yellow', linewidths=0.4, zorder=-2)
        except Exception as e:
            print(f"警告: 使用landmask繪製海岸線時出錯: {str(e)}")
    elif coastline_option == 2:
        # 使用basemap函數繪製
        try:
            bm.drawcoastlines(linewidth=2.0, color='black', zorder=-3)
            bm.drawcoastlines(linewidth=0.4, color='yellow', zorder=-2)
           # bm.drawstates(linewidth=0.6, color='gray')
           # bm.drawcountries(linewidth=0.6, color='gray')
        except Exception as e:
            print(f"警告: 使用basemap繪製海岸線時出錯: {str(e)}")
    else:
        print(f"警告: 無效的海岸線選項 {coastline_option}")

    # 繪製等值線填色
    try:
        cf = bm.contourf(x_bm, y_bm, VAR_2D, levels=levels, cmap=cmap, extend='both', zorder=-9)
    except Exception as e:
        print(f"警告: 繪製shaded時出錯: {str(e)}")
        try:
            print("嘗試使用自動生成的色階範圍...")
            cf = bm.contourf(x_bm, y_bm, VAR_2D, cmap=cmap, zorder=-9)
        except:
            print("無法繪製填色等值線，數據可能全為NaN")
            plt.close()
            ncfile.close()
            return fig, ax, output  # 返回即使失敗也需要返回對象

    # 添加顏色棒
    try:
        # 創建 colorbar 的新位置
        cbar = plt.colorbar(cf, ax=ax, pad=0.01, shrink=0.5, location='right', extend='both', orientation='vertical', aspect=20, drawedges=False)
        cbar.ax.set_position([0.80, 0.12, 0.03, 0.45])  # 調整這四個參數來改變位置and size  [xini, yini, 寬度, 高度]
        if hasattr(VAR_2D, 'units'):
            cbar.set_label(VAR_2D.units, fontsize=8, loc='center')
        else:
            cbar.set_label('Unknown units', fontsize=8, loc='center')
    except Exception as e:
        print(f"警告: 添加色標時出錯: {str(e)}")

    # 繪製等值線 (如果指定了變數)
    if contour is not None and VAR2_2D is not None:
        try:
            print(f"繪製等值線: {contour}")
            
            # 根據間距生成等值線級別
            if contour_cint is not None:
                vmin = np.floor(data2_min / contour_cint) * contour_cint
                vmax = np.ceil(data2_max / contour_cint) * contour_cint
                ct_levels = np.arange(vmin, vmax + contour_cint, contour_cint)
            else:
                # 使用默認10個等值線
                ct_levels = 10
                
            ct = bm.contour(x_bm, y_bm, VAR2_2D, levels=ct_levels,
                         colors=contour_color, linewidths=contour_linewidth, alpha=1, zorder=3)
            if len(ct_levels) <= 30:     # 只有當等值線不太密集時才添加標籤
                ax.clabel(ct, inline=1, fontsize=9.5, fmt='%g', zorder=3)

            # 添加等值線變數統計信息在right bottom
            if not no_annotation and not np.isnan(data2_mean):
                cstats_text1 = f"{contour}: mean={data2_mean:.2f}, std={data2_std:.2f}"
                cstats_text2 = f"  min={data2_min:.2f}, max={data2_max:.2f}"
                cstats_text3 = f"  cint={contour_cint} {var2_units}"
       
                # 創建文字框，放在左下角但稍微向上偏移以避免與主變數統計框重疊
                ax.text(0.99, 0.01, cstats_text1 + "\n" + cstats_text2 + "\n" + cstats_text3,
                       horizontalalignment='right',
                       verticalalignment='bottom',
                       transform=ax.transAxes,
                       zorder=9,
                       fontsize=7, color=contour_color,
                       bbox=dict(facecolor='white', alpha=1, pad=2, edgecolor=contour_color, linewidth=0.5))
            
            print(f"    等值線繪製完成: 變數={contour}, 間距={contour_cint}, 顏色={contour_color}, 寬度={contour_linewidth}")
        except Exception as e:
            print(f"警告: 繪製等值線時出錯: {str(e)}")
            
    # 繪製風場向量 (如果啟用)
    if vector:
        if u is not None and v is not None:  # 檢查u和v是否有效
            try:
                print(f"繪製winds")
                # 創建網格點坐標
                y, x = np.mgrid[0:VAR_2D.shape[0]:1, 0:VAR_2D.shape[1]:1]
                
                # 根據vector_skip跳過部分格點
                skip = vector_skip
                
                # 決定向量長度
                speed = np.sqrt(u**2 + v**2)
                max_speed = np.max(speed[~np.isnan(speed)])
                mean_speed = np.mean(speed[~np.isnan(speed)])
                if (vector_scale == -1) :
                     scale = max_speed * 15  # 根據最大風速調整比例
                else :                
                     scale = vector_scale

                # 繪製風場向量 with white bdy
                head_v = 3.2
                quiver = bm.quiver(x_bm[::skip, ::skip], y_bm[::skip, ::skip],
                                 u[::skip, ::skip], v[::skip, ::skip],
                                 zorder=1 ,
                                 color=vector_color, scale=scale, width=vector_width,  pivot='tail',
                                 headwidth=head_v, headlength=head_v*1.5,  # 控制箭頭頭部大小
                                 edgecolor='white', linewidth=0.8)  # 添加白色邊框
                
                # 計算參考風速 (取整)
                if vector_ref == -1 : 
                    ref_speed = max_speed 
                else :
                    ref_speed = vector_ref

                ref_speed = round(ref_speed, 0)
             #  print(f"max_speed, mean_speed, ref_speed = {max_speed:.2f}, {mean_speed:.2f}, {ref_speed}")

                # 添加參考箭頭
                qk = ax.quiverkey(quiver, 1.08, 0.97, ref_speed, f'{ref_speed}\nm s⁻¹', 
                               labelpos='S', coordinates='axes', color=vector_color, labelcolor=vector_color,
                               fontproperties={'size': 7})  # 調整字體大小以匹配您的文字框
                
                print(f"    向量: skip(-vs)={skip}, color(-vc)={vector_color}, width(-vw)={vector_width}, ")
                print(f"          縮小比例(-vl)={scale}, 參考風速(-vf)={ref_speed}")
                print(f"          最大風速={max_speed:.1f}, 平均風速={mean_speed:.1f}")
            except Exception as e:
                print(f"警告: 繪製風場向量時出錯: {str(e)}")
        else:
            print("風場數據不可用，跳過風場向量繪製")


    # 建立完整的標題
    long_name = getattr(VAR, 'description', var_plot) if hasattr(VAR, 'description') else var_plot

    # 檔案名放在標題上方，字體較小
    ax.text(1, 1.2, f"{ifn}", ha='right', va='top', transform=ax.transAxes, fontsize=6, alpha=1.0)
   # ax.text(1, 1.17, f"cint= {contour_cint}", ha='left', va='top', transform=ax.transAxes, fontsize=8, alpha=1.0, color=contour_color)


    # 標題中添加基礎運算信息 (如果有)
    if basic_arithmetic == None:
      title_plus = " "
    else:
      title_plus = f"{basic_arithmetic} "

    # 時間放在標題第一行，簡化顯示
    title_text = f"{time_str}\n{var_plot}{title_plus}at {level_title}" if time_str else f"{var_plot}{title_plus} at {level_title}"
    
    # 如果繪製了等值線，在標題中添加等值線信息
    if contour is not None and VAR2_2D is not None:
        title_text += f" with {contour}(cnt) "
    
    # 如果繪製了風場向量，在標題中添加風場信息
    if vector and u is not None and v is not None:
        if contour is not None and VAR2_2D is not None:
            title_text += " and winds(vct)"
        else:
            title_text += " with winds(vct)"
        
    plt.title(title_text, fontsize=12, pad=2.1)

    # 添加統計信息在圖的left bottom
    if not no_annotation and not np.isnan(data_mean):
        # 改為三行顯示，使統計數據更清晰
        stats_text1 = f"{var_plot}: mean={data_mean:.2f}, std={data_std:.2f}"
        stats_text2 = f"  min={data_min:.2f}, max={data_max:.2f}"
        stats_text3 = f"  Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}"

        # 創建文字框，放在left bottom
        ax.text(0.01, 0.01, stats_text1 + "\n" + stats_text2 + "\n" + stats_text3,
               horizontalalignment='left',
               verticalalignment='bottom',
               transform=ax.transAxes,
               fontsize=7, alpha=1.0,
               zorder=9,
               bbox=dict(facecolor='white', alpha=1.0, pad=2, linewidth=0.5))

    # 添加邊界框
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)  # 設置邊框寬度，數值越大越粗

    # 如果有裁剪過數據，重新設置刻度和標籤以顯示原始索引
    if orig_index is not None:
        # 獲取原始索引範圍
        x_start, x_end = orig_index['x_start'], orig_index['x_end']
        y_start, y_end = orig_index['y_start'], orig_index['y_end']
    else :
        x_start = 0
        _, x_end = VAR_2D.shape
        x_end = x_end - 1 
        y_start = 0
        y_end, _ = VAR_2D.shape
        y_end = y_end - 1 
        
    # 計算適當的刻度間隔(基於實際數據尺寸)
    rows, cols = VAR_2D.shape
        
    # 設置X軸刻度
    # 根據裁剪後矩陣的寬度自動計算刻度數量(最多5個)
    x_tick_count = min(cols, 5)
    # x_positions = np.linspace(0, cols-1, x_tick_count)
    x_positions = np.linspace(0, np.max(x_bm[0,:])-np.min(x_bm[0,:])-1, x_tick_count)
    ax.set_xticks(x_positions)
    x_tick_labels = [str(int(x_start + i*(x_end-x_start)/(x_tick_count-1))) for i in range(x_tick_count)]
    ax.set_xticklabels(x_tick_labels)
        
    # 設置Y軸刻度
    y_tick_count = min(rows, 5)
    y_positions = np.linspace(0, np.max(y_bm[:,0])-np.min(y_bm[:,0])-1, y_tick_count)
    ax.set_yticks(y_positions)
    # 刻度標籤顯示原始索引
    y_tick_labels = [str(int(y_start + i*(y_end-y_start)/(y_tick_count-1))) for i in range(y_tick_count)]
    ax.set_yticklabels(y_tick_labels)

    # 根據不同層次生成輸出文件名
    output_path = output
    if output:
        # 處理輸出檔名，加入層次信息
        base, ext = os.path.splitext(output)
        if level_title != "surface":
            level_str = level_title.replace(" ", "_")
            output_path = f"{base}_{level_str}{ext}"
            
        # 如果繪製了風場向量，在檔名中添加標識
        if vector and u is not None and v is not None and "_wind" not in output_path:
            output_path = output_path.replace(ext, f"_wind{ext}")
            
        # 如果繪製了等值線，在檔名中添加標識
        if contour is not None and VAR2_2D is not None and f"_cntr{contour}" not in output_path:
            output_path = output_path.replace(ext, f"_cntr{contour}{ext}")
        
        # 處理輸出目錄
        if output_dir:
            # 確保輸出目錄存在
            os.makedirs(output_dir, exist_ok=True)
            # 僅取輸出文件名稱，不含路徑
            output_filename = os.path.basename(output_path)
            # 合併目錄和文件名
            output_path = os.path.join(output_dir, output_filename)
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖檔已保存: {output_path}")

    # 返回圖形和軸對象以及輸出路徑，而不關閉
    print(f"返回圖形對象和軸對象: {fig}, {ax}")

    # 關閉NetCDF文件
    ncfile.close()

    return fig, ax, output_path

#------------------------------------
def parse_levels(level_str):
    """解析層次字串，返回層次列表"""
    if level_str == '-9999':
        return [-9999]  # 地表層

    try:
        return [float(level) for level in level_str.split(',')]
    except ValueError:
        print(f"警告: 無法解析層次 '{level_str}'，使用預設值 -9999")
        return [-9999]

#------------------------------------
def main(custom_args=None):
    """主程序"""
    # 解析命令列參數
    args = parse_arguments(custom_args)

    # 檢查必要參數
    if not args.input:
        print("錯誤: 必須提供輸入檔案 (-i)")
        sys.exit(1)

    # 解析層次列表
    levels = parse_levels(args.level)
    print(f"處理層次: {levels}")

    # 存儲所有圖形對象，以便在互動模式下顯示
    figures = []

    # 多層次處理
    for idx, level in enumerate(levels):
        print(f"\n--------- 處理第 {idx+1}/{len(levels)} 層: {level} ---------")

        # 繪製變數
        fig, ax, output_path = plot_wrf_variable(
            ifn       = args.input,
            var_plot  = args.variable,
            level_p   = level,
            output    = args.output,
            output_dir = args.output_dir,  # 添加輸出目錄參數
            colormap  = args.colormap,
            num_levels     = args.num_levels,
            level_contours = args.level_contours,
            contour        = args.contour,      # 修改為變數名，不再是布爾值
            gridlines      = args.gridlines,
            grid_labels    = args.grid_labels,
            extrapolate    = args.extrapolate,
            extrapolate_type = args.extrapolate_type,
            base_size      = args.base_size,
            time_idx       = args.time_idx,
            coastline_option = args.coastline_option,
            no_annotation  = args.no_annotation,
            sigma          = args.sigma,
            basic_arithmetic = args.basic_arithmetic,
            basic_unit      = args.basic_unit, 
            view_x_range    = args.view_x_range,
            view_y_range    = args.view_y_range,
            vector          = args.vector,
            vector_skip     = args.vector_skip,
            vector_color    = args.vector_color,
            vector_width    = args.vector_width,
            vector_scale    = args.vector_scale,
            vector_ref      = args.vector_ref,
            contour_cint    = args.contour_cint,    # 等值線間距參數
            contour_color   = args.contour_color,   # 等值線顏色參數
            contour_linewidth = args.contour_linewidth,  # 等值線線寬參數
            contour_extrapolate_type = args.contour_extrapolate_type  # 等值線外插時用的file_type
        )
        figures.append((fig, output_path))     # 保存圖形對象

    # 如果沒有指定輸出檔案，顯示所有圖形
    if not args.output:
        print(f"\n顯示 {len(figures)} 個圖形...")
        plt.show()

    # 關閉所有圖形
    for fig, _ in figures:
        plt.close(fig)

    print(f"\n已處理 {len(levels)} 個層次: {levels}")

if __name__ == "__main__":
    main()
    print(f"\n======= RUN END: {args_str} =========\n")
