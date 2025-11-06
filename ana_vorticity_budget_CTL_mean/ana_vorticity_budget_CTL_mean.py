#!/usr/bin/env python3
#======================================================================================================================
"""
相對渦度收支分析程式
Relative Vorticity Budget Analysis ana_vorticity_budget_CTL_mean.py

此程式用於計算與分析WRF模式輸出的相對渦度收支方程各項，包括：
- 局地變化項 (∂ζ/∂t)
- 水平平流項 (horizontal advection)
- 垂直平流項 (vertical advection)
- 輻散效應項 (divergence effect)
- 傾斜項 (tilting term)
- 殘差項 (residual)
"""
#======================================================================================================================
# import
#======================================================================================================================
import netCDF4 as nc
import numpy as np
import xarray as xr
import pint_xarray as px
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import uniform_filter
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker    
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker     
import pandas as pd
import wrf
import os
import pyproj
import argparse
from datetime import datetime
from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim)
from definitions.plot_2D_shaded import plot_2D_shaded as p2d
from definitions.def_show_array_info import array_info as array_info
from definitions.def_quantity_to_xarray import quantity_to_xarray as quantity_to_xarray

#======================================================================================================================

def spatial_smooth(data, n=33, fill_method="linear"):
    """
    使用 metpy.calc.smooth_gaussian 進行高斯平滑
    此函數會對最後兩個維度（y, x）進行平滑處理
    
    Parameters:
    -----------
    data : pint.Quantity or xarray.DataArray
        需要平滑的資料，可以是多維的
    n : int
        平滑程度參數。數值越大，平滑越強
        n 是從波峰到波峰的網格點數，其理論響應為 1/e
        
    Returns:
    --------
    smoothed : 與輸入相同類型平滑後的資料
               返回資料會變成無單位，需手動補回
    """
    # 記錄原始的 NaN 位置
    original_nan_mask = np.isnan(data)

    # 先補值
    y_dim = data.dims[-2]   # south_north
    x_dim = data.dims[-1]   # west_east
    data_filledyx = data.interpolate_na(dim=y_dim, method=fill_method).interpolate_na(dim=x_dim, method=fill_method)
    data_filledxy = data.interpolate_na(dim=x_dim, method=fill_method).interpolate_na(dim=y_dim, method=fill_method)
    data_filled = (data_filledyx + data_filledxy) / 2

    # 確保沒有殘餘 NaN
    data_filled = data_filled.fillna(0)

    # 使用 metpy 的高斯平滑函數 這個函數只對最後兩個維度進行平滑
    smoothed = mpcalc.smooth_gaussian(data_filled, n)
    
    # 恢復原始的 NaN 位置
    smoothed_nan = smoothed.where(~original_nan_mask)

    return smoothed_nan

#======================================================================================================================
# ======================================
# 設定分析參數與命令列參數解析
# ======================================
parser = argparse.ArgumentParser(
    description='相對渦度收支分析程式 - Relative Vorticity Budget Analysis',
    formatter_class=argparse.RawTextHelpFormatter,
    epilog="""
Run sample:
    python3 ana_vorticity_budget_CTL_mean.py -L 850 -T 2006-06-10T00
    python3 ana_vorticity_budget_CTL_mean.py -L 700 -T 2006-06-11T12 -n ./output/ana_vorticity_budget
    
Author: CYC (YakultSmoothie)
Create date: 2025-10-02
"""
)

parser.add_argument('-L', '--level', type=float, default=850.0,
                   help='目標氣壓層 (hPa), 預設: 850.0')
parser.add_argument('-T', '--time', type=str, default='2006-06-10T00',
                   help='目標分析時間 (格式: YYYY-MM-DDTHH), 預設: 2006-06-10T00')
parser.add_argument('-n', '--new_dir', type=str, default='./output_ana_vorticity_budget',
                   help='輸出路徑, 預設: ./output_ana_vorticity_budget')
parser.add_argument('-d', '--domain', type=str, default='d03',
                   help='輸出路徑, 預設: ./output_ana_vorticity_budget')

args = parser.parse_args()

# region - 設定分析參數
target_pressure = args.level                        # 目標氣壓層 (hPa)
target_time = pd.to_datetime(args.time)             # 目標分析時間
# target_ensemble = range(1, 65)                    # 目標集成成員編號
target_ensemble = [1]                               # 目標集成成員編號
time_interval = 1                                   # 時間間隔 (小時)，用於計算時間導數
target_domain = args.domain                         # 處理斗面
output_dir = args.new_dir                           # 輸出目錄
# endregion - 設定分析參數

# 選擇輸入路徑
IFDd = {}
IFDd['d03'] = "/jet/ox/work/MYHPE/1/2025-0909-Frontal_env/w2nc/StepI"
IFDd['d02'] = "/jet/ox/work/MYHPE/1/2025-0909-Frontal_env/w2nc/StepI/d02"
IFD = IFDd[target_domain]

# gird number for smooth
smooth_nd = {}
smooth_nd['d03'] = 33
smooth_nd['d02'] = 7

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

print(f"\n分析參數設定:")
print(f"    目標氣壓層: {target_pressure} hPa")
print(f"    目標時間: {target_time}")
print(f"    輸出目錄: {output_dir}")

# ======================================
# get topo info  WRF座標維度對應設定
# ======================================
# region
ncfile = nc.Dataset(f"/jet/ox/work/MYHPE/1/2024-0415-Hby_ETKF/ETKF/RUN-forecast/WRF-RUN/CTL/wrfin/wrfinput_{target_domain}")

# Get the map projection information
hgt = getvar(ncfile, "HGT")    # 2D xarray DataArray
landmask = getvar(ncfile, "LANDMASK")    # 2D xarray DataArray
array_info(hgt, f"hgt", indent=0)
array_info(landmask, f"landmask", indent=0)
proj = get_cartopy(hgt)

coord_map = {
    "time": "Time",
    "vertical": "interp_level",
    "y": "south_north",
    "x": "west_east"
}

"""
ds_land = xr.open_dataset(f"/jet/ox/work/MYHPE/1/2025-0909-Frontal_env/w2nc/StepI/sfc/{target_domain}.nc").metpy.parse_cf(coordinates=coord_map)
hgt = (ds_land['HGT'] * units.m).squeeze(['member', 'Time', 'interp_level'])  # 移除維度
landmask = ds_land['LANDMASK'].squeeze(['member', 'Time', 'interp_level'])

array_info(hgt, f"hgt", indent=0)
array_info(landmask, f"landmask", indent=0)

# 手動 get 投影
proj = ccrs.LambertConformal(
    central_longitude=ds_land.attrs['projection_standard_longitude'],
    central_latitude=ds_land.attrs['projection_center_latitude'],
    standard_parallels=(
        ds_land.attrs['projection_true_latitude_1'],
        ds_land.attrs['projection_true_latitude_2']
    )
)
"""
# endregion

# ======================================
# 開啟NetCDF資料檔案
# ======================================

# 讀取各氣象變數資料集
# ds_uv = xr.open_dataset(f"{IFD}/ua,va.nc").metpy.parse_cf(coordinates=coord_map)
# ds_w = xr.open_dataset(f"{IFD}/wa.nc").metpy.parse_cf(coordinates=coord_map)
# ds_z = xr.open_dataset(f"{IFD}/height.nc").metpy.parse_cf(coordinates=coord_map)
# ds_t = xr.open_dataset(f"{IFD}/tk.nc").metpy.parse_cf(coordinates=coord_map)
# ds_q = xr.open_dataset(f"{IFD}/QVAPOR.nc").metpy.parse_cf(coordinates=coord_map)
# ds_uvmet = xr.open_dataset(f"{IFD}/uvmet.nc").metpy.parse_cf(coordinates=coord_map)

ds_uv = xr.open_dataset(f"{IFD}/ua,va_ensemble_mean.nc").metpy.parse_cf(coordinates=coord_map)
ds_w = xr.open_dataset(f"{IFD}/wa_ensemble_mean.nc").metpy.parse_cf(coordinates=coord_map)
ds_z = xr.open_dataset(f"{IFD}/height_ensemble_mean.nc").metpy.parse_cf(coordinates=coord_map)
ds_t = xr.open_dataset(f"{IFD}/tk_ensemble_mean.nc").metpy.parse_cf(coordinates=coord_map)
ds_q = xr.open_dataset(f"{IFD}/QVAPOR_ensemble_mean.nc").metpy.parse_cf(coordinates=coord_map)
ds_uvmet = xr.open_dataset(f"{IFD}/uvmet_ensemble_mean.nc").metpy.parse_cf(coordinates=coord_map)

#breakpoint()
# ======================================
# 獲取模式網格資訊
# ======================================
dx = ds_uv.attrs['WRF_DX'] * units.m  # x方向網格間距
dy = ds_uv.attrs['WRF_DY'] * units.m  # y方向網格間距
lons = ds_uv.XLONG * units.deg        # 經度網格
lats = ds_uv.XLAT * units.deg         # 緯度網格

# ======================================
# 選取分析層次與時間
# ======================================
# 找到目標氣壓層索引，並選取上下各一層（共三層）用於垂直導數計算
target_level_index = (ds_uv.interp_level == target_pressure).argmax()
selected_pressure_levels = ds_uv.interp_level.values[[
    target_level_index - 1,
    target_level_index,
    target_level_index + 1
]]

# 建立目標時間序列：前後各一個時間間隔（共三個時刻）用於時間導數計算
selected_times = [
    target_time + pd.Timedelta(hours=-1 * time_interval),  # t-1
    target_time,                                           # t
    target_time + pd.Timedelta(hours=+1 * time_interval)   # t+1
]

print(f"\n分析設定資訊:")
print(f"選定的氣壓層: {selected_pressure_levels} hPa")
print(f"選定的時間: {selected_times}")
print(f"選定的成員: {target_ensemble}")

# ======================================
# 提取氣象變數資料
# ======================================
# 共同選取條件
sel_kwargs = dict(
    interp_level=selected_pressure_levels,
    Time=selected_times,
    member=target_ensemble
)

print(f"\n{'='*80}")
print(f"ds.sel ing...")
print(f"{'='*80}")
# 定義變數配置:資料集、變數名稱、單位、輸出變數名、中文說明
var_configs = [
    (ds_uv, 'ua', 'm/s', 'u_wind'),
    (ds_uv, 'va', 'm/s', 'v_wind'),
    #(ds_uvmet, 'uvmet_u', 'm/s', 'umet'),
    #(ds_uvmet, 'uvmet_v', 'm/s', 'vmet'),
    (ds_w, 'wa', 'm/s', 'w_wind'),
    (ds_z, 'height', 'm', 'gph'),
    (ds_t, 'tk', 'K', 'temperature'),
    (ds_q, 'QVAPOR', 'kg/kg', 'mixing_ratio')
    
]

# 用字典儲存結果
variables = {}

# 迴圈處理每個變數並計算系集平均
for dataset, var_name, unit, output_name in var_configs:
    variables[output_name] = dataset[var_name].sel(**sel_kwargs).mean(dim='member') * units(unit)  # 計算系集平均 at this
    array_info(variables[output_name], f"{output_name} - ensemble mean", indent=0)

# 將變數從字典中取出 dims: (t, p, y, x)  用迴圈將變數從字典中取出並賦值到區域變數
u_wind = variables['u_wind']
v_wind = variables['v_wind']
#umet = variables['umet']
#vmet = variables['vmet']
w_wind = variables['w_wind']
gph = variables['gph']
temperature = variables['temperature']
mixing_ratio = variables['mixing_ratio']

#for var_name in variables:
#    locals()[var_name] = variables[var_name]

# 在計算之前先平滑
# u_wind = spatial_smooth(u_wind, n=smooth_nd[target_domain])
# v_wind = spatial_smooth(v_wind, n=smooth_nd[target_domain])
# w_wind = spatial_smooth(w_wind, n=smooth_nd[target_domain])
# gph = spatial_smooth(gph, n=smooth_nd[target_domain])
# temperature = spatial_smooth(temperature, n=smooth_nd[target_domain])
# mixing_ratio = spatial_smooth(mixing_ratio, n=smooth_nd[target_domain])

# ======================================
# 建立座標軸數值陣列
# ======================================
pressure_coords = u_wind.interp_level.values * 100 * units.Pa  # 氣壓座標
time_coords = (u_wind.Time.values - u_wind.Time.values[0]) / np.timedelta64(1, 's') * units.s  # 時間座標（秒）

# 將一維氣壓座標擴展到與資料相同的形狀（用於計算omega）
pressure_coords_xr = u_wind.interp_level * 100 * units.Pa
pressure_field = pressure_coords_xr.broadcast_like(u_wind)

# ======================================
# 計算診斷變數
# ======================================
print(f"\n{'='*80}")
print('Defing...')
print(f"{'='*80}")

# 垂直速度的氣壓座標形式 (omega = dp/dt)
omega = mpcalc.vertical_velocity_pressure(
    w_wind, pressure_field, temperature, mixing_ratio=mixing_ratio
)

# 相對渦度場（ζ = ∂v/∂x - ∂u/∂y）
relative_vorticity = mpcalc.vorticity(
    u_wind, v_wind, x_dim=-1, y_dim=-2, dx=dx, dy=dy
)

# 水平輻散場（∇·V = ∂u/∂x + ∂v/∂y）
divergence = mpcalc.divergence(
    u_wind, v_wind, x_dim=-1, y_dim=-2, dx=dx, dy=dy
)

# 絕對渦度場（ζ + f）
absolute_vorticity = mpcalc.absolute_vorticity(
    u_wind, v_wind, x_dim=-1, y_dim=-2, dx=dx, dy=dy
)

# ======================================
# 計算空間梯度
# ======================================
# 絕對渦度的水平梯度 (∂(ζ+f)/∂x, ∂(ζ+f)/∂y)
d_abs_vort_dx, d_abs_vort_dy = mpcalc.geospatial_gradient(
    absolute_vorticity, x_dim=-1, y_dim=-2, dx=dx, dy=dy
)

# omega的水平梯度 (∂ω/∂x, ∂ω/∂y)
d_omega_dx, d_omega_dy = mpcalc.geospatial_gradient(
    omega, x_dim=-1, y_dim=-2, dx=dx, dy=dy
)

# 相對渦度的垂直梯度 (∂ζ/∂p)
d_rel_vort_dp = mpcalc.first_derivative(
    relative_vorticity,
    axis=u_wind.get_axis_num('interp_level'),
    x=pressure_coords
)
d_rel_vort_dp = quantity_to_xarray(d_rel_vort_dp, u_wind)  # 轉換回 xarray

# 風場的垂直梯度 (∂u/∂p, ∂v/∂p)
du_dp = mpcalc.first_derivative(
    u_wind,
    axis=u_wind.get_axis_num('interp_level'),
    x=pressure_coords
)
du_dp = quantity_to_xarray(du_dp, u_wind)  # 轉換回 xarray
dv_dp = mpcalc.first_derivative(
    v_wind,
    axis=v_wind.get_axis_num('interp_level'),
    x=pressure_coords
)
dv_dp = quantity_to_xarray(dv_dp, v_wind)  # 轉換回 xarray

# ======================================
# 計算渦度收支方程各項
# ======================================
# 局地變化項 (∂ζ/∂t)
local_tendency = mpcalc.first_derivative(
    relative_vorticity,
    axis=u_wind.get_axis_num('Time'),
    x=time_coords
)
local_tendency = quantity_to_xarray(local_tendency, u_wind, name='local_tendency', description='Tilting term (∂ζ/∂t)')
local_tendency_smoothed = spatial_smooth(local_tendency, n=smooth_nd[target_domain]) * local_tendency.data.units

# 水平絕對渦度平流項 (-V·∇(ζ+f) = -u·∂(ζ+f)/∂x - v·∂(ζ+f)/∂y)
horizontal_advection = (-1 * u_wind * d_abs_vort_dx) + (-1 * v_wind * d_abs_vort_dy)
horizontal_advection_smoothed = spatial_smooth(horizontal_advection, n=smooth_nd[target_domain]) * horizontal_advection.data.units 

# 垂直平流項 (-ω·∂ζ/∂p)
vertical_advection = -1 * omega * d_rel_vort_dp
vertical_advection_smoothed = spatial_smooth(vertical_advection, n=smooth_nd[target_domain]) * vertical_advection.data.units 

# 輻散效應項 (-(ζ+f)∇·V)
divergence_effect = -1 * absolute_vorticity * divergence
divergence_effect_smoothed = spatial_smooth(divergence_effect, n=smooth_nd[target_domain]) * divergence_effect.data.units 

# 傾斜項 (∂u/∂p·∂ω/∂y - ∂v/∂p·∂ω/∂x)
tilting_term = du_dp * d_omega_dy - dv_dp * d_omega_dx
#tilting_term = quantity_to_xarray(tilting_term, u_wind, name='tilting_term', description='(∂u/∂p·∂ω/∂y - ∂v/∂p·∂ω/∂x)')
tilting_term_smoothed = spatial_smooth(tilting_term, n=smooth_nd[target_domain]) * tilting_term.data.units 

# 殘差項（平衡方程誤差）
residual = local_tendency - horizontal_advection - vertical_advection - divergence_effect - tilting_term
residual_smoothed = local_tendency_smoothed - horizontal_advection_smoothed - vertical_advection_smoothed - divergence_effect_smoothed - tilting_term_smoothed

#breakpoint()
# ======================================
# 其他變數平滑
# ======================================
absolute_vorticity_smoothed = spatial_smooth(absolute_vorticity, n=smooth_nd[target_domain]) * absolute_vorticity.data.units 
omega_smoothed = spatial_smooth(omega, n=smooth_nd[target_domain]) * omega.data.units 
divergence_smoothed = spatial_smooth(divergence, n=smooth_nd[target_domain]) * divergence.data.units 
du_dp_smoothed = spatial_smooth(du_dp, n=smooth_nd[target_domain]) * du_dp.data.units 
dv_dp_smoothed = spatial_smooth(dv_dp, n=smooth_nd[target_domain]) * dv_dp.data.units 
d_rel_vort_dp_smoothed = spatial_smooth(d_rel_vort_dp, n=smooth_nd[target_domain]) * d_rel_vort_dp.data.units

# breakpoint()
# ============================================================================
# 繪製渦度收支各項的空間分布
# ============================================================================
# 選取中間時刻與中間氣壓層進行繪圖
# region
print(f"\n{'='*80}")
print(f"Plotting...")
print(f"{'='*80}")

matplotlib.use('Agg')  # 使用非互動式後端

# 創建 2x3 的子圖
fig_bai = 1.0
fig = plt.figure(figsize=(17*fig_bai, 6*fig_bai))
axes = {}

# plot setting
baid = {}
baid['d02'] = 1e9
baid['d03'] = 1e9
bai_set = baid[target_domain]

# Plot切片
extent_dict = {
    'd02': [109, 123, 20, 27],   # [lon_min, lon_max, lat_min, lat_max]
    'd03': None                  # 使用全域範圍
}
level_idx = 1
time_idx = 1
#breakpoint()
# 找出對應的索引範圍
if target_domain == 'd02':
    # 找出最接近目標經緯度的索引
    mask = (lons >= extent_dict[target_domain][0] * units.deg) & \
            (lons <= extent_dict[target_domain][1] * units.deg) & \
            (lats >= extent_dict[target_domain][2] * units.deg) &\
            (lats <= extent_dict[target_domain][3] * units.deg)
    # 獲取索引範圍
    x_indices = np.where(mask.any(axis=0))[0]
    y_indices = np.where(mask.any(axis=1))[0]
    
    x_slice = slice(x_indices[0], x_indices[-1]+1)
    y_slice = slice(y_indices[0], y_indices[-1]+1)
    
    plot_indices = [time_idx, level_idx, y_slice, x_slice]  # 使用對應的切片
else:
    plot_indices = [time_idx, level_idx, slice(None), slice(None)]  # 使用對應的切片

plot_slice = tuple(plot_indices)

# lot_indices = [time_idx, level_idx, slice(None), slice(None)]  # [time_idx, level_idx, y, x]
# plot_slice = tuple(plot_indices)
# max_abs = 20  
# common_levels = np.linspace(-max_abs, max_abs, 11)    # when extend='both' > 自動分成12個clevs


cint = {}
cint['d03'] = 5
cint['d02'] = 5
#ccolor="#1B06E0"
ccolor="magenta"

# 渦度診斷項的視覺化參數設定
# 每個tuple包含：(資料變數, 標題, color 倍率, 
#               向量u, 向量v, 向量間距x, 向量間距y,
#               等值線資料, 等值線間距, 粗等值線間距)
variables = [
    # 局地變化項：顯示相對渦度的時間變化趨勢，疊加風場向量 與 位勢高度
    (local_tendency_smoothed, 'Local Tendency_sm', bai_set, 
     u_wind, v_wind, 120, 25, 
     gph, cint[target_domain], cint[target_domain]*4),
    
    # 水平平流項：顯示風場對相對渦度的水平輸送效應，疊加風場向量 與 絕對渦度等值線
    (horizontal_advection_smoothed, 'Horizontal Advection_sm', bai_set, 
     u_wind, v_wind, 120, 25, 
     absolute_vorticity_smoothed*1e5, 5, 20),
    
    # 垂直平流項：顯示垂直運動對相對渦度的輸送效應，疊加omega等值線
    (vertical_advection_smoothed, 'Vertical Advection_sm', bai_set,
     None, None, None, None, 
     omega_smoothed, 0.25, 1.0),
    
    # 散度效應項：顯示水平散度對渦度的拉伸/壓縮效應，疊加風場向量 與 散度場等值線
    (divergence_effect_smoothed, 'Divergence Effect_sm', bai_set, 
     u_wind, v_wind, 120, 25, 
     divergence_smoothed*1e5, 3, 12),
    
    # 傾斜項：顯示水平渦度的垂直分量轉換為相對渦度的效應，疊加風場的垂直切變向量 與 omega等值線
    (tilting_term_smoothed, 'Tilting Term_sm', bai_set, 
     du_dp_smoothed*1e2, dv_dp_smoothed*1e2, 0.3, 0.07, 
     omega_smoothed, 0.25, 1.0),
    
    # 殘差項：包含未計算的物理過程（如摩擦、次網格過程等），疊加相對渦度的垂直梯度
    (residual_smoothed, 'Residual_sm', bai_set, 
     None, None, None, None, 
     d_rel_vort_dp_smoothed*1e9, 3, 12)
]


# 對每個變數繪圖   
for i, (var, var_name, bai, uuu, vvv, vscale, vref, cnt, cint, cint2) in enumerate(variables):
  
    # 處理向量場參數
    if uuu is not None and vvv is not None:
        vx_plot = uuu[plot_slice]
        vy_plot = vvv[plot_slice]
    else:
        vx_plot = None
        vy_plot = None    #print(i)

    axes[i] = fig.add_subplot(2, 3, i+1, projection=proj)   # 在fig中添加子圖，並決定投影方式
    #axes[i] = fig.add_subplot(2, 3, i+1)   # 在fig中添加子圖，並決定投影方式
    result = p2d(
        var[plot_slice]*bai, 
        x=lons[y_slice, x_slice],
        y=lats[y_slice, x_slice],
        transform=ccrs.PlateCarree(),        
        ax=axes[i], 
        fig=fig,
        cmap='RdBu_r',
        #cmap='seismic',
        # shaded_info
        levels=np.linspace(-20, 20, 11),  # 使用統一的色階
        #levels=np.linspace(-10, 10, 11),  # 使用統一的色階
        colorbar=True,
        colorbar_offset=-0.00,
        colorbar_fraction_offset=-0.07,
        colorbar_shrink_bai=0.4,
        colorbar_aspect_bai=0.7,
        colorbar_label="nolabel",
        annotation=False,

        # 向量場參數
        vx=vx_plot,
        vy=vy_plot,
        vc1='green',
        vc2='white',
        vwidth=3.5, vlinewidth=0.3,
        vscale=vscale, vref=vref,
        vskip=(20, 10),
        vkey_offset=(0.00, 0.025),
        vunit='no',
        
        # cnt參數
        cnt=cnt[plot_slice],
        cints=(cint, cint2),
        ccolor=ccolor,
        cwidth=(0.4, 0.8),
        
        # other
        title = (f"{var_name}"),
        #title=f"{var_name}[{1/bai:.0e}({var.data.units:~})]\nGPH[({gph.data.units:~})]",

        user_info=[f"cint:{cint}\n[{cnt[plot_slice].data.units:~}]"],
        user_info_loc="right upper",
        user_info_offset=(0.00, 0.00),
        user_info_color=ccolor,
        fig_info=[f"__file__: {__file__}", 
                    f"ifd: {IFD}", 
                    f"shd bai and units: {1/bai_set:.0e} [{var.data.units:~}]"
                ],
        fig_info_offset=(0, 0.05),   # 往上也可以用來增加面板大小
        
        silent=False,         
        coastline_color=None,
        grid=False,
        indent=0,
        show=False,
        aspect_ratio=0.9,
    )

  
    
    # plot vectors
    # X, Y = np.meshgrid(lons['west_east'], lons['south_north'])
    #breakpoint()
    #qu = axes[i].quiver(lons, lats, u_wind[plot_slice], v_wind[plot_slice], 
    #                    color='green', width=5/1000,
    #                    edgecolor='white', linewidth=0.4,
    #                    scale=100, scale_units='inches',
    #                    transform=ccrs.PlateCarree(), regrid_shape=(20, 20), zorder=20)     
    #qk = axes[i].quiverkey(qu, 1.05, 1.03, 25, f'25 [{u_wind.data.units:~}]',
    #                 labelpos='N', coordinates='axes', color='green', labelcolor='green',
    #                 fontproperties={'size': 10}, zorder=99)


    # 繪製 landmask 等值線
    # axes[i].contour(landmask.data, levels=[0.5], colors='black', linewidths=1.5, zorder=9,)

    # 設定地圖範圍（經緯度）
    # 定義不同 domain 的顯示範圍,None 表示使用全域

    if extent_dict[target_domain] is not None:
        axes[i].set_extent(extent_dict[target_domain], crs=ccrs.PlateCarree())
    else:
        # 使用全域範圍
        axes[i].set_xlim(cartopy_xlim(hgt))
        axes[i].set_ylim(cartopy_ylim(hgt))

    # 添加地理參考線
    # 添加地圖特徵 - 粗黑色海岸線在外，黃色細海岸線在內
    if target_domain == 'd03':
        coastline_resolution = '10m'   # 10m, 50m, 110m
    elif target_domain == 'd02':
        coastline_resolution = '50m'   # 10m, 50m, 110m

    axes[i].coastlines(coastline_resolution, linewidth=1.0, color='black', zorder=4, alpha=1.0)
    #axes[i].coastlines(coastline_resolution, linewidth=0.4, color='yellow', zorder=5, alpha=1.0)
    axes[i].add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='gray', facecolor='none', alpha=0.7, zorder=3) # 添加國界

    # 設定經緯度網格線 - for ccrs.LambertConformal
    glint={}
    glint['d03'] = (2, 2)
    glint['d02'] = (5, 2.5)
    xlocs = np.arange(-360, 361, glint[target_domain][0])
    ylocs = np.arange(-90, 91, glint[target_domain][1])
    # 添加經緯參考線
    gl = axes[i].gridlines(
        draw_labels={'bottom': 'x', 'left': 'y'},  # 明確指定標籤位置
        #draw_labels=False,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--',
        zorder=10,
        x_inline=False,  # 不要內嵌標籤
        y_inline=False,
        rotate_labels=False  # 關鍵：防止標籤旋轉
    )
    # 設定經緯線間隔（每 10 度）
    gl.xlocator = mticker.FixedLocator(xlocs)  # 明確指定經度
    gl.ylocator = mticker.FixedLocator(ylocs)    # 明確指定緯度
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()   
    ## 標籤樣式
    gl.xlabel_style = {'size': 10, 'color': 'black', 'rotation': 0}  # 明確設定 rotation=0
    gl.ylabel_style = {'size': 10, 'color': 'black'}


# 在 figure 中上方加總標題
fig.suptitle(f"Vorticity Budget Terms at {target_time}", fontsize=18, fontweight='bold')

# ============================================================================
# 生成輸出檔名
# ============================================================================
time_str = target_time.strftime('%Y%m%d%H')
output_filename = f'ana_vorticity_budget_CTL_mean_{int(target_pressure)}_{time_str}.png'
output_path = os.path.join(output_dir, output_filename)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n圖片已儲存:")
print(f"    {output_path}")

matplotlib.use('TkAgg', force=True)  # 回開互動式後端
#breakpoint()
# endregion


# ============================================================================
# write to nc ana_vorticity_budget_CTL_LEV_TIME
# ============================================================================
print(f"\n{'='*80}")
print(f"Writing NetCDF file...")
print(f"{'='*80}")

# 設定nc檔輸出目錄
nc_output_dir = os.path.join(output_dir, 'nc')
os.makedirs(nc_output_dir, exist_ok=True)

# 選取中間時間和中間氣壓層(保留維度)
time_idx = [1]
level_idx = [1]

# 建立 Dataset 來整合所有變數(只保留中間時間和中間層)
ds_output = xr.Dataset(
    {
        # 平滑後的渦度收支項
        'local_tendency_smoothed': local_tendency_smoothed.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'horizontal_advection_smoothed': horizontal_advection_smoothed.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'vertical_advection_smoothed': vertical_advection_smoothed.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'divergence_effect_smoothed': divergence_effect_smoothed.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'tilting_term_smoothed': tilting_term_smoothed.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'residual_smoothed': residual_smoothed.isel(Time=time_idx, interp_level=level_idx, drop=False),
        
        # 平滑前的渦度收支項
        'local_tendency': local_tendency.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'horizontal_advection': horizontal_advection.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'vertical_advection': vertical_advection.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'divergence_effect': divergence_effect.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'tilting_term': tilting_term.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'residual': residual.isel(Time=time_idx, interp_level=level_idx, drop=False),
        
        # 原始氣象變數
        'u_wind': u_wind.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'v_wind': v_wind.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'w_wind': w_wind.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'omega': omega.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'gph': gph.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'temperature': temperature.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'mixing_ratio': mixing_ratio.isel(Time=time_idx, interp_level=level_idx, drop=False),
        
        # 診斷變數
        'relative_vorticity': relative_vorticity.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'absolute_vorticity': absolute_vorticity.isel(Time=time_idx, interp_level=level_idx, drop=False),
        'divergence': divergence.isel(Time=time_idx, interp_level=level_idx, drop=False),
    }
)
#array_info(relative_vorticity.isel(Time=time_idx, interp_level=level_idx, drop=False))
#breakpoint()

# 添加全域屬性
ds_output.attrs['title'] = 'Relative Vorticity Budget Analysis'
ds_output.attrs['institution'] = 'Your Institution'
ds_output.attrs['source'] = 'WRF Model Output'
ds_output.attrs['analysis_time'] = target_time.strftime('%Y-%m-%d %H:%M:%S')
ds_output.attrs['target_pressure'] = f'{target_pressure} hPa'
ds_output.attrs['domain'] = target_domain
ds_output.attrs['dx'] = f'{dx.magnitude} m'
ds_output.attrs['dy'] = f'{dy.magnitude} m'
ds_output.attrs['smooth_parameter'] = smooth_nd[target_domain]
ds_output.attrs['smooth_method'] = 'gaussian'
ds_output.attrs['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 為每個變數添加屬性描述
var_attrs = {
    'local_tendency': {'long_name': 'Local tendency (raw)', 'description': '∂ζ/∂t'},
    'local_tendency_smoothed': {'long_name': 'Local tendency (smoothed)', 'description': '∂ζ/∂t'},
    'horizontal_advection': {'long_name': 'Horizontal advection (raw)', 'description': '-V·∇(ζ+f)'},
    'horizontal_advection_smoothed': {'long_name': 'Horizontal advection (smoothed)', 'description': '-V·∇(ζ+f)'},
    'vertical_advection': {'long_name': 'Vertical advection (raw)', 'description': '-ω·∂ζ/∂p'},
    'vertical_advection_smoothed': {'long_name': 'Vertical advection (smoothed)', 'description': '-ω·∂ζ/∂p'},
    'divergence_effect': {'long_name': 'Divergence effect (raw)', 'description': '-(ζ+f)∇·V'},
    'divergence_effect_smoothed': {'long_name': 'Divergence effect (smoothed)', 'description': '-(ζ+f)∇·V'},
    'tilting_term': {'long_name': 'Tilting term (raw)', 'description': '∂u/∂p·∂ω/∂y - ∂v/∂p·∂ω/∂x'},
    'tilting_term_smoothed': {'long_name': 'Tilting term (smoothed)', 'description': '∂u/∂p·∂ω/∂y - ∂v/∂p·∂ω/∂x'},
    'residual': {'long_name': 'Residual (raw)', 'description': 'Balance equation error'},
    'residual_smoothed': {'long_name': 'Residual (smoothed)', 'description': 'Balance equation error'},
    'u_wind': {'long_name': 'U wind component', 'description': 'Zonal wind'},
    'v_wind': {'long_name': 'V wind component', 'description': 'Meridional wind'},
    'w_wind': {'long_name': 'Vertical velocity', 'description': 'W wind component'},
    'omega': {'long_name': 'Omega', 'description': 'Pressure vertical velocity'},
    'gph': {'long_name': 'Geopotential height', 'description': 'Height field'},
    'temperature': {'long_name': 'Temperature', 'description': 'Air temperature'},
    'mixing_ratio': {'long_name': 'Water vapor mixing ratio', 'description': 'Mixing ratio'},
    'relative_vorticity': {'long_name': 'Relative vorticity', 'description': 'ζ = ∂v/∂x - ∂u/∂y'},
    'absolute_vorticity': {'long_name': 'Absolute vorticity', 'description': 'ζ + f'},
    'divergence': {'long_name': 'Horizontal divergence', 'description': '∇·V = ∂u/∂x + ∂v/∂y'},
}

for var_name, attrs in var_attrs.items():
    if var_name in ds_output:
        ds_output[var_name].attrs.update(attrs)

# 生成輸出檔名
nc_filename = f'ana_vorticity_budget_CTL_mean_{int(target_pressure)}_{time_str}.nc'
nc_output_path = os.path.join(nc_output_dir, nc_filename)

# 設定編碼:轉換為32位元並加上壓縮
encoding = {}
for var in ds_output.data_vars:
    encoding[var] = {
        'dtype': 'float32',
        'zlib': True,
        'complevel': 4,
        '_FillValue': np.nan
    }

# 寫出NetCDF檔案
ds_output.to_netcdf(nc_output_path, encoding=encoding)

print(f"\nNetCDF檔案已儲存:")
print(f"    {nc_output_path}")
print(f"    檔案包含變數數量: {len(ds_output.data_vars)}")
print(f"    資料維度: {dict(ds_output.sizes)}")

#======================================================================================================================
