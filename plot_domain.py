#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from netCDF4 import Dataset
from wrf import getvar, interpline, CoordPair, xy_to_ll, ll_to_xy

import definitions as mydef
from definitions.plot_2D_shaded import plot_2D_shaded as p2d

# ======================================================================================================

def extract_boundary_points(da):
    """提取 2D DataArray 的四個邊界值（順時針方向：下 -> 左 -> 上 -> 右）"""
    bottom = da[0, :]
    left = da[:, 0]
    top = da[-1, :]
    right = da[:, -1]

    boundary_pts = np.concatenate(
        [bottom.values, left.values, top.values, right.values]
    )
    return boundary_pts.tolist()

def plot_domains(inputs, cints_val):
    """
    處理 WRF geo_em 檔案，提取地形與邊界，並進行繪圖邏輯。
    接收 inputs (檔案列表) 與 cints_val (控制間隔的數值)。
    """
    print("\n--- 開始處理資料與繪圖流程 ---")
    
    bdy_lons = {}
    bdy_lats = {}
    d01_hgt = None
    d01_lons = None
    d01_lats = None
    d01_land = None
    d01_ncfile = None

    domain_pattern = re.compile(r"geo_em\.(d\d+)\.nc")

    for file_path in inputs:
        filename = os.path.basename(file_path)
        match = domain_pattern.search(filename)

        if not match:
            print(f"警告: 檔案名稱 {filename} 不符合 geo_em.d0X.nc 格式，跳過處理。")
            continue

        dom_id = match.group(1)
        print(f"正在處理檔案: {filename} ({dom_id})")

        with xr.open_dataset(file_path) as ds:
            ds_compat = ds.squeeze()

            if dom_id == "d01":
                if "HGT_M" in ds_compat:
                    d01_hgt = ds_compat["HGT_M"]
                    d01_lons = ds_compat["XLONG_M"]
                    d01_lats = ds_compat["XLAT_M"]
                    d01_land = ds_compat["LANDMASK"]
                    d01_ncfile = Dataset(file_path)

                    # 若經度小於 0，則加上 360；否則保持原值
                    d01_lons = xr.where(d01_lons < 0, d01_lons + 360, d01_lons)
                    
                else:
                    raise KeyError("在 d01 檔案中找不到 HGT_M 變數！")
            else:
                if "XLONG_M" in ds_compat and "XLAT_M" in ds_compat:
                    lon_da = ds_compat["XLONG_M"]
                    lat_da = ds_compat["XLAT_M"]
                    bdy_lons[dom_id] = extract_boundary_points(lon_da)
                    bdy_lats[dom_id] = extract_boundary_points(lat_da)

    if d01_hgt is None:
        raise ValueError("錯誤: 未輸入 d01 檔案！")

    # 計算 figsize
    long = 7.0
    ny, nx = d01_lons.shape
    fig_w, fig_h = (long, long * (ny / nx)) if nx >= ny else (long * (nx / ny), long)

    # 在這裡使用傳入的 cints_val
    result = p2d(
        array=d01_hgt,
        cmap='terrain',
        colorbar_label="[m]",
        colorbar_shrink_bai=0.7,

        cnt=[d01_lons, d01_lats, d01_land],
        # 將原本硬編碼的 (10, 10) 替換為使用者輸入的變數
        cints=[(cints_val, cints_val), (cints_val, cints_val), (None, None)],
        clevels=[(None, None), (None, None), ([0.5], [0.5])],
        cwidth=[(0.6, 0.6), (0.6, 0.6), (0.9, 0.9)],
        ccolor=['gray', 'gray', 'black'],
        ctype=[('--', '--'), ('--', '--'), ('-', '-')],
        cntype=[('--', '--'), ('--', '--'), ('-', '-')],
        clab=[(False, False), (False, False), (False, False)],
        grid_alpha=0,
        figsize=(fig_w, fig_h),
        show=False 
    )

    fig = result['fig']
    ax = result['ax']

    for dom in bdy_lons.keys():
        bdy_xs, bdy_ys = ll_to_xy(d01_ncfile, bdy_lats[dom], bdy_lons[dom])
        ax.plot(bdy_xs, bdy_ys, color='red', linestyle='-', linewidth=1.5, zorder=200)

        max_x = bdy_xs.max().item()
        max_y = bdy_ys.max().item()
        ax.text(max_x + 1, max_y + 1, dom, color='red', fontsize=10, fontweight='bold', 
                ha='left', va='bottom', zorder=201,
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])

    return fig, ax

# ======================================================================================================
def main():
    parser = argparse.ArgumentParser(description="處理 WRF geo_em.nc 檔案並抓出地形與子網格邊界。")
    parser.add_argument("-i", "--inputs", nargs="+", required=True, help="輸入一個或多個 geo_em.d0X.nc 檔案")
    parser.add_argument("-o", "--output", default="./plot_domain_output.png", help="輸出路徑/檔名")
    parser.add_argument("-c", "--cints", type=float, default=10.0, help="設定經緯度等值線的間隔 (預設: 10.0)")
    # -----------------

    # 解析參數
    args = parser.parse_args()

    # 將 args.cints 傳入函式
    fig, ax = plot_domains(args.inputs, args.cints)
    
    # save
    mydef.f2p(figure=fig, out=args.output)    

if __name__ == "__main__":
    main()
