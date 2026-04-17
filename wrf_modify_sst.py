#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import numpy as np
from netCDF4 import Dataset

def main():
    # 1. 設定命令列參數解析
    parser = argparse.ArgumentParser(description="調整 WRF WPS met_em.nc 檔案中的 SST (海表溫度)")
    
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="輸入的 met_em*.nc 檔案路徑")
    parser.add_argument('-o', '--out_path', type=str, required=True, 
                        help="輸出資料夾路徑")
    parser.add_argument('-LL', '--lonlat', type=float, nargs=4, 
                        default=[105.0, 117.0, 15.0, 25.0],
                        metavar=('LON1', 'LON2', 'LAT1', 'LAT2'),
                        help="經緯度範圍預設: 105 117 15 25 (lon1 lon2 lat1 lat2)")
    parser.add_argument('-adj', '--adjust', type=float, required=True, 
                        help="要調整的溫度數值 (例如: 2 或 -1.5)")

    args = parser.parse_args()

    input_file = args.input
    out_dir = args.out_path
    lon1, lon2, lat1, lat2 = args.lonlat
    adj_temp = args.adjust

    # 2. 準備輸出路徑與檔案
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"已建立輸出資料夾: {out_dir}")

    # 取得原始檔名，並組裝輸出路徑
    filename = os.path.basename(input_file)
    out_file = os.path.join(out_dir, filename)

    print("--------------------------------------------------")
    print(f"input_file: {input_file}")
    print(f"正在處理檔案: {filename}")
    print(f"目標範圍: 經度 {lon1}–{lon2}E, 緯度 {lat1}–{lat2}N")
    print(f"SST 調整量: {adj_temp:g} K")
    print("--------------------------------------------------")

    # 複製檔案至輸出路徑 (保留原始檔案不被修改)
    try:
        shutil.copy2(input_file, out_file)
        print(f"已複製檔案至: {out_file}")
    except Exception as e:
        print(f"複製檔案失敗: {e}")
        return

    # 3. 開啟並修改 NetCDF 檔案 (使用 'r+' 模式進行讀寫)
    try:
        with Dataset(out_file, 'r+') as nc:
            # 讀取經緯度網格 (取第一個時間維度 [0,:,:]) 形狀是 (Time, south_north, west_east)
            lon_grid = nc.variables['XLONG_M'][0, :, :]
            lat_grid = nc.variables['XLAT_M'][0, :, :]
            
            # 建立空間遮罩 (Spatial Mask)，找出在經緯度範圍內的網格
            spatial_mask = (lon_grid >= lon1) & (lon_grid <= lon2) & \
                           (lat_grid >= lat1) & (lat_grid <= lat2)
            
            # 讀取 SST 資料
            sst_var = nc.variables['SST']
            sst_data = sst_var[:] # 包含所有的時間維度
            
            modified_count = 0
            
            # 處理所有的時間維度 (雖然通常 met_em 只有 1 個時間步長，但寫迴圈比較安全)
            for t in range(sst_data.shape[0]):
                current_sst = sst_data[t, :, :]
                
                # 建立修改條件：在空間範圍內，且 SST 不等於 0 (排除陸地)
                valid_mask = spatial_mask & (current_sst != 0)
                
                # 應用溫度調整
                sst_data[t, valid_mask] += adj_temp
                
                # 計算被修改的網格點數量 (方便確認結果)
                modified_count += np.sum(valid_mask)
            
            # 將修改後的資料寫回 NetCDF 檔案
            sst_var[:] = sst_data

            # 取得總網格數 (用於計算比例)
            # sst_data.size 包含所有維度的總乘積 (Time * Lat * Lon)
            total_elements = sst_data.size 
            
            # 將修改後的資料寫回 NetCDF 檔案
            sst_var[:] = sst_data
            
            # 計算比例
            percentage = (modified_count / total_elements) * 100
            
            print(f"處理完成！")
            print(f"修改點數: {modified_count} pts / 總點數: {total_elements} pts")
            print(f"所佔比例: {percentage:g}%")
            
    except Exception as e:
        print(f"處理 NetCDF 檔案時發生錯誤: {e}")
        return

    print("==================================================")
    print("處理結束。")
    print("==================================================")

if __name__ == "__main__":
    main()
