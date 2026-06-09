#!/usr/bin/env python3
"""
將 WPS 產生的 met_em*.nc 檔案中的 SST 替換為觀測海溫。

主要流程：
1. 複製原始 met_em 檔到輸出資料夾，避免直接改動原始檔。
2. 用 xarray 開啟觀測 SST 檔，依 -T 指定日期取出當天 SST 場。
3. 從 met_em 讀取 WRF 格點經緯度與原始 SST。
4. 呼叫 definitions.interpolate_griddata 內插觀測 SST 到 WRF 格點。
5. 僅替換 met_em 原本為海面的 SST 格點，保留陸地或無效值。

注意：
- met_em 的 SST 通常是 Kelvin；OISST / NOAA daily SST 常見為 Celsius。
  程式會先依 units 屬性與數值量級推估單位，再轉成 met_em 原本單位。
- 觀測資料與 WRF 網格可能一個使用 0..360 經度、一個使用 -180..180 經度；
  程式會在內插前自動對齊經度座標。
"""

import argparse
import importlib.util
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import definitions as mydef


# 腳本所在目錄，用來尋找 PY_No_MoNo/definitions 裡的共用函式。
SCRIPT_DIR = Path(__file__).resolve().parent

# 若使用者未指定 -n，預設輸出到目前工作目錄底下的專用資料夾。
DEFAULT_OUTPUT_DIR = f"./output_{Path(__file__).stem}"


def load_mydef():
    """載入專案內的 definitions，並取得 mydef.interpolate_griddata。

    definitions/__init__.py 會動態載入許多工具；若使用者環境少了部分繪圖套件，
    完整 import 可能失敗。這裡提供 fallback，只直接載入本腳本真正需要的
    definitions/interpolate_griddata.py，讓替換 SST 的核心功能不受其他工具影響。
    """
    try:
        import definitions as mydef

        return mydef
    except Exception as exc:
        # fallback：只載入 interpolation 模組，避免因 cartopy/matplotlib 等套件缺失而中斷。
        interp_path = SCRIPT_DIR / "definitions" / "interpolate_griddata.py"
        spec = importlib.util.spec_from_file_location(
            "interpolate_griddata", interp_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {interp_path}") from exc

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class MyDef:
            pass

        mydef = MyDef()
        mydef.interpolate_griddata = module.interpolate_griddata
        print(
            "[WARN] Failed to import full definitions package; "
            "loaded definitions/interpolate_griddata.py only."
        )
        print(f"[WARN] Original import error: {exc}")
        return mydef


def load_netcdf4():
    """延後載入 netCDF4。

    這樣使用者執行 --help 時不需要先具備完整科學運算環境；真正開始處理檔案
    時才檢查 netCDF4 是否存在。
    """
    try:
        from netCDF4 import Dataset, num2date

        return Dataset, num2date
    except ImportError as exc:
        raise ImportError(
            "This script requires netCDF4. Please run it in the same Python "
            "environment used for WRF/WPS NetCDF tools."
        ) from exc


def load_xarray():
    """延後載入 xarray，用於開啟觀測 SST 與依日期選取資料。"""
    try:
        import xarray as xr

        return xr
    except ImportError as exc:
        raise ImportError(
            "This script requires xarray to read and select observed SST data."
        ) from exc


def parse_args():
    """解析命令列參數。

    -i1: met_em*.nc，通常每個檔案只含一個 WPS 時間點。
    -i2: 觀測 SST 檔，通常 daily SST 一年一檔。
    -T : 要從觀測檔取出的日期。
    -n : 輸出資料夾。
    -o : 輸出檔名。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replace SST in a WPS met_em NetCDF file with an observed SST field."
        )
    )
    parser.add_argument(
        "-i1",
        dest="met_file",
        required=True,
        help="Input met_em*.nc file. Each file is expected to contain one time.",
    )
    parser.add_argument(
        "-i2",
        dest="obs_file",
        required=True,
        help="Observed SST NetCDF file, for example sst.day.mean.2006.nc.",
    )
    parser.add_argument(
        "-T",
        dest="target_date",
        required=True,
        help='Observed SST date to use, for example "2006-06-08".',
    )
    parser.add_argument(
        "-n",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-o",
        dest="output_name",
        default=None,
        help="Output file name. Default: same basename as -i1.",
    )
    return parser.parse_args()


def parse_date_string(value):
    """將 -T 傳入的字串轉成 date。

    允許使用者輸入 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS" 類型字串；
    觀測 SST 是每日一筆，所以只取前 10 個字元作為日期。
    """
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f'-T must look like "YYYY-MM-DD". Got: {value}') from exc


def time_value_to_date_string(value):
    """把不同型態的時間值統一轉成 YYYY-MM-DD 字串。

    xarray 讀進來的 time 可能是 numpy.datetime64、cftime/datetime 物件、
    byte string 或一般字串。日期比對時統一轉成字串可避免型態差異。
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return np.datetime_as_string(value.astype("datetime64[D]"), unit="D")

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    text = str(value)
    if len(text) >= 10:
        return text[:10]
    return None


def find_name(ds, candidates, label):
    """依候選名稱尋找 NetCDF 變數或座標名稱。"""
    for name in candidates:
        if name in ds.variables or name in ds.coords:
            return name
    available = sorted(set(ds.variables) | set(ds.coords))
    raise KeyError(f"Cannot find {label}. Tried {candidates}. Available: {available}")


def find_obs_sst_name(ds):
    """尋找觀測檔中的 SST 變數名稱。

    NOAA OISST 常用小寫 sst；WRF/met_em 則常用大寫 SST。這裡同時支援兩者。
    """
    for name in ("sst", "SST"):
        if name in ds.data_vars:
            return name

    lower_map = {name.lower(): name for name in ds.data_vars}
    if "sst" in lower_map:
        return lower_map["sst"]

    raise KeyError(f"Cannot find observed SST variable. Data variables: {list(ds.data_vars)}")


def find_time_name(ds, field):
    """尋找觀測 SST 使用的時間維度或時間座標名稱。"""
    for name in ("time", "Time"):
        if name in field.dims or name in ds.variables or name in ds.coords:
            return name

    for dim in field.dims:
        if "time" in dim.lower():
            return dim

    raise KeyError(f"Cannot find time dimension in observed SST. Dims: {field.dims}")


def select_obs_date(field, ds, time_name, target_date):
    """從觀測 SST 中選出 -T 指定日期的 2D/待壓縮資料。

    此處不用 xarray.sel(time=...)，而是把時間逐筆轉成 YYYY-MM-DD 再比對，
    目的是同時支援標準 datetime64 與 cftime 類型的 NetCDF 時間座標。
    """
    if time_name not in field.dims:
        raise KeyError(f"Observed SST variable does not use time dimension: {time_name}")

    time_values = ds[time_name].values
    target_date_string = target_date.strftime("%Y-%m-%d")
    matches = [
        index
        for index, value in enumerate(np.ravel(time_values))
        if time_value_to_date_string(value) == target_date_string
    ]

    if not matches:
        # 找不到日期時列出前幾筆時間，方便檢查年份檔或日期格式是否拿錯。
        available_sample = [
            time_value_to_date_string(value) for value in np.ravel(time_values[:5])
        ]
        raise ValueError(
            f"Date {target_date_string} not found in observed SST time coordinate. "
            f"First available dates: {available_sample}"
        )

    if len(matches) > 1:
        print(
            f"[WARN] Date {target_date_string} matched {len(matches)} times; "
            "using the first one."
        )

    return field.isel({time_name: matches[0]})


def squeeze_non_spatial_dims(field, lon_da, lat_da):
    """移除非空間維度，並整理成 interpolate_griddata 需要的空間維度順序。

    觀測 SST 取出單一日期後，理想上只剩 lat/lon 兩個維度。若還存在其他維度
    （例如 depth、member 等），代表資料不是本腳本預期的 daily surface SST，
    會直接報錯避免誤寫。
    """
    field = field.squeeze(drop=True)

    if lon_da.ndim == 1 and lat_da.ndim == 1:
        # 規則經緯度網格：lon(lat) 各是一維座標，SST 需整理成 (lat, lon)。
        lon_dim = lon_da.dims[0]
        lat_dim = lat_da.dims[0]
        if lon_dim not in field.dims or lat_dim not in field.dims:
            raise ValueError(
                "Observed SST dimensions do not contain lon/lat dimensions. "
                f"SST dims={field.dims}, lon_dim={lon_dim}, lat_dim={lat_dim}"
            )

        other_dims = [dim for dim in field.dims if dim not in (lat_dim, lon_dim)]
        if other_dims:
            sizes = {dim: field.sizes[dim] for dim in other_dims}
            raise ValueError(
                "Observed SST still has non-spatial dimensions after squeezing: "
                f"{sizes}"
            )

        return field.transpose(lat_dim, lon_dim)

    if lon_da.ndim == 2 and lat_da.ndim == 2:
        # 曲線網格：lon/lat 本身就是二維，SST 需使用相同的兩個維度。
        if lon_da.dims != lat_da.dims:
            raise ValueError(
                f"Observed 2D lon/lat dims differ: lon={lon_da.dims}, lat={lat_da.dims}"
            )
        if not all(dim in field.dims for dim in lon_da.dims):
            raise ValueError(
                "Observed SST dimensions do not contain 2D lon/lat dimensions. "
                f"SST dims={field.dims}, lon/lat dims={lon_da.dims}"
            )

        other_dims = [dim for dim in field.dims if dim not in lon_da.dims]
        if other_dims:
            sizes = {dim: field.sizes[dim] for dim in other_dims}
            raise ValueError(
                "Observed SST still has non-spatial dimensions after squeezing: "
                f"{sizes}"
            )

        return field.transpose(*lon_da.dims)

    raise ValueError(
        "Observed lon/lat must be paired 1D arrays or paired 2D arrays. "
        f"Got lon.ndim={lon_da.ndim}, lat.ndim={lat_da.ndim}"
    )


def finite_minmax(values):
    """回傳有限值的最小與最大值；若全是 NaN 則回傳 NaN。"""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(valid)), float(np.nanmax(valid))


def align_longitudes(source_lon, target_lon):
    """讓來源與目標經度使用同一套座標系。

    觀測 SST 常見經度是 0..360；WRF 網格有時是 -180..180。
    scipy griddata 只看數值座標，不理解兩者其實代表同一個位置，因此內插前
    必須把其中一方轉到另一方的經度系統。
    """
    source = np.asarray(source_lon, dtype=float).copy()
    target = np.asarray(target_lon, dtype=float).copy()

    source_min, source_max = finite_minmax(source)
    target_min, target_max = finite_minmax(target)

    if source_min >= 0 and source_max > 180 and target_min < 0:
        target = np.mod(target, 360.0)
        print("[INFO] Converted target longitudes to 0..360 for interpolation.")
    elif target_min >= 0 and target_max > 180 and source_min < 0:
        source = np.mod(source, 360.0)
        print("[INFO] Converted source longitudes to 0..360 for interpolation.")

    return source, target


def mean_spacing_1d(values):
    """估算一維座標的典型格距，用於裁切來源觀測範圍時保留邊界緩衝。"""
    values = np.asarray(values, dtype=float)
    diff = np.diff(values)
    diff = diff[np.isfinite(diff)]
    diff = np.abs(diff)
    diff = diff[diff > 0]
    if diff.size == 0:
        return 0.0
    return float(np.nanmedian(diff))


def crop_regular_source(field, lon_values, lat_values, target_lon, target_lat):
    """裁切規則觀測網格到 WRF 網格附近，降低 griddata 計算量。

    OISST 這類全球 daily SST 若整張丟進 scipy.griddata，點數很多且速度慢。
    對一維 lon/lat 規則網格，可以先依 WRF 目標範圍裁出附近區域，並保留約兩格
    緩衝，避免邊界內插缺值。若來源不是規則一維網格，則直接使用全場。
    """
    if lon_values.ndim != 1 or lat_values.ndim != 1:
        return field, lon_values, lat_values

    lon_dim = field.dims[-1]
    lat_dim = field.dims[-2]

    lon_min, lon_max = finite_minmax(target_lon)
    lat_min, lat_max = finite_minmax(target_lat)
    dlon = mean_spacing_1d(lon_values)
    dlat = mean_spacing_1d(lat_values)
    lon_margin = max(2.0 * dlon, 0.01)
    lat_margin = max(2.0 * dlat, 0.01)

    lon_mask = (lon_values >= lon_min - lon_margin) & (lon_values <= lon_max + lon_margin)
    lat_mask = (lat_values >= lat_min - lat_margin) & (lat_values <= lat_max + lat_margin)

    if np.count_nonzero(lon_mask) < 2 or np.count_nonzero(lat_mask) < 2:
        # 裁切結果太小通常代表經度系統或資料範圍不一致，改用全場並讓後續檢核處理。
        print("[WARN] Source crop would be too small; using full observed grid.")
        return field, lon_values, lat_values

    lon_indices = np.where(lon_mask)[0]
    lat_indices = np.where(lat_mask)[0]
    cropped = field.isel({lon_dim: lon_indices, lat_dim: lat_indices})
    print(
        "[INFO] Cropped observed grid: "
        f"lon {lon_values[lon_indices[0]]:.3f}..{lon_values[lon_indices[-1]]:.3f}, "
        f"lat {lat_values[lat_indices[0]]:.3f}..{lat_values[lat_indices[-1]]:.3f}, "
        f"shape={cropped.shape}"
    )
    return cropped, lon_values[lon_indices], lat_values[lat_indices]


def infer_temperature_unit(values, attrs):
    """推估 SST 單位是 Kelvin 或 Celsius。

    優先使用 NetCDF 變數的 units 屬性；若沒有明確單位，則以數值大小判斷：
    SST 最大值大於 100 時通常是 Kelvin，否則視為 Celsius。
    """
    units = str(attrs.get("units", "")).strip().lower()
    if units in ("k", "kelvin", "degree_kelvin", "degrees_kelvin"):
        return "K"
    if (
        units in ("c", "degc", "degree_c", "degrees_c", "celsius")
        or "celsius" in units
        or "degree c" in units
        or "degrees c" in units
    ):
        return "C"

    _, vmax = finite_minmax(values)
    if np.isfinite(vmax) and vmax > 100:
        return "K"
    return "C"


def convert_temperature(values, source_unit, target_unit):
    """將觀測 SST 轉成 met_em 原始 SST 的單位。"""
    converted = np.asarray(values, dtype=float).copy()
    if source_unit == target_unit:
        return converted
    if source_unit == "C" and target_unit == "K":
        return converted + 273.15
    if source_unit == "K" and target_unit == "C":
        return converted - 273.15
    raise ValueError(f"Unsupported temperature conversion: {source_unit} to {target_unit}")


def read_obs_sst(obs_file, target_date, target_lon, target_lat):
    """讀取觀測 SST，選出指定日期並整理成可內插的資料結構。"""
    xr = load_xarray()

    with xr.open_dataset(obs_file) as ds:
        # 不同資料集可能使用 sst/SST、lon/longitude、lat/latitude 等名稱。
        sst_name = find_obs_sst_name(ds)
        lon_name = find_name(ds, ("lon", "longitude", "XLONG", "XLONG_M"), "longitude")
        lat_name = find_name(ds, ("lat", "latitude", "XLAT", "XLAT_M"), "latitude")

        # 取出指定日期，並確認最後只剩空間維度。
        field = ds[sst_name]
        time_name = find_time_name(ds, field)
        selected = select_obs_date(field, ds, time_name, target_date)
        selected = squeeze_non_spatial_dims(selected, ds[lon_name], ds[lat_name])

        # 內插前先讓觀測經度與 WRF 經度位於同一座標系。
        source_lon, target_lon_for_interp = align_longitudes(
            ds[lon_name].values, target_lon
        )
        source_lat = np.asarray(ds[lat_name].values, dtype=float)

        selected, source_lon, source_lat = crop_regular_source(
            selected, source_lon, source_lat, target_lon_for_interp, target_lat
        )

        # 在關閉檔案前把資料實際載入記憶體，避免 lazy array 失效。
        selected = selected.load()

        return {
            "sst_name": sst_name,
            "time_name": time_name,
            "lon_name": lon_name,
            "lat_name": lat_name,
            "values": np.asarray(selected.values, dtype=float),
            "lon": source_lon,
            "lat": source_lat,
            "attrs": dict(field.attrs),
            "target_lon_for_interp": target_lon_for_interp,
        }


def read_met_coord(ncfile, candidates):
    """從 met_em 檔讀取二維經緯度場。

    WPS met_em 常見座標名稱是 XLONG_M/XLAT_M；某些檔案可能只有 XLONG/XLAT。
    讀取後 squeeze 掉 Time 維度，最後只接受二維陣列。
    """
    for name in candidates:
        if name in ncfile.variables:
            data = np.ma.filled(ncfile.variables[name][:], np.nan)
            data = np.asarray(data, dtype=float)
            data = np.squeeze(data)
            if data.ndim == 2:
                return name, data
    raise KeyError(f"Cannot find a 2D coordinate from: {candidates}")


def get_met_sst_plane(sst_var):
    """取得 met_em 中要替換的 SST 平面與其寫回索引。

    met_em 的 SST 多半是 (Time, south_north, west_east)，且每個檔案一個時間點。
    若遇到 2D SST 也支援；若 Time 超過一筆，依需求只替換第 0 筆。
    """
    if sst_var.ndim == 2:
        return (slice(None), slice(None)), np.ma.filled(sst_var[:, :], np.nan)

    if sst_var.ndim == 3:
        if sst_var.shape[0] != 1:
            print(
                f"[WARN] SST has {sst_var.shape[0]} time levels; replacing index 0 only."
            )
        return (0, slice(None), slice(None)), np.ma.filled(sst_var[0, :, :], np.nan)

    raise ValueError(f"Expected SST to be 2D or 3D. Got shape: {sst_var.shape}")


def prepare_output_path(met_file, output_dir, output_name):
    """建立輸出路徑，並避免輸出位置等於原始 met 檔。"""
    met_path = Path(met_file)
    out_dir = Path(output_dir)
    out_name = output_name if output_name else met_path.name
    out_path = out_dir / Path(out_name).name

    if met_path.resolve() == out_path.resolve():
        raise ValueError(
            "Output path is the same as input met file. "
            "Use a different -n or -o to avoid modifying the original file."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_path


def replace_sst(args):
    """執行完整替換流程。"""
    # 真正處理資料時才載入大型依賴，讓 --help 可以在簡單環境中正常執行。
    Dataset, _ = load_netcdf4()
    mydef = load_mydef()

    met_file = Path(args.met_file)
    obs_file = Path(args.obs_file)
    target_date = parse_date_string(args.target_date)

    if not met_file.exists():
        raise FileNotFoundError(f"met file not found: {met_file}")
    if not obs_file.exists():
        raise FileNotFoundError(f"observed SST file not found: {obs_file}")

    # 先複製 met_em 到輸出位置；後續所有寫入都只發生在複製檔。
    out_path = prepare_output_path(met_file, args.output_dir, args.output_name)
    shutil.copy2(met_file, out_path)

    print("=" * 72)
    print("Replace met_em SST with observed SST")
    print("=" * 72)
    print(f"met input : {met_file}")
    print(f"obs input : {obs_file}")
    print(f"date      : {target_date}")
    print(f"output    : {out_path}")
    print("-" * 72)

    with Dataset(out_path, "r+") as met_ds:
        if "SST" not in met_ds.variables:
            raise KeyError(f"Cannot find SST in met file: {out_path}")

        # 從 met_em 取目標格點與原始 SST。這些資料決定內插目標與替換遮罩。
        lon_name, met_lon = read_met_coord(met_ds, ("XLONG_M", "XLONG"))
        lat_name, met_lat = read_met_coord(met_ds, ("XLAT_M", "XLAT"))
        sst_var = met_ds.variables["SST"]
        sst_index, old_sst = get_met_sst_plane(sst_var)
        old_sst = np.asarray(old_sst, dtype=float)

        if met_lon.shape != old_sst.shape or met_lat.shape != old_sst.shape:
            raise ValueError(
                "met lon/lat shape must match SST shape. "
                f"lon={met_lon.shape}, lat={met_lat.shape}, SST={old_sst.shape}"
            )

        print(f"[INFO] met lon/lat variables: {lon_name}, {lat_name}")
        print(f"[INFO] met SST shape: {old_sst.shape}")

        # 讀取觀測檔指定日期的 SST，並整理來源 lon/lat 與 WRF 目標經度座標。
        obs = read_obs_sst(obs_file, target_date, met_lon, met_lat)
        print(
            "[INFO] observed SST: "
            f"var={obs['sst_name']}, time={obs['time_name']}, "
            f"lon={obs['lon_name']}, lat={obs['lat_name']}, shape={obs['values'].shape}"
        )

        # 觀測 SST 與 met SST 單位需一致，否則 WRF 後續會讀到錯誤量級。
        old_unit = infer_temperature_unit(old_sst, dict(sst_var.__dict__))
        obs_unit = infer_temperature_unit(obs["values"], obs["attrs"])
        obs_values = convert_temperature(obs["values"], obs_unit, old_unit)
        print(f"[INFO] temperature units: observed={obs_unit}, met={old_unit}")

        # 將觀測 SST 從原始規則/曲線網格內插到 met_em 的 WRF 格點。
        interpolated = mydef.interpolate_griddata(
            obs_values,
            obs["lon"],
            obs["lat"],
            obs["target_lon_for_interp"],
            met_lat,
            method="linear",
        )
        interpolated = np.asarray(interpolated, dtype=float)

        if interpolated.shape != old_sst.shape:
            raise ValueError(
                "Interpolated SST shape does not match met SST shape. "
                f"interpolated={interpolated.shape}, met={old_sst.shape}"
            )

        # 只替換有效海面：
        # - 內插值必須是有限值
        # - met 原本 SST 必須是有限值
        # - met 原本 SST != 0，避免把 WPS 用 0 表示的陸地點改成海溫
        valid_mask = np.isfinite(interpolated) & np.isfinite(old_sst) & (old_sst != 0)
        modified_count = int(np.count_nonzero(valid_mask))
        total_count = int(old_sst.size)

        # 保留非海面或內插缺值點原樣，只把 valid_mask 範圍寫成觀測 SST。
        new_sst = old_sst.copy()
        new_sst[valid_mask] = interpolated[valid_mask]
        sst_var[sst_index] = new_sst  # 把新 SST 寫入檔案

        old_min, old_max = finite_minmax(old_sst[valid_mask])
        new_min, new_max = finite_minmax(new_sst[valid_mask])
        print("-" * 72)
        print(f"[DONE] Replaced points: {modified_count} / {total_count} ({modified_count / total_count:.2%})")
        print(f"[INFO] old SST values range on replaced points: {old_min:.3f} ~ {old_max:.3f}")
        print(f"[INFO] new SST values range on replaced points: {new_min:.3f} ~ {new_max:.3f}")

        # 離開 with 區塊：自動關檔並完成保存
        # breakpoint()

    print("=" * 72)
    print(f"Saved: {out_path}")
    print("=" * 72)
    return out_path


def main():
    """命令列入口。"""
    args = parse_args()
    try:
        replace_sst(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
