#!/usr/bin/env python3
"""
Run extract_wrf_to_nc.py and save results in a fixed w2nc tree.

Example:
    python run_w2nc.py -i ./path -V z -D d02 -L 850 -T 2006-06-09_00:00:00 -E 1

Output:
    ./w2nc/path/z/d02/850/m001/20060609000000.nc
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


INVALID_PATH_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')


def split_csv(value: str) -> list[str]:
    return [part.strip().strip('"').strip("'") for part in value.split(",") if part.strip()]


def safe_path_part(value: str) -> str:
    cleaned = INVALID_PATH_CHARS.sub("_", value.strip().strip('"').strip("'"))
    cleaned = cleaned.strip().strip(".")
    if cleaned in {"", ".."}:
        return "_"
    return cleaned


def time_to_digits(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits or safe_path_part(value)


def wsl_cwd_variant(cwd: Path) -> str | None:
    drive = cwd.drive.rstrip(":").lower()
    if not drive:
        return None
    rest = cwd.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def strip_known_cwd_prefix(path_text: str, cwd: Path) -> str:
    normalized = path_text.replace("\\", "/").rstrip("/")
    cwd_variants = [cwd.as_posix().rstrip("/")]
    wsl_variant = wsl_cwd_variant(cwd)
    if wsl_variant:
        cwd_variants.append(wsl_variant.rstrip("/"))

    for prefix in cwd_variants:
        prefix_with_slash = f"{prefix}/"
        if normalized.lower().startswith(prefix_with_slash.lower()):
            return normalized[len(prefix_with_slash) :]
    return normalized


def input_path_key(input_dir: str) -> Path:
    raw = input_dir.strip().strip('"').strip("'")
    cwd = Path.cwd().resolve()

    try:
        input_path = Path(raw).expanduser()
        if input_path.is_absolute():
            try:
                rel = input_path.resolve().relative_to(cwd)
                parts = rel.parts
            except (OSError, ValueError):
                text = strip_known_cwd_prefix(raw, cwd)
                text = re.sub(r"^[A-Za-z]:/?", "", text.replace("\\", "/"))
                parts = tuple(part for part in text.lstrip("/").split("/") if part)
        else:
            text = raw.replace("\\", "/")
            while text.startswith("./"):
                text = text[2:]
            parts = tuple(part for part in text.split("/") if part and part != ".")
    except OSError:
        text = strip_known_cwd_prefix(raw, cwd)
        parts = tuple(part for part in text.replace("\\", "/").lstrip("/").split("/") if part)

    safe_parts = [safe_path_part(part) for part in parts if part not in {"", "."}]
    return Path(*safe_parts) if safe_parts else Path("input")


def member_for_extract(member: str) -> str:
    match = re.fullmatch(r"member0*(\d+)", member.strip(), flags=re.IGNORECASE)
    return match.group(1) if match else member


def member_output_dir(member: str) -> str:
    cleaned = member.strip().strip('"').strip("'")
    match = re.fullmatch(r"(?:m|member)?0*(\d+)", cleaned, flags=re.IGNORECASE)
    if match:
        return f"m{int(match.group(1)):03d}"
    return safe_path_part(cleaned)


def adjusted_output_candidates(target_file: Path, started_at: float) -> list[Path]:
    return sorted(
        path
        for path in target_file.parent.glob(f"{target_file.stem}*.nc")
        if path.name != target_file.name and path.stat().st_mtime >= started_at - 2
    )


def normalize_output_name(target_file: Path, started_at: float) -> None:
    if target_file.exists():
        return

    candidates = adjusted_output_candidates(target_file, started_at)
    if len(candidates) == 1:
        candidates[0].replace(target_file)
        print(f"Renamed adjusted output to fixed name: {target_file}")
    elif candidates:
        print("Warning: extractor wrote adjusted filenames; could not choose one to rename:")
        for candidate in candidates:
            print(f"  {candidate}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wrapper for extract_wrf_to_nc.py with fixed ./w2nc output layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", required=True, help="WRF input root directory.")
    parser.add_argument("-V", "--variable", required=True, help="Variable name(s), comma-separated.")
    parser.add_argument("-L", "--levels", required=True, help="Level(s), comma-separated.")
    parser.add_argument("-T", "--times", required=True, help='Time(s), comma-separated, e.g. "2006-06-09_00:00:00".')
    parser.add_argument("-E", "--ensemble", required=True, help="Ensemble member(s), comma-separated.")
    parser.add_argument("-D", "--domain", default="d02", help="WRF domain.")
    parser.add_argument("-LL", "--lonlat_grid", default=None, help="Optional lon/lat interpolation grid.")
    parser.add_argument("-c", "--compression", type=int, default=0, help="NetCDF compression level 0-9.")
    parser.add_argument("--decompose_multi", action="store_true", help="Forward --decompose_multi to extractor.")
    parser.add_argument("--w2nc-root", default="./w2nc", help="Root directory for fixed output tree.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run extractor even when the fixed output file already exists.",
    )
    parser.add_argument(
        "--extract-script",
        default=None,
        help="Path to extract_wrf_to_nc.py. Defaults to ./extract_wrf_to_nc.py beside this script.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running extractor.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    script_dir = Path(__file__).resolve().parent
    extract_script = Path(args.extract_script).expanduser() if args.extract_script else script_dir / "extract_wrf_to_nc.py"
    if not extract_script.exists():
        print(f"Error: extractor not found: {extract_script}", file=sys.stderr)
        return 2

    input_key = input_path_key(args.input_dir)
    variables = split_csv(args.variable)
    levels = split_csv(args.levels)
    times = split_csv(args.times)
    members = split_csv(args.ensemble)

    if not all([variables, levels, times, members]):
        print("Error: -V, -L, -T, and -E must each contain at least one value.", file=sys.stderr)
        return 2

    total = len(variables) * len(levels) * len(times) * len(members)
    count = 0

    for var in variables:
        for level in levels:
            for time_value in times:
                for member in members:
                    count += 1
                    output_dir = (
                        Path(args.w2nc_root)
                        / input_key
                        / safe_path_part(var)
                        / safe_path_part(args.domain)
                        / safe_path_part(level)
                        / member_output_dir(member)
                    )
                    output_file = f"{time_to_digits(time_value)}.nc"
                    target_file = output_dir / output_file

                    cmd = [
                        sys.executable,
                        str(extract_script),
                        "-i",
                        args.input_dir,
                        "-V",
                        var,
                        "-L",
                        level,
                        "-T",
                        time_value,
                        "-E",
                        member_for_extract(member),
                        "-D",
                        args.domain,
                        "-n",
                        str(output_dir),
                        "-o",
                        output_file,
                        "-c",
                        str(args.compression),
                    ]
                    if args.decompose_multi:
                        cmd.append("--decompose_multi")
                    if args.lonlat_grid:
                        cmd.extend(["-LL", args.lonlat_grid])

                    print(f"[{count}/{total}] {var} / {level} / {time_value} / {member}")
                    print(f"Output: {target_file}")

                    if target_file.exists() and not args.overwrite:
                        print(f"Skip existing output: {target_file}")
                        continue

                    if args.dry_run:
                        print("Command:")
                        print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
                        continue

                    output_dir.mkdir(parents=True, exist_ok=True)
                    started_at = time.time()
                    result = subprocess.run(cmd)
                    if result.returncode != 0:
                        return result.returncode
                    normalize_output_name(target_file, started_at)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
