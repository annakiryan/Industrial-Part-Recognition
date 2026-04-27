from pathlib import Path
import shutil
import subprocess


def convert_dwg_to_dxf(
    oda_exe: Path,
    input_dir: Path,
    output_dir: Path,
    version: str = "ACAD2018",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(oda_exe),
        str(input_dir),
        str(output_dir),
        version,
        "DXF",
        "0",  # recursive
        "1",  # audit
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Error ODA File Converter\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def convert_single_dwg_to_dxf(
    oda_exe: Path,
    input_dwg_path: Path,
    output_dxf_path: Path,
    version: str = "ACAD2018",
) -> None:
    if not oda_exe.exists():
        raise FileNotFoundError(f"ODA File Converter not found: {oda_exe}")

    if not input_dwg_path.exists():
        raise FileNotFoundError(f"DWG file not found: {input_dwg_path}")

    if input_dwg_path.suffix.lower() != ".dwg":
        raise ValueError(f"Expected .dwg file, got: {input_dwg_path.suffix}")

    input_dir = input_dwg_path.parent
    temp_output_dir = input_dwg_path.parent / "_tmp_dxf"

    temp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        convert_dwg_to_dxf(
            oda_exe=oda_exe,
            input_dir=input_dir,
            output_dir=temp_output_dir,
            version=version,
        )

        produced_dxf = temp_output_dir / f"{input_dwg_path.stem}.dxf"
        if not produced_dxf.exists():
            raise FileNotFoundError(f"Converted DXF was not created: {produced_dxf}")

        output_dxf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced_dxf), str(output_dxf_path))

    finally:
        shutil.rmtree(temp_output_dir, ignore_errors=True)
