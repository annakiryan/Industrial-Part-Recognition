from pathlib import Path

from app.configs.config import AppConfig
from app.use_cases.add_template import TemplateAddUseCase
from app.core.utils import make_template_name_from_file


def add_templates_from_queue(
    overwrite: bool = False,
) -> list[Path]:
    config = AppConfig()
    use_case = TemplateAddUseCase(config)

    input_dir = Path(config.paths.new_templates_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    dwg_files = sorted(input_dir.glob("*.dwg"))
    if not dwg_files:
        print(f"No DWG files found in: {input_dir}")
        return

    print(f"Found {len(dwg_files)} DWG file(s) in: {input_dir}")

    success_count = 0
    error_count = 0

    for dwg_path in dwg_files:
        try:
            template_name = make_template_name_from_file(dwg_path)

            use_case.add_from_dwg(
                dwg_path=dwg_path,
                template_name=template_name,
                overwrite=overwrite,
            )

            dwg_path.unlink()

            success_count += 1

        except Exception as ex:
            error_count += 1
            print(f"[ERROR] {dwg_path.name}: {ex}")

    remaining_files = sorted(input_dir.glob("*.dwg"))

    return remaining_files
