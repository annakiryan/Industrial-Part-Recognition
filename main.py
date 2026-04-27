import sys

from scripts.add_templates_from_queue import add_templates_from_queue
from scripts.run_camera_classification import run_camera_classification
from scripts.run_folder_classification import run_folder_classification


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python main.py [add_templates|camera_classify|folder_classify]"
        )

    command = sys.argv[1]

    if command == "add_templates":
        remaining_files = add_templates_from_queue(overwrite=False)

        if remaining_files:
            print("Some files were not processed:")
            for path in remaining_files:
                print(f"  - {path.name}")
        else:
            print("All templates were added successfully.")

    elif command == "camera_classify":
        run_camera_classification()

    elif command == "folder_classify":
        run_folder_classification()

    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
