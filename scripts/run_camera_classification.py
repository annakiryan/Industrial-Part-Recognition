import cv2

from app.core.camera import CameraSource, remove_black_borders
from app.configs.config import AppConfig
from app.use_cases.classify_detail import RecognitionPipeline


def run_camera_classification():
    config = AppConfig()
    pipeline = RecognitionPipeline(config)

    camera = CameraSource(
        camera_index=1,
        width=2000,
        height=2000,
    )
    camera.open()

    try:
        while True:
            frame = camera.read()
            frame = remove_black_borders(frame)

            preview = frame.copy()
            cv2.putText(
                preview,
                "Press S to scan, ESC to exit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Camera", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key in (ord("s"), ord("S")):
                result = pipeline.process_frame_multi(frame)

                print("=" * 60)
                print("num_objects_found:", result["num_objects_found"])
                print("counts_by_label:", result["counts_by_label"])
                print("objects:")

                for obj in result["objects"]:
                    print(
                        f"  object_id={obj['object_id']} | "
                        f"predicted_label={obj['predicted_label']} | "
                        f"best_dwg_path={obj['best_dwg_path']} | "
                        f"candidate_labels={obj['candidate_labels']}"
                    )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_classification()
