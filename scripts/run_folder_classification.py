from app.configs.config import AppConfig
from app.use_cases.classify_detail import RecognitionPipeline


def run_folder_classification():
    config = AppConfig()
    pipeline = RecognitionPipeline(config)
    pipeline.process_input_dir_multi()


if __name__ == "__main__":
    run_folder_classification()
