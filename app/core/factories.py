from app.core.template_classification import TemplateClassifier
from app.configs.config import AppConfig


def build_template_classifier(config: AppConfig) -> TemplateClassifier:
    return TemplateClassifier(
        templates_dir=config.paths.templates_dir,
        cache_dir=config.classification.cache_dir,
        template_cache_meta=config.classification.template_cache_meta,
        cache_npz_path=config.classification.cache_npz_path,
        template_features_config=config.classification.template_features_config,
        metric=config.classification.metric,
        normalize=config.classification.normalize,
    )