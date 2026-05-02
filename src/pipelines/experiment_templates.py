from copy import deepcopy


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_experiment_template(config: dict, template_name: str | None) -> dict:
    experiments = config.get("experiments", {})
    templates = experiments.get("templates", {})
    selected_name = template_name or experiments.get("default_template", "baseline")
    template = templates.get(selected_name, {})

    default_models = list(config.get("models", {}).keys())
    models = template.get("models", default_models)
    tuning_enabled = bool(template.get("tuning_enabled", False))
    model_overrides = template.get("model_overrides", {})

    merged_models = {}
    for model_name in default_models:
        merged_models[model_name] = dict(config.get("models", {}).get(model_name, {}))
    for model_name, overrides in model_overrides.items():
        if model_name not in merged_models:
            merged_models[model_name] = {}
        merged_models[model_name] = _deep_merge(merged_models[model_name], overrides)

    return {
        "name": selected_name,
        "models": models,
        "tuning_enabled": tuning_enabled,
        "model_configs": merged_models,
    }


def get_tuning_grid(config: dict) -> dict:
    tuning = config.get("tuning", {})
    return tuning.get("grids", {})


def get_tuning_defaults(config: dict) -> dict:
    tuning = config.get("tuning", {})
    return {
        "enabled": bool(tuning.get("enabled", False)),
        "max_trials_per_model": int(tuning.get("max_trials_per_model", 8)),
    }
