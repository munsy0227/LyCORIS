import copy

import toml

from ..config_sdk import VALID_PRESET_KEYS, PresetConfig, PresetValidationError


def configure_preset(instance, preset, fields):
    """Snapshot legacy class defaults, then apply a factory's instance-local preset.

    Direct apply_preset() calls still configure class defaults. Factory calls do
    not modify those defaults, even when adapter construction raises an error.
    """
    if preset is not None:
        for key in preset:
            if key not in VALID_PRESET_KEYS:
                raise KeyError(
                    f'Unknown preset key "{key}". Valid keys: {VALID_PRESET_KEYS}'
                )
    for key, attribute in fields.items():
        value = (preset or {}).get(key, getattr(instance, attribute))
        setattr(instance, attribute, copy.deepcopy(value))


def read_preset(preset_path):
    try:
        raw_config = toml.load(preset_path)
    except Exception as e:
        print("Error: cannot read preset file. ", e)
        return None

    try:
        preset = PresetConfig.from_dict(raw_config)
    except PresetValidationError as exc:
        print(f"Error: invalid preset content ({exc}).")
        return None
    return preset.to_dict()
