import yaml


def override_config_variables(config_path, overrides):
    # Read the config.yaml file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Update the variables with the overrides
    for key, value in overrides.items():
        if key in config:
            config[key] = value

    # Write the updated configuration back to the file
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content
