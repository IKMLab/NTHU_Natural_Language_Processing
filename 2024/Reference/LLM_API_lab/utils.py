import yaml


def load_prompts(prompts_path: str) -> dict:
    with open(prompts_path, "r") as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    return prompts
