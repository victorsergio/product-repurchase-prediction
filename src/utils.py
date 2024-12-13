import os
import json


def save_to_json(objects_dict, output_dir_path):
    for name, obj in objects_dict.items():
        save_path = os.path.join(output_dir_path, f"{name}.json")

        with open(save_path, 'w') as f:
            json.dump(obj, f, indent=4)
