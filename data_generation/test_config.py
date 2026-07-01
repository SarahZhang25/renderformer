import json
from dacite import from_dict, Config
import sys
import os

# Add parent dir to path to import scene_config
sys.path.append('/home/sazhang/Neural-Radiosity-Renderer/renderformer/scene_processor')
from scene_config import SceneConfig

test_json = {
    "scene_name": "test",
    "version": "1.0",
    "objects": {
        "test_obj": {
            "mesh_path": "fake.obj",
            "transform": {
                "translation": [0,0,0],
                "rotation": [0,0,0],
                "scale": [1,1,1],
                "normalize": True
            },
            "material": {
                "diffuse": [1,1,1],
                "specular": [0,0,0],
                "roughness": 0.5,
                "emissive": [0,0,0],
                "smooth_shading": False,
                "random_diffuse_type": "procedural",
                "procedural_pattern": "checkerboard",
                "procedural_frequency": [10, 10, 10],
                "procedural_color_a": [1, 0, 0],
                "procedural_color_b": [0, 0, 1]
            }
        }
    },
    "cameras": []
}

try:
    config = from_dict(data_class=SceneConfig, data=test_json, config=Config(check_types=True, strict=True))
    print("Successfully parsed SceneConfig!")
    print(config.objects["test_obj"].material)
except Exception as e:
    print(f"Error parsing SceneConfig: {e}")
