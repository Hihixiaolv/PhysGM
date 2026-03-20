import argparse
import os
import json
import shutil
import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from contextlib import nullcontext

from utils import create_logger
from data.dataset_infer import Dataset
from model.physgm import PhysGM

MAT_MAP = {
    "Wood": "metal",
    "Metal": "metal",
    "Plastic": "metal",
    "Glass": "metal",
    "Fabric": "foam",
    "Leather": "foam",
    "Ceramic": "metal",
    "Stone": "metal",
    "Rubber": "jelly",
    "Paper": "foam",
    "Sand": "sand",
    "Snow": "snow",
    "Plasticine": "plasticine",
    "Foam": "foam",
}


def run_pipeline(args):
    E_MEAN, E_STD = 7.387210, 2.456477
    NU_MEAN, NU_STD = 0.398, 0.111
    CLASS_TO_MATERIAL = {
        0: "Wood",
        1: "Metal",
        2: "Plastic",
        3: "Glass",
        4: "Fabric",
        5: "Leather",
        6: "Ceramic",
        7: "Stone",
        8: "Rubber",
        9: "Paper",
        10: "Sand",
        11: "Snow",
        12: "Plasticine",
        13: "Foam",
    }

    with open(args.config, "r", encoding="utf-8") as f:
        config = edict(yaml.safe_load(f))
    config.evaluation = True

    device = args.device
    model = PhysGM(config, device=device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval()

    dataset = Dataset(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    autocast_cm = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if "cuda" in device
        else nullcontext()
    )

    scene_dir = os.path.join(args.output_dir, args.scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(dataloader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with autocast_cm:
            ret_dict = model(batch)

        E_mu_norm = ret_dict["E_mu"][0].float().item() if "E_mu" in ret_dict else 0.0
        E_value = round((10 ** (E_mu_norm * E_STD + E_MEAN)) * 0.1, 2)
        nu_mu_norm = ret_dict["nu_mu"][0].float().item() if "nu_mu" in ret_dict else 0.0
        nu_value = round(nu_mu_norm * NU_STD + NU_MEAN, 4)
        mat_idx = torch.argmax(ret_dict["phys_logits"][0].float()).item()
        material_raw = CLASS_TO_MATERIAL.get(mat_idx, "Unknown")

        phys_data = {"E": E_value, "nu": nu_value, "material": material_raw}
        with open(os.path.join(scene_dir, "predicted_phys.json"), "w") as f:
            json.dump(phys_data, f, indent=4)

        model.save_visualization(
            batch, ret_dict, scene_dir, save_gaussian=True, save_video=False
        )
        for root, _, files in os.walk(scene_dir):
            for f in files:
                if f.endswith(".ply") and f != "point_clouds.ply":
                    shutil.move(
                        os.path.join(root, f),
                        os.path.join(scene_dir, "point_clouds.ply"),
                    )

        print(
            f"Inference Finished: E={E_value}, nu={nu_value}, material={material_raw}"
        )

    with open(args.template_config, "r") as f:
        sim_config = json.load(f)

    sim_config["E"] = E_value
    sim_config["nu"] = nu_value
    sim_config["material"] = MAT_MAP.get(material_raw, "jelly")

    merged_config_path = os.path.join(scene_dir, "sim_config_merged.json")
    with open(merged_config_path, "w") as f:
        json.dump(sim_config, f, indent=4)

    sim_out_dir = os.path.join(scene_dir, "simulation")
    os.makedirs(sim_out_dir, exist_ok=True)

    sim_cmd = (
        f"python gs_simulation.py "
        f"--model_path '{scene_dir}' "
        f"--output_path '{sim_out_dir}' "
        f"--config '{merged_config_path}' "
        f"--render_img --compile_video --white_bg"
    )

    exit_code = os.system(sim_cmd)
    if exit_code == 0:
        print(f"\nPipeline Success! Result at: {sim_out_dir}")
    else:
        print("\nSimulation Failed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--scene-name", type=str, required=True)
    parser.add_argument("--amp-dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template-config", type=str, required=True)

    args = parser.parse_args()
    run_pipeline(args)
