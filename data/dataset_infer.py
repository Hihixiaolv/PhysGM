import os
import json
import random
import traceback
import numpy as np
import PIL.Image as Image
import cv2
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import itertools

class InfiniteDataLoader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = self._create_iterator()

    def _create_iterator(self):
        return itertools.cycle(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def __len__(self):
        return len(self.dataloader)

class Dataset(Dataset):

    # Material to class mapping
    MATERIAL_TO_CLASS = {
        "Wood": 0,
        "Metal": 1,
        "Plastic": 2,
        "Glass": 3,
        "Fabric": 4,
        "Leather": 5,
        "Ceramic": 6,
        "Stone": 7,
        "Rubber": 8,
        "Paper": 9,
        "Sand": 10,
        "Snow": 11,
        "Plasticine": 12,
        "Foam": 13
    }

    def __init__(self, config):
        self.config = config
        self.evaluation = config.get("evaluation", False)
        if self.evaluation and "data_eval" in config:
            self.config.data.update(config.data_eval)
        data_path_text = config.data.data_path
        data_folder = data_path_text.rsplit('/', 1)[0] 
        with open(data_path_text, 'r') as f:
            self.data_path = f.readlines()
        self.data_path = [x.strip() for x in self.data_path]
        self.data_path = [x for x in self.data_path if len(x) > 0]
        for i, data_path in enumerate(self.data_path):
            if not data_path.startswith("/"):
                self.data_path[i] = os.path.join(data_folder, data_path)

    def __len__(self):
        return len(self.data_path)

    def load_physical_class(self, data_dir):
        """
        Load physical class from phys.json file
        
        Args:
            data_dir: Directory containing the phys.json file
            
        Returns:
            physical_class: Integer class ID (0-13), or 0 if not found
        """
        phys_json_path = os.path.join(data_dir, "physical.json")
        
        try:
            if os.path.exists(phys_json_path):
                with open(phys_json_path, 'r') as f:
                    phys_data = json.load(f)
                    material = phys_data.get("material", None)
                    E_raw = phys_data.get("E", "0.0")
                    nu_raw = phys_data.get("nu", "0.0")

                    try:
                        E = float(E_raw) if E_raw is not None else 0.0
                    except (ValueError, TypeError):
                        print(f"Warning: Cannot convert E value '{E_raw}' to float, using 0.0")
                        E = 0.0
                    
                    try:
                        nu = float(nu_raw) if nu_raw is not None else 0.0
                    except (ValueError, TypeError):
                        print(f"Warning: Cannot convert nu value '{nu_raw}' to float, using 0.0")
                        nu = 0.0
                    
                    if material is not None:
                        physical_class = self.MATERIAL_TO_CLASS.get(material, 0)
                        return physical_class, E, nu
                    else:
                        print(f"Warning: 'material' field not found in {phys_json_path}")
                        return 0
            else:
                print(f"Warning: phys.json not found at {phys_json_path}")
                return 0
        except Exception as e:
            print(f"Error loading phys.json from {phys_json_path}: {e}")
            return 0


    def process_frames(self, frames, image_base_dir, random_crop_ratio=None):
        resize_h = self.config.data.get("resize_h", -1)
        resize_w = self.config.data.get("resize_w", -1)
        patch_size = self.config.model.patch_size
        patch_size = patch_size * 2 ** len(self.config.model.get("merge_layers", [])) 
        square_crop = self.config.data.square_crop
        random_crop = self.config.data.get("random_crop", 1.0)

        if "objaversexl" in image_base_dir:
            images = [Image.open(frame["file_path"]).convert("RGBA") for frame in frames]
        else:
            images = [Image.open(os.path.join(image_base_dir, frame["file_path"])).convert("RGBA") for frame in frames]
        images = np.stack([np.array(image) for image in images]) # (num_frames, H, W, 4) 或 (num_frames, H, W, 3)
        if resize_h == -1 and resize_w == -1:
            resize_h = images.shape[1]
            resize_w = images.shape[2]
        elif resize_h == -1:
            resize_h = int(resize_w / images.shape[2] * images.shape[1])
        elif resize_w == -1:
            resize_w = int(resize_h / images.shape[1] * images.shape[2])
        resize_h = int(round(resize_h / patch_size)) * patch_size
        resize_w = int(round(resize_w / patch_size)) * patch_size
        images = np.stack([cv2.resize(image, (resize_w, resize_h)) for image in images]) # (num_frames, resize_h, resize_w, 3)
        start_h, start_w = 0, 0 # 初始化
        if square_crop:
            min_size = min(resize_h, resize_w)
            # center crop
            start_h = (resize_h - min_size) // 2
            start_w = (resize_w - min_size) // 2
            images = images[:, start_h:start_h+min_size, start_w:start_w+min_size, :]
        images = images / 255.0
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() # (num_frames, 4, resize_h, resize_w)
        
        h = np.array([frame["h"] for frame in frames])
        w = np.array([frame["w"] for frame in frames])
        fx = np.array([frame["fx"] for frame in frames])
        fy = np.array([frame["fy"] for frame in frames])
        cx = np.array([frame["cx"] for frame in frames])
        cy = np.array([frame["cy"] for frame in frames])
        intrinsics = np.stack([fx, fy, cx, cy], axis=1) # (num_frames, 4)
        intrinsics = intrinsics.astype(np.float64)
        intrinsics[:, 0] *= resize_w / w
        intrinsics[:, 1] *= resize_h / h
        intrinsics[:, 2] *= resize_w / w
        intrinsics[:, 3] *= resize_h / h
        if square_crop:
            intrinsics[:, 2] -= start_w
            intrinsics[:, 3] -= start_h
        intrinsics = torch.from_numpy(intrinsics).float()

        # random crop
        if random_crop < 1.0:
            random_crop_ratio = np.random.uniform(random_crop, 1.0) if random_crop_ratio is None else random_crop_ratio
            magnify_ratio = 1.0 / random_crop_ratio
            cur_h, cur_w = images.shape[2], images.shape[3]
            images = F.interpolate(images, scale_factor=magnify_ratio, mode='bilinear', align_corners=False)
            mag_h, mag_w = images.shape[2], images.shape[3]
            start_h = (mag_h - cur_h) // 2
            start_w = (mag_w - cur_w) // 2
            images = images[:, :, start_h:start_h+cur_h, start_w:start_w+cur_w]
            intrinsics[:, 0] *= (mag_w / cur_w)
            intrinsics[:, 1] *= (mag_h / cur_h) 

        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws, random_crop_ratio


    def __getitem__(self, idx):
        try:
            data_path = self.data_path[idx]
            data_json = json.load(open(data_path, 'r'))
            scene_name = data_json['scene_name']
            frames = data_json['frames']
            image_base_dir = data_path.rsplit('/', 1)[0]


            num_input_frames = self.config.data.num_input_frames
            num_target_frames = self.config.data.get("num_target_frames", 0)
            target_has_input = self.config.data.target_has_input

            # get frame range
            frame_idx = list(range(len(frames)))
            input_frame_idx = [0,1,2,3]

            target_frame_idx = [0,1,2,3] 
            
            random_crop_ratio = None
            target_frames = [frames[i] for i in target_frame_idx]
            target_images, target_intr, target_c2ws, random_crop_ratio = self.process_frames(target_frames, image_base_dir)
     
            input_frames = [frames[i] for i in input_frame_idx]
            input_images, input_intr, input_c2ws, _ = self.process_frames(input_frames, image_base_dir, random_crop_ratio)

            # normalize input camera poses
            position_avg = input_c2ws[:, :3, 3].mean(0) # (3,)
            forward_avg = input_c2ws[:, :3, 2].mean(0) # (3,)
            down_avg = input_c2ws[:, :3, 1].mean(0) # (3,)
            # gram-schmidt process
            forward_avg = F.normalize(forward_avg, dim=0)
            down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
            right_avg = torch.linalg.cross(down_avg, forward_avg)
            pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
            pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)
            pos_avg_inv = torch.inverse(pos_avg)

            input_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), input_c2ws)
            target_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), target_c2ws)
     
            # scale scene size
            position_max = target_c2ws[:, :3, 3].abs().max()
            scene_scale = self.config.data.get("scene_scale", 1.0) * position_max
            scene_scale = 1.0 / scene_scale

            input_c2ws[:, :3, 3] *= scene_scale
            target_c2ws[:, :3, 3] *= scene_scale

            if torch.isnan(input_c2ws).any() or torch.isinf(input_c2ws).any():
                print("encounter nan or inf in input poses")
                assert False

            if torch.isnan(target_c2ws).any() or torch.isinf(target_c2ws).any():
                print("encounter nan or inf in target poses")
                assert False
     
            ret_dict = {
                "scene_name": scene_name,
                "input_images": input_images,
                "input_intr": input_intr,
                "input_c2ws": input_c2ws,
                "test_images": target_images,
                "test_intr": target_intr,
                "test_c2ws": target_c2ws,
                "pos_avg_inv": pos_avg_inv,
                "scene_scale": scene_scale,
                "input_frame_idx": torch.tensor(input_frame_idx).long(),
                "test_frame_idx": torch.tensor(target_frame_idx).long(),
            }
        except:
            traceback.print_exc()
            print(f"error loading data: {self.data_path[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return ret_dict



