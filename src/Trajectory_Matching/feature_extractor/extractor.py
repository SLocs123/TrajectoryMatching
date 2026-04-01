import cv2
import torch
import time
import threading
import numpy as np

# import sys
# sys.path.append("./feature_extractor/fast_reid/fastreid")

from .fastreid.config.config import get_cfg
from .fastreid.engine import DefaultPredictor

from .config import FeatureExtrator


class Extractor:
    def __init__(self, config: FeatureExtrator) -> None:
        self.config = config
        self._setup()
  
    def _setup(self):
        cfg = get_cfg()
        cfg.merge_from_file(self.config.reid_config.reid_config_path)
        cfg.MODEL.WEIGHTS = self.config.reid_config.reid_weight
        cfg.MODEL.DEVICE = self.config.reid_config.reid_device
        cfg.freeze()
        self.model = DefaultPredictor(cfg)

    def _create_input4embed(self, trajectory_dict):
        '''
        Create input for embedding from the image dictionary.
        img_dict: Dict[track_id: List[(image, confidence, frame_id)]]
        Returns a dictionary with track_id as keys and averaged features, features, and frame_ids as values.
        '''
        
        cfg = self.model.cfg
        h, w = cfg.INPUT.SIZE_TEST
        device = torch.device(self.config.reid_config.reid_device)

        keys = list(trajectory_dict.keys()) # define key list to make double sure order is kept, no reason for it not to but doesnt hurt
        img_list = []
        for key in keys:
            imgs = trajectory_dict[key]['crops']  # Assuming trajectory_dict[key] is a list of (image, confidence, frame_id, ...)
            for img in imgs:
                img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                img = torch.from_numpy(img.astype("float32").transpose(2, 0, 1))  # CHW, torch.Tensor
                img_list.append(img)

        if len(img_list) == 0:
            return trajectory_dict

        batch = torch.stack(img_list, dim=0).to(device)  # BCHW, float32
        # conf_tensor = torch.tensor(conf_list, dtype=batch.dtype, device=device)  # (batch_size,)

        with torch.no_grad():
            feats = self.model(batch)  # The feats should be push into CPU already, but i changed the fastreid code. return predictions.cpu() to return predictions


        for t_idx, key in enumerate(keys):
            base = 2 * t_idx
            trajectory_dict[key]['start_appearance'] = feats[base].detach().cpu().numpy()
            trajectory_dict[key]['end_appearance']   = feats[base + 1].detach().cpu().numpy()

        return trajectory_dict