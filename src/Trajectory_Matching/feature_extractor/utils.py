import torch
import torch.nn as nn
import torchvision.transforms as T
from .reid_inference.reid_model import build_reid_model
import sys
sys.path.append('../')
from .reid_config import cfg


# onnx_path = "reid_model.onnx"
# def export_to_onnx(model, dummy_input, onnx_path):
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         export_params=True,
#         opset_version=11,  # Set compatible ONNX version
#         input_names=["input"],
#         output_names=["output"],
#         dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#     )
#     print(f"Model exported to ONNX at: {onnx_path}")
# def replace_batch_norm_with_identity(model):
#     """Replace all BatchNorm2d layers with Identity layers to bypass normalization."""
#     for name, module in model.named_children():
#         if isinstance(module, torch.nn.BatchNorm2d):
#             print(f"Replacing BatchNorm2d: {name}")
#             setattr(model, name, torch.nn.Identity())  # Replace BatchNorm with Identity
#         else:
#             replace_batch_norm_with_identity(module)  # Recursively replace in children

# def register_debug_hooks(model):
#     def forward_hook(module, input, output):
#         print(f"Layer: {module}, Input: {input[0].shape}, Output: {output.shape}")
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.BatchNorm2d):
#             module.register_forward_hook(forward_hook)
# def clean_batchnorm(model):
#     """Clean all BatchNorm2d operations from the model."""
#     for name, module in model.named_children():
#         if isinstance(module, nn.BatchNorm2d):
#             print(f"Replacing {name} (BatchNorm2d) -> Identity")
#             setattr(model, name, nn.Identity())
#         else:
#             clean_batchnorm(module)



class ReidFeature():
    """Extract reid feature."""

    def __init__(self, _mcmt_cfg):
        print("init reid model")
        self.model, self.reid_cfg = build_reid_model(_mcmt_cfg)
        device = torch.device('cuda') # thinking about how to deal with this in docker compose
        # Debug hooks to locate problematic BatchNorm
        # register_debug_hooks(self.model)

        # # Clean up all BatchNorm layers
        # clean_batchnorm(self.model)

        # The model is called in here through the build_reid_model function
        # Then the configuration file are here and send to a make_model() function

        self.model = self.model.to(device)
        
        self.model.eval()
        # dummy_input = torch.randn(1, 3, 384, 384).to(device)
        # export_to_onnx(self.model, dummy_input, "reid_model.onnx")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_transforms = T.Compose([T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),\
                              T.ToTensor(), T.Normalize(mean=mean, std=std)]) # T.Compose 串联起多个变换, they first do T.Resize, then T.ToTensor thn T.Normalize

    def extract(self, img_list):
        """Extract image feature with given image path.
        Feature shape (2048,) float32."""

        img_batch = []
        for cropped_img in img_list:
            img = self.val_transforms(cropped_img)
            img = img.unsqueeze(0) # Add one extra channel for the img, the previous one is img.shape = [channel, height,width], after the processing it will becoems img.shpe = [1,channel,height,width]
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0) # torch.cat is the function to concatenate imgs

        with torch.no_grad():
            img = img.to('cuda')
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == 'yes': flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda() 
                        # torch.arange returns [end-start/step] 1-D tensor
                        # img.size(3) returns the width of the img
                        # e.g. width = 100 it returns [99,98,97,...,1,0]
                        img = img.index_select(3, inv_idx)
                        # torch.index_select function is a function returns select_index correspond img
                        # so what this function do is to flip the img and then extract the featrue 
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy() # move features from gpu to cpu, and then detach() it(don't need the gradient) and convert it to an numpy array!
        return feat # Feature shape (2048,) float32.