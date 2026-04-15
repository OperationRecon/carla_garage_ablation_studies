
class ActivationExtractor:
  """Extracts and stores activations from model layers"""
  def __init__(self, model, config):
    self.model = model
    self.config = config
    self.activations = {}
    self.hooks = []
    self.register_hooks()

  def register_hooks(self):
    """Register forward hooks on all layers of the backbone"""
    backbone = self.model.module.backbone if hasattr(self.model, 'module') else self.model.backbone

    # Hook into image encoder layers
    if hasattr(backbone, 'image_encoder'):
      for name, module in backbone.image_encoder.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.MaxPool2d)):
          hook = module.register_forward_hook(self.create_hook(f"image_encoder.{name}"))
          self.hooks.append(hook)

    # Hook into transformer fusion layers
    if hasattr(backbone, 'transformers'):
      for idx, transformer in enumerate(backbone.transformers):
        # Hook into GPT blocks
        for block_idx, block in enumerate(transformer.blocks):
          for name, module in block.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Dropout)):
              hook = module.register_forward_hook(self.create_hook(f"transformer_{idx}.block_{block_idx}.{name}"))
              self.hooks.append(hook)

    # Hook into lidar encoder if it's not video-based
    if hasattr(backbone, 'lidar_encoder') and not getattr(backbone, 'lidar_video', False):
      for name, module in backbone.lidar_encoder.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.MaxPool2d)):
          hook = module.register_forward_hook(self.create_hook(f"lidar_encoder.{name}"))
          self.hooks.append(hook)

    # Hook into fusion layers
    if hasattr(backbone, 'lidar_channel_to_img') and hasattr(backbone, 'img_channel_to_lidar'):
      for i in range(4):
        hook_img = backbone.img_channel_to_lidar[i].register_forward_hook(self.create_hook(f"fusion_lidar_to_img_{i}"))
        hook_lidar = backbone.lidar_channel_to_img[i].register_forward_hook(self.create_hook(f"fusion_img_to_lidar_{i}"))
        self.hooks.append(hook_img)
        self.hooks.append(hook_lidar)

    # Hook into global pooling layers
    if hasattr(backbone, 'global_pool_img'):
      hook = backbone.global_pool_img.register_forward_hook(self.create_hook("global_pool_img"))
      self.hooks.append(hook)
    if hasattr(backbone, 'global_pool_lidar'):
      hook = backbone.global_pool_lidar.register_forward_hook(self.create_hook("global_pool_lidar"))
      self.hooks.append(hook)

  def create_hook(self, name):
    """Create a hook function that captures activations"""
    def hook(module, input, output):
      if name not in self.activations:
        self.activations[name] = []
      # Store output activation
      if isinstance(output, torch.Tensor):
        self.activations[name].append(output.detach().cpu().clone())
      elif isinstance(output, tuple):
        self.activations[name].append(tuple(o.detach().cpu().clone() if isinstance(o, torch.Tensor) else o for o in output))
    return hook

  def clear_activations(self):
    """Clear stored activations"""
    self.activations = {}

  def remove_hooks(self):
    """Remove all registered hooks"""
    for hook in self.hooks:
      hook.remove()

import argparse
import json
import os
import pathlib
import datetime
import random
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp


from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data
from plant import PlanT



# ...existing code...

# All logic should be in main()

class ActivationVisualizer:
  """Visualizes and saves activations alongside input images"""
  
  def __init__(self, save_dir, config=None, overlay_only=False):
    self.save_dir = save_dir
    self.config = config
    self.overlay_only = overlay_only
    os.makedirs(save_dir, exist_ok=True)
    # Save config metadata for reference
    if config is not None:
      import json
      with open(os.path.join(save_dir, "config_metadata.json"), "w") as f:
        json.dump({
          "image_architecture": getattr(config, "image_architecture", None),
          "lidar_architecture": getattr(config, "lidar_architecture", None),
          "n_layer": getattr(config, "n_layer", None),
          "n_head": getattr(config, "n_head", None),
          "img_vert_anchors": getattr(config, "img_vert_anchors", None),
          "img_horz_anchors": getattr(config, "img_horz_anchors", None),
          "lidar_vert_anchors": getattr(config, "lidar_vert_anchors", None),
          "lidar_horz_anchors": getattr(config, "lidar_horz_anchors", None),
          "use_semantic": getattr(config, "use_semantic", None),
          "use_depth": getattr(config, "use_depth", None),
          "use_bev_semantic": getattr(config, "use_bev_semantic", None),
          "add_features": getattr(config, "add_features", None),
          "transformer_decoder_join": getattr(config, "transformer_decoder_join", None),
        }, f, indent=2)
  
  def save_activations(self, batch_idx, rgb, lidar, activations, sample_indices=None):
    """Save activations and input images, with config-driven annotation"""
    batch_size = rgb.shape
    config = self.config
    for b in range(batch_size[0]):
      if sample_indices is not None and b not in sample_indices:
        continue
      sample_dir = os.path.join(self.save_dir, f"batch_{batch_idx:04d}_sample_{b:02d}")
      os.makedirs(sample_dir, exist_ok=True)
      # Save input RGB image
      rgb_img = rgb[b].cpu().numpy()
      # Handle (C, H, W) or (H, W, C)
      if rgb_img.shape[0] == 3 and len(rgb_img.shape) == 3:
        # (C, H, W) -> (H, W, C)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
      # Try to denormalize if common normalization is detected
      # 1. If values in [-1, 1], map to [0, 255]
      if rgb_img.min() < 0:
        rgb_img = ((rgb_img + 1.0) / 2.0) * 255.0
      # 2. If values in [0, 1], map to [0, 255]
      elif rgb_img.max() <= 1.0:
        rgb_img = rgb_img * 255.0
      # 3. If already in [0, 255], do nothing
      # Print min/max for debugging
      print(f"[Debug] RGB image min: {rgb_img.min()}, max: {rgb_img.max()}, shape: {rgb_img.shape}")
      rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
      import cv2
      # keep a BGR copy for overlay operations
      rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(sample_dir, "input_rgb.png"), rgb_bgr)###
      # Save config metadata for this sample
      if config is not None:
        with open(os.path.join(sample_dir, "config_info.txt"), "w") as f:
          f.write(f"image_architecture: {getattr(config, 'image_architecture', None)}\n")
      # if overlay-only mode, we still want overlays created later but skip channel save loops
          f.write(f"lidar_architecture: {getattr(config, 'lidar_architecture', None)}\n")
          f.write(f"n_layer: {getattr(config, 'n_layer', None)}\n")
          f.write(f"n_head: {getattr(config, 'n_head', None)}\n")
          f.write(f"img_vert_anchors: {getattr(config, 'img_vert_anchors', None)}\n")
          f.write(f"img_horz_anchors: {getattr(config, 'img_horz_anchors', None)}\n")
          f.write(f"lidar_vert_anchors: {getattr(config, 'lidar_vert_anchors', None)}\n")
          f.write(f"lidar_horz_anchors: {getattr(config, 'lidar_horz_anchors', None)}\n")
          f.write(f"use_semantic: {getattr(config, 'use_semantic', None)}\n")
          f.write(f"use_depth: {getattr(config, 'use_depth', None)}\n")
          f.write(f"use_bev_semantic: {getattr(config, 'use_bev_semantic', None)}\n")
          f.write(f"add_features: {getattr(config, 'add_features', None)}\n")
          f.write(f"transformer_decoder_join: {getattr(config, 'transformer_decoder_join', None)}\n")
      # Save activations from each layer
      for layer_name, layer_activations in activations.items():
        if len(layer_activations) > 0:
          activation = layer_activations[-1]
          if isinstance(activation, torch.Tensor):
            act_np = activation.numpy()
            # grab the sample-specific activation (some hooks return [B,...])
            if act_np.ndim > 0 and act_np.shape[0] > b:
              sample_act = act_np[b]
            else:
              sample_act = act_np

            # Annotate image filenames with config-driven info
            layer_prefix = f"{layer_name}"
            if config is not None:
              if hasattr(config, 'image_architecture') and 'image_encoder' in layer_name:
                layer_prefix = f"{config.image_architecture}_{layer_name}"
              if hasattr(config, 'lidar_architecture') and 'lidar_encoder' in layer_name:
                layer_prefix = f"{config.lidar_architecture}_{layer_name}"

            # generate overlay heatmap for rgb-side layers only
            if layer_name.startswith("image_encoder"):
              # reduce across channel dimension by max
              heatmap = None
              if isinstance(sample_act, np.ndarray):
                if sample_act.ndim == 3:
                  heatmap = sample_act.max(axis=0)
                elif sample_act.ndim == 2:
                  heatmap = sample_act
              if heatmap is not None:
                # normalize
                hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                hm_resized = cv2.resize(hm_norm, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                hm_color = cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(rgb_bgr, 0.6, hm_color, 0.4, 0)
                cv2.imwrite(os.path.join(sample_dir, f"{layer_prefix}_overlay.png"), overlay)

            # if not running overlay-only mode, save individual channels
            if not self.overlay_only:
              if len(act_np.shape) == 4:
                channels_to_save = min(8, act_np.shape[0])
                for c in range(channels_to_save):
                  channel_img = act_np[c]
                  channel_img = ((channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-8) * 255).astype(np.uint8)
                  if len(channel_img.shape) == 2 or (len(channel_img.shape) == 3 and channel_img.shape[2] in [1, 3, 4]):
                    cv2.imwrite(os.path.join(sample_dir, f"{layer_prefix}_channel_{c:02d}.png"), channel_img)
                  else:
                    print(f"[Warning] Skipping activation save for {layer_prefix} channel {c} with shape {channel_img.shape}")
              elif len(act_np.shape) == 2:
                channel_img = ((act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-8) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(sample_dir, f"{layer_prefix}.png"), channel_img)
              elif len(act_np.shape) == 3:
                if act_np.shape[0] <= 4 and act_np.shape[1] > 4 and act_np.shape[2] > 4:
                  channels_to_save = min(8, act_np.shape[0])
                  for c in range(channels_to_save):
                    channel_img = act_np[c]
                    channel_img = ((channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-8) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(sample_dir, f"{layer_prefix}_channel_{c:02d}.png"), channel_img)
                elif act_np.shape[2] == 3 or act_np.shape[2] == 4:
                  channel_img = ((act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-8) * 255).astype(np.uint8)
                  cv2.imwrite(os.path.join(sample_dir, f"{layer_prefix}.png"), cv2.cvtColor(channel_img, cv2.COLOR_RGB2BGR))
                else:
                  channel_img = act_np[0]
                  channel_img = ((channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-8) * 255).astype(np.uint8)
                  cv2.imwrite(os.path.join(sample_dir, f"{layer_prefix}_channel_00.png"), channel_img)
  
  def save_activations_to_numpy(self, batch_idx, activations, save_path):
    """Save all activations as numpy arrays for later analysis"""
    activations_dict = {}
    for layer_name, layer_activations in activations.items():
      if len(layer_activations) > 0:
        # Convert all activations to numpy
        activations_dict[layer_name] = [act.numpy() if isinstance(act, torch.Tensor) else act for act in layer_activations]
    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
      os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(save_path, **activations_dict)


def main():
  torch.cuda.empty_cache()

  # Loads the default values for the argparse so we have only one default
  config = GlobalConfig()

  parser = argparse.ArgumentParser()
  parser.add_argument('--id', type=str, default=None, help='Unique experiment identifier.')
  parser.add_argument('--logdir', type=str, required=True, help='Directory to log data and models to.')
  parser.add_argument('--load_file', type=str, default=None, help='Model to load for initialization.')
  parser.add_argument('--root_dir', type=str, required=True, nargs='+', help='Root directory of your training data')
  parser.add_argument('--cpu_cores', type=int, required=True, help='How many cpu cores are available on the machine.')
  parser.add_argument('--sample_indices', type=str, default=None, help='Comma-separated list of sample indices to process.')
  parser.add_argument('--save_activations', type=str, default=None, help='Directory to save activations.')
  parser.add_argument('--save_activations_np', type=str, default=None, help='Directory to save activations as numpy arrays.')
  parser.add_argument('--overlay_only', type=int,  default=None, help='Only save overlay heatmap images (skip raw channel dumps).')
  parser.add_argument('--batch_size', type=int, default=1, help='Batch size for activation extraction.')
  parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to process.')
  parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
  parser.add_argument('--config_json', type=str, default=None, help='Path to config.json file.')

  args = parser.parse_args()

  # Load config from JSON if provided, else use default
  if args.config_json is not None:
    with open(args.config_json, 'r') as f:
      config_dict = json.load(f)
    config.initialize(**config_dict)

  # Set random seed for reproducibility
  if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

  # Set up device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Initialize model
  if config.use_plant:
    model = PlanT(config)
  else:
    model = LidarCenterNet(config)

  # Load model weights
  if args.load_file is not None:
    model.load_state_dict(torch.load(args.load_file, map_location=device), strict=False)

  model = model.to(device)
  model.eval()
  print("roots used: ", config.data_roots)
  # Set up data loader
  train_set = CARLA_Data(root=config.data_roots, config=config, validation=False)
  dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

  # Initialize activation extractor
  extractor = ActivationExtractor(model, config)

  # Initialize visualizer with config
  visualizer = ActivationVisualizer(args.save_activations, config=config, overlay_only=args.overlay_only) if args.save_activations else None

  # Process batches
  batch_idx = 0
  sample_indices = None
  if args.sample_indices is not None:
    sample_indices = [int(x) for x in args.sample_indices.split(',')]

  for batch in tqdm(dataloader, desc="Processing batches"):
    if batch_idx >= args.num_batches:
      break

    checkpoint = batch['route'][:, :config.predict_checkpoint_len].to(device, dtype=torch.float32)
    rgb = batch['rgb'].to(device, dtype=torch.float32)
    rgb = rgb[:, [1,2,0], :, :]

    if config.use_semantic:
        semantic_label = batch['semantic'].to(device, dtype=torch.long)
    else:
        semantic_label = None
    if config.use_bev_semantic:
        bev_semantic_label = batch['bev_semantic'].to(device, dtype=torch.long)
    else:
        bev_semantic_label = None
    if config.use_depth:
        depth_label = batch['depth'].to(device, dtype=torch.float32)
    else:
        depth_label = None
    if config.lidar_seq_len > 1:
        lidar = batch['temporal_lidar'].to(device, dtype=torch.float32)
    else:
        lidar = batch['lidar'].to(device, dtype=torch.float32)

    target_point = batch['target_point'].to(device, dtype=torch.float32)
    target_point_next = batch['target_point_next'].to(device, dtype=torch.float32)
    command = batch['command'].to(device, dtype=torch.float32)

    ego_vel = batch['speed'].to(device, dtype=torch.float32).unsqueeze(1)
    # Extract activations
    with torch.no_grad():
      # Clear previous activations
      extractor.clear_activations()

      pred_wp,\
        pred_target_speed,\
        pred_checkpoint,\
        pred_semantic, \
        pred_bev_semantic, \
        pred_depth, \
        pred_bounding_box, _, \
        pred_wp_1, \
        selected_path = model(rgb=rgb,
                            lidar_bev=lidar,
                            target_point=target_point,
                            ego_vel=ego_vel,
                            command=command,
                            target_point_next=target_point_next if config.two_tp_input else None,)


      # Get activations
      activations = extractor.activations

      # Save visualizations
      if visualizer is not None:
        visualizer.save_activations(batch_idx, rgb, lidar, activations, sample_indices)

      # Save numpy arrays
      if args.save_activations_np is not None:
        save_path = os.path.join(args.save_activations_np, f"batch_{batch_idx:04d}.npz")
        visualizer.save_activations_to_numpy(batch_idx, activations, save_path)

    batch_idx += 1

  # Clean up
  extractor.remove_hooks()
  print("Activation extraction complete.")


if __name__ == '__main__':
  # Select how the threads in the data loader are spawned
  available_start_methods = mp.get_all_start_methods()
  if 'fork' in available_start_methods:
    mp.set_start_method('fork')
  # Available on all OS.
  elif 'spawn' in available_start_methods:
    mp.set_start_method('spawn')
  elif 'forkserver' in available_start_methods:
    mp.set_start_method('forkserver')
  print('Start method of multiprocessing:', mp.get_start_method())

  main()