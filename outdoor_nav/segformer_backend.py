"""SegFormer (HF transformers) road segmentation backend.

Use with ``backend: outdoor_nav.segformer_backend:create_backend``; ``weights`` is a
local model directory (config.json + model weights), e.g. a downloaded snapshot of
``nvidia/segformer-b0-finetuned-cityscapes-1024-1024``.

A fine-tuned model with extra classes works the same way: point ``road_class_ids`` at its
road labels and, optionally, ``stairs_class_ids`` at its stairs labels. Then ``has_stairs``
is True and after each ``predict`` the backend keeps ``last_stairs`` (bool HxW, before the
confidence threshold).
"""
import numpy as np

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
OPTIONS = {'input_scale', 'road_class_ids', 'stairs_class_ids'}


def _options(options):
    if set(options) - OPTIONS:
        raise ValueError('Unknown SegFormer backend options; allowed: ' + ', '.join(sorted(OPTIONS)))
    scale = options.get('input_scale', 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 4:
        raise ValueError('SegFormer input_scale must be a number in (0, 4]')
    ids = options.get('road_class_ids', [0, 1])  # Cityscapes: 0 road, 1 sidewalk
    if not _id_list(ids) or not ids:
        raise ValueError('SegFormer road_class_ids must be a non-empty list of class indices')
    stairs = options.get('stairs_class_ids', [])
    if not _id_list(stairs) or set(stairs) & set(ids):
        raise ValueError('SegFormer stairs_class_ids must be a list of class indices not in road_class_ids')
    return float(scale), ids, stairs


def _id_list(ids):
    return isinstance(ids, list) and all(not isinstance(i, bool) and isinstance(i, int) and i >= 0 for i in ids)


class SegformerRoad:
    def __init__(self, model, device, input_scale=1.0, road_class_ids=(0, 1), stairs_class_ids=()):
        import torch
        self.torch, self.device = torch, torch.device(device)
        self.model = model.to(self.device).eval()
        self.scale = input_scale
        self.road = torch.tensor(sorted(set(road_class_ids)), device=self.device)
        self.stairs = torch.tensor(sorted(set(stairs_class_ids)), device=self.device, dtype=torch.long)
        self.has_stairs = len(stairs_class_ids) > 0
        self.last_stairs = None
        num = getattr(getattr(model, 'config', None), 'num_labels', None)
        if num is not None and max([*road_class_ids, *stairs_class_ids]) >= num:
            raise ValueError('SegFormer class ids exceed model num_labels')

    def _tensor(self, rgb):
        import cv2
        if self.scale != 1.0:
            rgb = cv2.resize(rgb, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        rgb = cv2.copyMakeBorder(rgb, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REPLICATE)
        x = (rgb.astype(np.float32) / 255 - MEAN) / STD
        return self.torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device), h, w

    def predict(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('SegFormer expects HxWx3 uint8 RGB')
        F = self.torch.nn.functional
        with self.torch.inference_mode():
            x, h, w = self._tensor(rgb)
            y = self.model(pixel_values=x).logits
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)[:, :, :h, :w]
            y = F.interpolate(y, size=rgb.shape[:2], mode='bilinear', align_corners=False)
            conf, cls = y.float().softmax(1).max(1)
            road = self.torch.isin(cls[0], self.road)
            if self.has_stairs:
                self.last_stairs = self.torch.isin(cls[0], self.stairs).cpu().numpy()
        mask = road.cpu().numpy().astype(np.uint8)
        return mask, conf[0].cpu().numpy().astype(np.float32)


def create_backend(*, weights, device, options):
    import torch
    from transformers import SegformerForSemanticSegmentation
    scale, ids, stairs = _options(options)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegformerForSemanticSegmentation.from_pretrained(weights, local_files_only=True)
    return SegformerRoad(model, device, scale, ids, stairs)
