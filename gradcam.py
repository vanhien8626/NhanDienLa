import numpy as np
import torch
from torch.nn import functional as F


def heatmap(model, image):
    model.eval()
    captured = []
    def capture(module, inputs, output):
        output.retain_grad()
        captured.append(output)
    handle = model.target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image.detach().clone().requires_grad_(True))
        predicted = logits.argmax(1).item()
        logits[0, predicted].backward()
        activation = captured[0]
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)
        return cam, predicted
    finally:
        handle.remove()


def save_overlay(model, tensor, original, path):
    from matplotlib import colormaps
    from PIL import Image
    cam, predicted = heatmap(model, tensor)
    image = original.convert('RGB').resize((cam.shape[1], cam.shape[0]))
    colors = Image.fromarray((colormaps['jet'](cam)[:, :, :3] * 255).astype(np.uint8))
    Image.blend(image, colors, 0.4).save(path)
    return predicted
