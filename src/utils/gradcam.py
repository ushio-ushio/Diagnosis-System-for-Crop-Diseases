import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, task='type', target_class=None):
        output = self.model(input_image)

        if isinstance(output, tuple):
            type_pred, severity_pred = output
            if task == 'type':
                pred = type_pred
            else:
                pred = severity_pred
        else:
            pred = output

        if target_class is None:
            target_class = torch.argmax(pred, dim=1)
        else:
            if not isinstance(target_class, torch.Tensor):
                target_class = torch.tensor(target_class)
            if target_class.dim() == 0:
                target_class = target_class.unsqueeze(0)

        score = pred[0, target_class]

        self.model.zero_grad()
        score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        return heatmap.cpu().numpy(), target_class.item()

    def __call__(self, scores, class_idx, input_size):
        self.model.zero_grad()
        loss = scores[:, class_idx].sum()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam_up = F.interpolate(
            cam.unsqueeze(1),
            size=input_size,
            mode="bilinear",
            align_corners=False
        )
        cam_up = cam_up[0, 0].detach().cpu().numpy()
        return cam_up
