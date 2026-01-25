import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handle = None

    def _forward_hook(self, module, input, output):
        self.activations = output
        # Register hook on the output tensor to capture backpropagated gradients, when grads are enabled
        if output.requires_grad:
            output.register_hook(self.save_gradient)
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap for input tensor x.
        """
        self.model.eval()

        # Register forward hook just for this call
        self._hook_handle = self.target_layer.register_forward_hook(self._forward_hook)
        try:
            # Ensure gradients are enabled even if outer code used torch.no_grad()
            with torch.enable_grad():
                # Forward pass
                logits = self.model(x)

                if class_idx is None:
                    # Default to the predicted class
                    class_idx = logits.argmax(dim=1).item()

                # Backward pass
                self.model.zero_grad()

                # We want to maximize the score of the target class
                score = logits[:, class_idx]
                score.backward(retain_graph=True) 

                # Get captured gradients and activations
                gradients = self.gradients
                activations = self.activations

                # Safety checks
                if gradients is None or activations is None:
                    # Fallback: return zero heatmap to avoid crashing
                    h = x.shape[2]
                    w = x.shape[3]
                    return np.zeros((h, w), dtype=np.float32)

                # Global average pooling of gradients: weights for each channel
                # gradients shape: [B, C, H, W] -> weights shape: [B, C, 1, 1]
                weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

                # Weighted combination of activation maps
                # activations shape: [B, C, H, W]
                cam = torch.sum(weights * activations, dim=1, keepdim=True)

                # Apply ReLU (we are only interested in features that have a positive influence)
                cam = F.relu(cam)

                # Post-processing: detach, normalize, resize
                cam = cam.squeeze().cpu().detach().numpy()

                # Handling batch dimension if present
                if cam.ndim == 0: # Scalar check just in case
                    return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

                # Resize to input image resolution
                # x shape is [1, 3, H, W], so we resize to (W, H)
                cam = cv2.resize(cam, (x.shape[3], x.shape[2])) 

                # Min-Max Normalization to [0, 1]
                cam = cam - np.min(cam)
                cam = cam / (np.max(cam) + 1e-7)

                return cam
        finally:
            # Remove hook to avoid interfering with future forward passes
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None
