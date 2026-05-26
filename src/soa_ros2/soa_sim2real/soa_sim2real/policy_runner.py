"""TorchScript policy loader and inference wrapper.

The exported ``policy.pt`` bakes in the ``EmpiricalNormalization`` module
(verified by inspecting the JIT graph), so callers pass raw 63-dim
observations and receive raw 6-dim action outputs.
"""

import numpy as np
import torch

from soa_sim2real.joint_order import OBS_DIM, ACTION_DIM


class PolicyRunner:
    def __init__(self, model_path: str, device: str = 'cuda', warmup_steps: int = 3):
        self._device = torch.device(device)
        self._model = torch.jit.load(model_path, map_location=self._device).eval()
        # Warmup so the first real call doesn't pay JIT specialization cost.
        with torch.inference_mode():
            zeros = torch.zeros((1, OBS_DIM), dtype=torch.float32, device=self._device)
            for _ in range(warmup_steps):
                self._model(zeros)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run one forward pass.

        Args:
            obs: shape ``(OBS_DIM,)`` float32 array.
        Returns:
            shape ``(ACTION_DIM,)`` float32 array (raw, un-clamped).
        """
        if obs.shape != (OBS_DIM,):
            raise ValueError(f'expected obs shape ({OBS_DIM},), got {obs.shape}')

        # TODO run observations through the model to infer actions:
        #   using torch inference mode (same as above):
        #       use numpy to create a float32 continguous array of the observations
        #       create a tensor from the numpy contiguous array
        #       send the tensor to the self._device (your GPU)
        #       unsqueeze the tensor on the GPU 
        #           this changes the vector shape from (63,) to (1, 63)
        #       call self._model with the new unsqueezed tensor
        #       move the output of self._model back to the CPU
        #           out.squeeze(0).cpu().numpy().astype(np.float32)
        #       return the final CPU hosted values
