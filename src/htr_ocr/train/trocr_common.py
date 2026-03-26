import torch
from transformers import VisionEncoderDecoderModel


def fix_trocr_sinusoidal_positional_weights(
    model: VisionEncoderDecoderModel,
    device: torch.device,
) -> None:
    """
    Transformers 5 initializes modules on `meta`, and TrOCR sinusoidal positional weights are
    plain tensors (not parameters/buffers). They are therefore not moved by `model.to(device)`.
    """
    target_dtype = next(model.parameters()).dtype

    for module in model.modules():
        if module.__class__.__name__ != "TrOCRSinusoidalPositionalEmbedding":
            continue

        weights = getattr(module, "weights", None)
        if not torch.is_tensor(weights):
            continue

        if weights.device.type == "meta":
            rebuilt = module.get_embedding(
                int(weights.size(0)),
                int(module.embedding_dim),
                module.padding_idx,
            )
            module.weights = rebuilt.to(device=device, dtype=target_dtype)
            continue

        if weights.device != device or weights.dtype != target_dtype:
            module.weights = weights.to(device=device, dtype=target_dtype)
