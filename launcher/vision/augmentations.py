import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def batched_random_crop(img: torch.Tensor, rng: torch.Generator, padding: int, num_batch_dims: int = 1) -> torch.Tensor:
    """Support [B, H, W, C], [B, T, H, W, C] formats"""
    original_shape = img.shape
    batch_size = 1
    for i in range(num_batch_dims):
        batch_size *= original_shape[i]
    
    H, W, C = original_shape[-3], original_shape[-2], original_shape[-1]
    img = img.reshape(batch_size, H, W, C)
    
    # Pad: [B, H+2*padding, W+2*padding, C]
    img_bchw = img.permute(0, 3, 1, 2)  # [B, C, H, W]
    padded = F.pad(img_bchw, (padding, padding, padding, padding), mode="replicate")
    padded = padded.permute(0, 2, 3, 1)  # [B, H+2*padding, W+2*padding, C]
    
    # Generate offsets: [B, 2]
    offsets = torch.randint(0, 2 * padding + 1, (batch_size, 2), generator=rng, device=img.device)
    
    h_indices = torch.arange(H, device=img.device).unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
    w_indices = torch.arange(W, device=img.device).unsqueeze(0).unsqueeze(0)  # [1, 1, W]
    h_offsets = offsets[:, 0].view(batch_size, 1, 1)  # [B, 1, 1]
    w_offsets = offsets[:, 1].view(batch_size, 1, 1)  # [B, 1, 1]
    
    h_idx = (h_indices + h_offsets).expand(batch_size, H, W)  # [B, H, W]
    w_idx = (w_indices + w_offsets).expand(batch_size, H, W)  # [B, H, W]
    b_idx = torch.arange(batch_size, device=img.device).view(batch_size, 1, 1).expand(batch_size, H, W)  # [B, H, W]
    
    cropped = padded[b_idx, h_idx, w_idx, :]  # [B, H, W, C]
    
    return cropped.reshape(original_shape)

def batched_color_transform(
    image: torch.Tensor,
    rng: torch.Generator,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.1,
    num_batch_dims: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    original_shape = image.shape
    batch_size = 1
    for i in range(num_batch_dims):
        batch_size *= original_shape[i]
    
    H, W, C = original_shape[-3], original_shape[-2], original_shape[-1]
    image = image.reshape(batch_size, H, W, C)
    
    # Convert to [B, C, H, W] for torchvision
    image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
    
    # Apply transforms (same random params for all images in batch for temporal consistency)
    if brightness > 0:
        factor = 1 + (torch.rand(1, generator=rng, device=device).item() * 2 - 1) * brightness
        image = TF.adjust_brightness(image, factor)
    if contrast > 0:
        factor = 1 + (torch.rand(1, generator=rng, device=device).item() * 2 - 1) * contrast
        image = TF.adjust_contrast(image, factor)
    if saturation > 0:
        factor = 1 + (torch.rand(1, generator=rng, device=device).item() * 2 - 1) * saturation
        image = TF.adjust_saturation(image, factor)
    if hue > 0:
        factor = (torch.rand(1, generator=rng, device=device).item() * 2 - 1) * hue
        image = TF.adjust_hue(image, factor)
    
    # Convert back to [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, H, W, C]
    image = torch.clamp(image, 0.0, 1.0)
    
    return image.reshape(original_shape)

                             
def resize(image: torch.Tensor, image_dim: tuple) -> torch.Tensor:
    assert len(image_dim) == 2
    return F.interpolate(
        image.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
        size=image_dim,
        mode='bilinear',
        align_corners=False
    ).squeeze(0).permute(1, 2, 0)  # Back to [H, W, C]

def _maybe_apply(apply_fn, inputs: torch.Tensor, rng: torch.Generator, apply_prob: float) -> torch.Tensor:
    should_apply = torch.rand(1, generator=rng, device=inputs.device).item() <= apply_prob
    return apply_fn(inputs) if should_apply else inputs

def rgb_to_hsv(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> tuple:
    return TF.rgb_to_hsv(torch.stack([r, g, b], dim=-1))

def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> tuple:
    hsv = torch.stack([h, s, v], dim=-1)
    rgb = TF.hsv_to_rgb(hsv)
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]

def adjust_brightness(rgb_tuple: tuple, delta: float) -> tuple:
    return tuple(x + delta for x in rgb_tuple)

def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return factor * (image - mean) + mean

def adjust_saturation(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, factor: float) -> tuple:
    return h, torch.clamp(s * factor, 0.0, 1.0), v

def adjust_hue(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor, delta: float) -> tuple:
    return (h + delta) % 1.0, s, v

def color_transform(
    image: torch.Tensor,
    rng: torch.Generator,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    to_grayscale_prob: float = 0.0,
    color_jitter_prob: float = 1.0,
    apply_prob: float = 1.0,
    shuffle: bool = True
) -> torch.Tensor:
    def _to_grayscale(image):
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
        grayscale = (image * rgb_weights).sum(dim=-1, keepdim=True)
        return grayscale.repeat(1, 1, 3)

    should_apply = torch.rand(1, generator=rng, device=image.device).item() <= apply_prob
    should_apply_gs = torch.rand(1, generator=rng, device=image.device).item() <= to_grayscale_prob
    should_apply_color = torch.rand(1, generator=rng, device=image.device).item() <= color_jitter_prob

    if should_apply and should_apply_color:
        transforms = []
        if brightness > 0:
            transforms.append(lambda img: TF.adjust_brightness(img, 1 + torch.rand(1, generator=rng).item() * brightness))
        if contrast > 0:
            transforms.append(lambda img: TF.adjust_contrast(img, 1 + torch.rand(1, generator=rng).item() * contrast))
        if saturation > 0:
            transforms.append(lambda img: TF.adjust_saturation(img, 1 + torch.rand(1, generator=rng).item() * saturation))
        if hue > 0:
            transforms.append(lambda img: TF.adjust_hue(img, torch.rand(1, generator=rng).item() * hue))
            
        if shuffle:
            indices = torch.randperm(len(transforms), generator=rng)
            transforms = [transforms[i] for i in indices]
            
        image_NCHW = image.permute(2, 0, 1).unsqueeze(0)
        for t in transforms:
            image_NCHW = t(image_NCHW)
        image = image_NCHW.squeeze(0).permute(1, 2, 0)
        
    if should_apply and should_apply_gs:
        image = _to_grayscale(image)
        
    return torch.clamp(image, 0.0, 1.0)

def gaussian_blur(
    image: torch.Tensor,
    rng: torch.Generator,
    blur_divider: float = 10.0,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0,
    apply_prob: float = 1.0
) -> torch.Tensor:
    kernel_size = int(image.shape[0] / blur_divider) | 1  # Ensure odd kernel size
    
    def _apply(image):
        sigma = sigma_min + torch.rand(1, generator=rng, device=image.device).item() * (sigma_max - sigma_min)
        return TF.gaussian_blur(
            image.permute(2, 0, 1).unsqueeze(0),
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma]
        ).squeeze(0).permute(1, 2, 0)
        
    return _maybe_apply(_apply, image, rng, apply_prob)

def solarize(image: torch.Tensor, rng: torch.Generator, threshold: float, apply_prob: float) -> torch.Tensor:
    def _apply(image):
        return torch.where(image < threshold, image, 1.0 - image)
    return _maybe_apply(_apply, image, rng, apply_prob) 

def _unpack(batch: dict, image_keys: tuple) -> dict:
    for pixel_key in image_keys:
        # Check if pixel_key is in observations but not in next_observations
        if pixel_key in batch["observations"] and pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key]
            if isinstance(obs_pixels, torch.Tensor):
                # Packed format: (B, T+1, H, W, C) -> split into obs (B, T, H, W, C) and next_obs
                obs = dict(batch["observations"])
                next_obs = dict(batch["next_observations"])
                
                obs[pixel_key] = obs_pixels[:, :-1, ...]
                next_obs[pixel_key] = obs_pixels[:, 1:, ...]
                
                batch = dict(batch)
                batch["observations"] = obs
                batch["next_observations"] = next_obs
    return batch

def data_augmentation_fn(image_keys: tuple, observations: dict, seed: int, device: str = "cpu") -> dict:
    # Create a generator from the seed
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    
    for pixel_key in image_keys:
        if pixel_key in observations:
            observations = {
                **observations,
                pixel_key: batched_random_crop(
                    observations[pixel_key], 
                    rng=rng, 
                    padding=4, 
                    num_batch_dims=2
                )
            }
    return observations

def make_batch_augmentation_func(image_keys: tuple, enable_next_obs=True) -> callable:
    def augment_batch_RL(batch: dict, seed: int, device: str = "cpu") -> dict:
        # First unpack packed obs and next_obs if needed
        batch = _unpack(batch, image_keys)
        
        obs_seed = seed
        next_obs_seed = seed + 1
        
        obs = data_augmentation_fn(image_keys, batch["observations"], obs_seed, device)
        next_obs = data_augmentation_fn(image_keys, batch["next_observations"], next_obs_seed, device)
        
        return {
            **batch,
            "observations": obs,
            "next_observations": next_obs,
        }

    def augment_batch_IL(batch: dict, seed: int, device: str = "cpu") -> dict:
        obs = data_augmentation_fn(image_keys, batch["observations"], seed, device)
        return {
            **batch,
            "observations": obs,
        }
    
    return augment_batch_RL if enable_next_obs else augment_batch_IL