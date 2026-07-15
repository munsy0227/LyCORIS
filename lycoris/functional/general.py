import torch
import torch.nn.functional as F


FUNC_LIST = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]


def rebuild_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    return rebuild2


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    second value is a value for weight.

    Because of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    """

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def power2factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    m = 2k
    n = 2**p
    m*n = dim
    """
    if factor == -1:
        factor = dimension

    # Find the first solution and check if it is even doable
    m = n = 0
    while m <= factor:
        m += 2
        while dimension % m != 0 and m < dimension:
            m += 2
        if m > factor:
            break
        if sum(int(i) for i in f"{dimension // m:b}") == 1:
            n = dimension // m

    if n == 0:
        return None, n
    return dimension // n, n


def tucker_weight_from_conv(up, down, mid):
    up = up.reshape(up.size(0), up.size(1))
    down = down.reshape(down.size(0), down.size(1))
    return torch.einsum("m n ..., i m, n j -> i j ...", mid, up, down)


def tucker_weight(wa, wb, t):
    temp = torch.einsum("i j ..., j r -> i r ...", t, wb)
    return torch.einsum("i j ..., i r -> r j ...", temp, wa)


def apply_dora_scale(
    org_weight,
    rebuild,
    dora_scale,
    scale,
    dora_zero_mask=None,
):
    compute_dtype = torch.promote_types(org_weight.dtype, rebuild.dtype)
    compute_dtype = torch.promote_types(compute_dtype, dora_scale.dtype)
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32
    base_weight = org_weight.to(dtype=compute_dtype)
    direction = base_weight + rebuild.to(
        device=base_weight.device,
        dtype=compute_dtype,
    )

    output_shape = (direction.shape[0], *[1] * (direction.dim() - 1))
    input_shape = (1, direction.shape[1], *[1] * (direction.dim() - 2))
    magnitude_shape = tuple(dora_scale.shape)
    if magnitude_shape == output_shape:
        norm_dims = tuple(range(1, direction.dim()))
        norm_direction = direction
        base_norm_direction = base_weight
    elif magnitude_shape == input_shape:
        norm_dims = (0, *range(2, direction.dim()))
        norm_direction = direction
        base_norm_direction = base_weight
    elif len(magnitude_shape) == direction.dim() + 1:
        groups = magnitude_shape[0]
        if (
            magnitude_shape[1] != 1
            or magnitude_shape[2] != direction.shape[1]
            or direction.shape[0] % groups != 0
            or any(size != 1 for size in magnitude_shape[3:])
        ):
            raise ValueError(
                "Invalid grouped-input DoRA magnitude shape: "
                f"weight={tuple(direction.shape)}, dora_scale={magnitude_shape}."
            )
        norm_direction = direction.reshape(
            groups,
            direction.shape[0] // groups,
            direction.shape[1],
            *direction.shape[2:],
        )
        base_norm_direction = base_weight.reshape_as(norm_direction)
        norm_dims = (1, *range(3, norm_direction.dim()))
    else:
        raise ValueError(
            "Cannot infer the DoRA norm axis from dora_scale: "
            f"weight={tuple(direction.shape)}, dora_scale={magnitude_shape}."
        )

    direction_norm = torch.linalg.vector_norm(
        norm_direction,
        dim=norm_dims,
        keepdim=True,
    )
    direction_norm = direction_norm.clamp_min(
        torch.finfo(direction.dtype).tiny
    ).detach()
    normalized_direction = norm_direction * (
        dora_scale.to(device=direction.device, dtype=direction.dtype) / direction_norm
    )
    if dora_zero_mask is None:
        base_norm = torch.linalg.vector_norm(
            base_norm_direction,
            dim=norm_dims,
            keepdim=True,
        )
        dora_zero_mask = base_norm == 0
    else:
        if not isinstance(dora_zero_mask, torch.Tensor):
            raise TypeError("dora_zero_mask must be a Tensor.")
        if tuple(dora_zero_mask.shape) != magnitude_shape:
            raise ValueError(
                "dora_zero_mask must match dora_scale: "
                f"mask={tuple(dora_zero_mask.shape)}, dora_scale={magnitude_shape}."
            )
    zero_mask = dora_zero_mask.to(device=direction.device, dtype=torch.bool)
    dora_weight = torch.where(
        zero_mask,
        norm_direction,
        normalized_direction,
    )
    dora_weight = dora_weight.reshape_as(direction)
    return base_weight + (dora_weight - base_weight) * scale
