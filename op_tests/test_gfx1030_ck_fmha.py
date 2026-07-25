# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical smoke test for the CK FMHA software-MMA path on gfx1030."""

import math

import torch
import torch.nn.functional as F

from aiter.ops.mha import mha_fwd


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A ROCm GPU is required")

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx1030":
        raise RuntimeError(f"This smoke test requires gfx1030, got {arch}")

    torch.manual_seed(1234)
    shape = (1, 128, 8, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(shape[-1])

    output, *_ = mha_fwd(
        q,
        k,
        v,
        0.0,
        scale,
        False,
        -1,
        -1,
        0,
        True,
        False,
    )
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).transpose(1, 2)

    max_abs = (output - reference).abs().max().item()
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    print(f"gfx1030 CK FMHA numerical smoke test passed; max_abs={max_abs:.6f}")


if __name__ == "__main__":
    main()
