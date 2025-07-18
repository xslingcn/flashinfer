import pytest
import torch
import torch.nn.functional as F

from flashinfer import gemm_bf16


@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize("n", [64, 128, 256])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("with_bias", [True, False])
def test_gemm_bf16(m, n, k, with_bias):
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if with_bias else None
    
    result = gemm_bf16(A, B, bias)
    
    reference = F.linear(A, B, bias)
    
    assert torch.allclose(result, reference)


if __name__ == "__main__":
    pytest.main([__file__])
