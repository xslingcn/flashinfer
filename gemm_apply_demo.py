import torch
import torch.nn.functional as F

from flashinfer import gemm_bf16

def test_gemm_bf16(m, n, k, with_bias):
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if with_bias else None
    
    result = gemm_bf16(A, B, bias)
    
    reference = F.linear(A, B, bias)
    
    assert torch.allclose(result, reference)


if __name__ == "__main__":
    test_gemm_bf16(32, 4096, 4096, False)
