import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

EPS = 1e-8

@torch.no_grad()
def quantize_symmetric_per_tensor(tensor:torch.Tensor, n_bits:int):
    t_max = torch.max(torch.abs(tensor))
    q_max = 2**(n_bits - 1) - 1
    scale = t_max.clamp(EPS).div(q_max)
    q_tensor = tensor.div(scale).round()
    return q_tensor, scale, torch.tensor(.0)

@torch.no_grad()
def quantize_asymmetric_per_tensor(tensor:torch.Tensor, n_bits:int):
    t_max = torch.max(tensor)
    t_min = torch.min(tensor)
    scale = (t_max - t_min) / (2**n_bits - 1)
    zero_point = torch.round((t_max + t_min)/ 2 / scale)
    q_tensor = torch.round(tensor / scale) - zero_point
    return q_tensor, scale, zero_point

    
class FakeQuantLinear(nn.Module):
    def __init__(self, weight:torch.Tensor, bias:torch.Tensor, n_bits:int, quantize=quantize_symmetric_per_tensor) -> None:
        super().__init__()

        self.quantize = partial(quantize, n_bits=n_bits)
        q_w, s_w, z_w = self.quantize(weight)
        self.register_buffer('weight', q_w.squeeze(0))
        self.register_buffer('s_w', s_w)
        self.register_buffer('z_w', z_w)
        self.register_buffer("bias", bias)
        
    def forward(self, x):
        q_x, s_x, z_x = self.quantize(x)
        scale = s_x * self.s_w
        bias = self.bias if self.bias is None else self.bias.div(scale).round()
        out = F.linear(q_x.sub(z_x), self.weight.sub(self.z_w), bias)
        return out.mul_(scale)
    

# ===================================================
# ===================== TEST ========================
# ===================================================

def matmul_with_quantize(t1, t2, quantize):
    q1, s1, z1 = quantize(t1, 8)
    q2, s2, z2 = quantize(t2, 8)
    return torch.matmul(q1+z1, (q2+z2).T) * s1 * s2

def test_fakequant(t1, t2):
    result = torch.matmul(t1, t2.T)
    print('[symmetric] MSE loss', F.mse_loss(result, matmul_with_quantize(t1, t2, quantize_symmetric_per_tensor)))
    print('[asymmetric] MSE loss', F.mse_loss(result, matmul_with_quantize(t1, t2, quantize_asymmetric_per_tensor)))

def test_fake_quant_linear(t):
    linear = nn.Linear(t.size(1), 512)
    result = linear(t)
    quant_linear = FakeQuantLinear(linear.weight.detach(), linear.bias.detach(), 8)
    print('[quantized linear] MSE loss', F.mse_loss(result, quant_linear(t)).item())

def main():
    torch.manual_seed(42)
    # evenly distributed tensor
    t1 = torch.rand(128, 768) * 2 - 1
    t2 = torch.rand(128, 768) * 2 - 1
    print('=== test fake quantization')
    test_fakequant(t1, t2)
    
    print('=== test quantization of biased tensor')
    test_fakequant(t1 + 10, t2 - 10)

    print('=== test fakequant linear')
    test_fake_quant_linear(t1)
    
if __name__ == '__main__':
    main()