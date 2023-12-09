import torch
import torch.nn as nn
import torch.nn.functional as F
from fake_quant import FakeQuantLinear

EPS = 1e-8

class SmoothQuantLinear(nn.Module):
    def __init__(self, weight:torch.Tensor, bias:torch.Tensor, act_scale:torch.Tensor, n_bits:int, alpha:float = 0.5):
        super(SmoothQuantLinear, self).__init__()
        
        w_abs_max = weight.abs().max(dim=0)[0].clamp(min=EPS)
        scale = act_scale.pow(alpha).div(w_abs_max.pow(1 - alpha)).clamp(min=EPS)
        self.register_buffer("scale", scale) # smoothing scale
        self.fake_quant_linear = FakeQuantLinear(weight * scale.view(1, -1), bias, n_bits)

    def smooth(self, x):
        return x / self.scale
    
    def forward(self, x):
        smooth_x = self.smooth(x)
        output = self.fake_quant_linear(smooth_x)
        return output

# ===================================================
# ===================== TEST ========================
# ===================================================

def test():
    torch.manual_seed(42)
    t = torch.rand((1, 128, 768))
    t[:,:,torch.randint(0, 767, (5,))] = torch.rand((1, 128, 5)) * 10 + 50

    linear = nn.Linear(768, 128)
    result = linear(t)
    
    fq_linear = FakeQuantLinear(linear.weight.detach(), linear.bias.detach(), 8)
    print('[fake quant] MSE loss', F.mse_loss(result, fq_linear(t)).item())

    act_scale = torch.max(t, dim=1).values
    smooth_linear = SmoothQuantLinear(linear.weight.detach(), linear.bias.detach(), act_scale, 8)
    print('[smooth quant] MSE loss', F.mse_loss(result, smooth_linear(t)).item())


def main():
    torch.manual_seed(42)
    test()
    
if __name__ == '__main__':
    main()