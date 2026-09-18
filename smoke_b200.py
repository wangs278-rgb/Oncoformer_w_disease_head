import torch
import torch.nn.functional as F

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
print("arch_list:", torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}:", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))

# Real kernel launches on the B200 — this is what failed with torch 2.3.0 ("no kernel image").
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
c = a @ b
torch.cuda.synchronize()
print("matmul ok, sum:", float(c.sum()))

x = torch.randn(8, 8, 128, 64, device="cuda")
y = F.scaled_dot_product_attention(x, x, x)
torch.cuda.synchronize()
print("sdpa ok:", tuple(y.shape))

print("SMOKE TEST PASSED")
