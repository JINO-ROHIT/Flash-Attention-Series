import torch
import math

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

batch_size = 8
n_head = 12
seq_len = 128
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print(q.dtype)

att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
att = torch.nn.functional.softmax(att, dim=-1)
y = att @ v

print(y.shape)
