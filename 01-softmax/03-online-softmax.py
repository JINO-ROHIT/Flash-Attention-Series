### now that we get a hang for softmax, we need to think deeply into this -
# 1. we will be doing softmax in the gpu, so what happens if our tensor is like super long, so long that it doesnt fit into memory? 
# 2. currently we need 3 passes, thats a lot of read and write to and from the HBM, can we make this faster?

import torch

x = torch.tensor([0.2, 0.5, 0.1, -0.5, -0.4, 100])

curr_max = x[0]
running_denom = torch.exp(x[0] - curr_max)

for i in range(1, x.shape[0]):
    if x[i] > curr_max:
        old_max = curr_max
        curr_max = x[i]
        running_denom = running_denom * torch.exp(old_max - curr_max) + torch.exp(x[i] - curr_max)
    else:
        running_denom += torch.exp(x[i] - curr_max)

num = torch.exp(x - curr_max)

result = num/running_denom
print(result)

## wanna make sure this is right? sum across the row, it should add upto 1
print(f"this should be equal to 1 : {torch.sum(result)}")
