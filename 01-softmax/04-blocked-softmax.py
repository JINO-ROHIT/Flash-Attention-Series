## the idea of this is to break the tensor into blocks to parallelize the computation, this is especially beenficial when implementing in CUDA.

import torch

x = torch.tensor([0.2, 0.5, 0.1, -0.5, -0.4, 100])

block1, block2 = x.split(3)

print(f"the blocks are: \n {block1} \n {block2}")

def calculate_denom(x: torch.Tensor, curr_max, running_denom):
    for i in range(1, x.shape[0]):
        if x[i] > curr_max:
            old_max = curr_max
            curr_max = x[i]
            running_denom = running_denom * torch.exp(old_max - curr_max) + torch.exp(x[i] - curr_max)
        else:
            running_denom += torch.exp(x[i] - curr_max)
    return curr_max, running_denom


# for block1
curr_max_b1 = block1[0]
running_denom_b1 = torch.exp(block1[0] - curr_max_b1)

# for block2
curr_max_b2 = block2[0]
running_denom_b2 = torch.exp(block2[0] - curr_max_b2)

b1_max, b1_denom = calculate_denom(block1, curr_max_b1, running_denom_b1)
b2_max, b2_denom = calculate_denom(block2, curr_max_b2, running_denom_b2)

print(f"max across block1 is: {b1_max}")
print(f"max across block2 is: {b2_max}")

final_max = max(b1_max, b2_max)
final_denom = b1_denom * torch.exp(b1_max - final_max) + b2_denom * torch.exp(b2_max - final_max)

num = torch.exp(x - final_max)

result = num/final_denom
print(result)

## wanna make sure this is right? sum across the row, it should add upto 1
print(f"this should be equal to 1 : {torch.sum(result)}")


