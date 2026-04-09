import torch

x = torch.tensor([0.2, 0.5, 0.1, -0.5, -0.4])

# 1. we take the sum of exp of x
denom = torch.sum(torch.exp(x))

# 2. we take the exp of x
num = torch.exp(x)

# 3. we divide num/denom
result = num/denom
print(result)

## wanna make sure this is right? sum across the row, it should add upto 1
print(f"this should be equal to 1 : {torch.sum(result)}")
