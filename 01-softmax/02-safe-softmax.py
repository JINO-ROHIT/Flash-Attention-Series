## the problem with the previous kernel is that if we try to take the exp of large numbers, we run into overflows really quick, wanna see?

import torch

print(torch.exp(torch.tensor(100))) # this gives you an infinity lol

## idea - since in softmax, we always have more than a single input, so what if we take the largest value and subtract it from each number of the row
## and then do softmax, will this work? lets see

x = torch.tensor([0.2, 0.5, 0.1, -0.5, -0.4, 100])

# 1. take a largest values
largest = torch.max(x)

# 2. subtract the largest from each value
normalized_x = x - largest

# do the same as before

# 3. we take the sum of exp of x
denom = torch.sum(torch.exp(normalized_x))

# 2. we take the exp of x
num = torch.exp(normalized_x)

# 3. we divide num/denom
result = num/denom
print(result)

## wanna make sure this is right? sum across the row, it should add upto 1
print(f"this should be equal to 1 : {torch.sum(result)}")