import torch




mask = torch.ones(10, 29, dtype=torch.bool)
mask[:, -2:] = False

coors = torch.randn(10, 29, 3, dtype=torch.float32)

rand_num = torch.rand(10,)

random_int = torch.floor(rand_num * mask.sum(dim=-1)).long()

print(random_int, random_int.long())

print(coors[0])
coors[torch.arange(10), random_int] += torch.randn(10, 3)
print(random_int[0])
print(coors[0])