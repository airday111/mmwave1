import torch
import torch.nn.functional as F

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
normalized_tensor = F.normalize(tensor, dim=1)
print(normalized_tensor)