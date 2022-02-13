from models import models
import torch
import torch.nn as nn

example_input = torch.rand((1,3,224,224))
cdcnext_model = models.get_cdcnext()
print(cdcnext_model)
# model = nn.Sequential(
#     cdcnext_model.downsample_layers,
#     cdcnext_model.stages
# )
outmap, embedding_result = cdcnext_model(example_input)
print("outmap shape:",outmap.shape)
print("embedding shape:",embedding_result.shape)

