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
outmap_old, label_old = cdcnext_model(example_input)

print("outmap shape:",outmap_old.shape)
print("label shape:",label_old.shape)
print("outmap", outmap_old)
print("label", label_old)
outmap = torch.sigmoid(outmap_old)
label =  torch.sigmoid(label_old)
print("outmap", outmap)
print("label", label)
# print(torch.flatten(label))
# print(torch.flatten(label).shape)
# print(outmap_old == outmap)
# print(label == label_old)
