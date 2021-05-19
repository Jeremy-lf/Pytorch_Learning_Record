# -*- coding: utf-8 -*-
"""
Created on : 2021/3/22 16:53

@author: Jeremy
"""
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
# model = torchvision.models
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
model = torchvision.models.mobilenet_v3_small()
# model = torch.load("./mobilenet_v3_small-047dcff4.pth")
model.load_state_dict(torch.load("./mobilenet_v3_small-047dcff4.pth"))
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model.save("./HelloWorldApp/app/src/main/assets/model.pt")