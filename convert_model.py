import torch
from smaller_resnet_with_box_cls import custom_resnet
#from mobilenetv2_stride16 import MobileNetV2_stride16
from smaller_resnet_with_box_cls import custom_resnet
from momo_pytorch2caffe import *

# model = custom_resnet()
# model_dict = torch.load('models/custom_resnet_v2_192x192_stride16_86.pth')

model = custom_resnet()
model_dict = torch.load('models/custom_resnet_v2_160x160_bigger_stride16_79.pth')

model.load_state_dict(model_dict)

# save caffe model
caffe_net = caffe.NetSpec()
layer = L.Input(shape=dict(dim=[1, 3, 160, 160]))
caffe_net.tops['data'] = layer
model.generate_caffe_prototxt(caffe_net, layer)
print(caffe_net.to_proto())
with open('model_ycj/custom_resnet_v2_160x160' + '.prototxt', 'w') as f:
    f.write(str(caffe_net.to_proto()))
caffe_net = caffe.Net('model_ycj/custom_resnet_v2_160x160' + '.prototxt', caffe.TEST)
convert_weight_from_pytorch_to_caffe(model, caffe_net)
caffe_net.save('model_ycj/custom_resnet_v2_160x160' + '.caffemodel')