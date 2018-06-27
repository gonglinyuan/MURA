import torch
from torch import nn
from torchvision import datasets, models, transforms
import pretrainedmodels
import numpy as np
import PIL
import os
from matplotlib import cm

# Define the class for loading Inception v4 network.
# The class is spetial tuned for only outputing the feature map predicted by the network.
class ConvnetModel(nn.Module):
    def __init__(self):
        super(ConvnetModel, self).__init__()
        self.convnet = pretrainedmodels.models.inceptionv4()
        kernel_count = self.convnet.last_linear.in_features
        self.convnet.last_linear = nn.Linear(kernel_count, 1)

    def forward(self, x):
        x = self.convnet.features(x)
        return x
    
    def predict(self, feature):
        feature = self.convnet.avg_pool(feature)
        feature = feature.reshape((1, 1536))
        feature = self.convnet.last_linear(feature)
        return feature

    
# Given the input tensor, return the image with diagnostic location.
# Input_tensor should have shape (1, 3, 299, 299)

def advanced_locate(input_tensor):
    
    # Generate a neural network and load parameters.
    net = ConvnetModel()
    model_checkpoint = torch.load("m-20180518-024519-A.pth.tar")
    net.load_state_dict(model_checkpoint['state_dict'])
    
    # Calculate gradients of the last feature maps.
    feature = net(input_tensor)
    feature_np = feature.detach().numpy()
    feature = torch.tensor(feature_np, requires_grad=True)
    prediction = net.predict(feature)
    prediction.backward()
    grad = feature.grad.detach().numpy()

    # Calculate weights by grads.
    mean_grad = np.mean(grad, axis=3)
    weight = np.mean(mean_grad, axis=2)

    # Calculate weighted sum of all feature map.
    feature_map = feature.data.detach().numpy()
    weighted_feature_map = weight.reshape((1, 1536, 1, 1)) * feature_map
    sumed_feature_map = np.sum(weighted_feature_map, axis=0)
    sumed_feature_map = np.sum(sumed_feature_map, axis=0)

    # Normalize weight-sumed feature map and convert it to 8-bit gray scale.
    negative_entry = sumed_feature_map < 0
    sumed_feature_map *= (1 - negative_entry)
    if np.max(sumed_feature_map) >= 0.16: # This is specially tuned for inception v4
        sumed_feature_map /= np.max(sumed_feature_map)
        sumed_feature_map *= 255
    else:
        sumed_feature_map *= 1593
    sumed_feature_map = np.around(sumed_feature_map)
    sumed_feature_map = sumed_feature_map.astype(np.uint8)

    # Scale feature map to (299, 299) with bluring.
    scaled_feature_map = PIL.Image.fromarray(sumed_feature_map)
    gaussian = PIL.ImageFilter.GaussianBlur()
    scaled_feature_map = scaled_feature_map.resize((32, 32))
    scaled_feature_map = scaled_feature_map.filter(gaussian)
    scaled_feature_map = scaled_feature_map.resize((64, 64))
    scaled_feature_map = scaled_feature_map.filter(gaussian)
    scaled_feature_map = scaled_feature_map.resize((128, 128))
    scaled_feature_map = scaled_feature_map.filter(gaussian)
    scaled_feature_map = scaled_feature_map.resize((299, 299))
    scaled_feature_map = scaled_feature_map.filter(gaussian)

    # Build heatmap from feature map
    gray_feature_map = np.asarray(scaled_feature_map)
    colormap = cm.get_cmap('inferno') # or what ever color map you want
    heatmap = colormap(gray_feature_map)
    heatmap = np.delete(heatmap, 3, axis=2)

    # Transfer input tensor back to image.
    DATA_MEAN = 0.20558404267255
    DATA_STD = 0.17694948680626902473216631207703
    input_array = input_tensor.detach().numpy()[0]
    input_array = input_array * DATA_STD + DATA_MEAN
    input_array = input_array.transpose(1, 2, 0)

    # Apply masked merge between heatmap and input.
    mask = np.asarray(scaled_feature_map)
    mask = mask / 255
    mask = mask.reshape(299, 299, 1)
    result_array = input_array * (1 - mask) + heatmap * mask
    result_array *= 255
    result_array = np.around(result_array)
    result_array = np.clip(result_array, 0, 255)
    result_array = result_array.astype(np.uint8)
    result_image = PIL.Image.fromarray(result_array)
    
    return result_image

### Usage ###
# input_sample = get_input()
# output_sample = locate(input_sample)
