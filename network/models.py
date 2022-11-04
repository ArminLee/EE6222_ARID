# Require Pytorch Version >= 1.2.0
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision


try:
	from . import initializer
	from .utils import load_state
	from .utils import load_state
except: 
	import initializer
	from utils import load_state

from . import resnet
from . import r2plus1d
# from . import rgb_r2plus1d

class RESNET18(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(RESNET18, self).__init__()

		self.resnet = torchvision.models.video.r3d_18(pretrained=False, progress=False, num_classes=num_classes, **kwargs)

		###################
		# Initialization #
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/r3d_18-b3b3357e.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self.resnet, pretrained)
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet(x)

		return h

class R2PLUS1D(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(R2PLUS1D, self).__init__()

		self.resnet = r2plus1d.generate_model(model_depth=50, n_classes=num_classes, **kwargs)

		###################
		# Initialization #
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/r2p1d50_KM_200ep.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self.resnet, pretrained['state_dict'])
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet.forward(x)

		return h
"""
class R21D_BERT(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(R21D_BERT, self).__init__()

		self.resnet = rgb_r2plus1d.rgb_r2plus1d_64f_34_bert10(num_classes=num_classes, length=64)

		###################
		# Initialization #
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/model_best.pth.tar')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self.resnet, pretrained['state_dict'])
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet.forward(x)

		return h
"""

class R3D50(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(R3D50, self).__init__()

		self.resnet = resnet.generate_model(model_depth=50, n_classes=num_classes, **kwargs)

		###################
		# Initialization #
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/r3d50_KS_200ep.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self.resnet, pretrained['state_dict'])
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet.forward(x)

		return h

class RESMC3(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(RESMC3, self).__init__()

		self.resnet = torchvision.models.video.r2plus1d_18(pretrained=False, progress=False, num_classes=num_classes, **kwargs)

		###################
		# Initialization #
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/mc3_18-a90a0ba3.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			load_state(self.resnet, pretrained)
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet(x)

		return h


if __name__ == "__main__":
	logging.getLogger().setLevel(logging.DEBUG)
	# ---------
	net = RESNET18(num_classes=100, pretrained=True)
	data = torch.autograd.Variable(torch.randn(1,3,16,224,224))
	output = net(data)
	print (output.shape)
