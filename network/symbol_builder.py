import logging
import torch

# from .models import RESNET18, R2PLUS1D, R21D_BERT, RESMC3, R3D50  # This require Pytorch >= 1.2.0 support
from .models import RESNET18, R2PLUS1D, RESMC3, R3D50  # This require Pytorch >= 1.2.0 support

from .config import get_config

def get_symbol(name, print_net=False, **kwargs):
	
	if name.upper() == "R3D18":
		net = RESNET18(**kwargs)
	elif name.upper() == "R2PLUS1D":
		net = R2PLUS1D(**kwargs)
	# elif name.upper() == "R21D_BERT":
	#	net = R21D_BERT(**kwargs)
	elif name.upper() == "RESMC3":
		net = RESMC3(**kwargs)
	elif name.upper() == "R3D50":
		net = R3D50(**kwargs)
	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf

