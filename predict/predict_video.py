import sys
sys.path.append("..")

import os
import time
import json
import csv
import logging
import argparse
from zipfile import ZipFile

import torch
import numpy as np
import torch.backends.cudnn as cudnn

import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Prediction) default ARID")
# debug
parser.add_argument('--debug-mode', type=bool, default=True, help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='ARID', help="path to dataset")
parser.add_argument('--clip-length', default=16, help="define the length of each input sample.")
parser.add_argument('--frame-interval', type=int, default=2, help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='R3D50_dark', help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="../exps/models/", help="set model directory.")
parser.add_argument('--log-file', type=str, default="./predict-arid.log", help="set logging file.")
parser.add_argument('--data-root', type=str, default="../dataset/ARID_test_light", help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7", help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='R3D50', help="choose the base network")
# evaluation
parser.add_argument('--load-epoch', type=int, default=5, help="resume trained model")
parser.add_argument('--batch-size', type=int, default=4, help="batch size")

#other parameters
parser.add_argument('--list-file', type=str, default='ARID1.1_t1_validation_gt_pub.csv', help='list of testing videos, see list_cvt folder of each dataset for details')
parser.add_argument('--workers', type=int, default=4, help='num_workers during evaluation data loading')
parser.add_argument('--zip-file', type=str, default='Track2_1.zip', help='zip file destination for prediction csv file')


def autofill(args):
	# customized
	if not args.task_name:
		args.task_name = os.path.basename(os.getcwd())
	# fixed
	args.model_prefix = os.path.join(args.model_dir, args.task_name)
	return args

def set_logger(log_file='', debug_mode=False):
	if log_file:
		if not os.path.exists("./"+os.path.dirname(log_file)):
			os.makedirs("./"+os.path.dirname(log_file))
		handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	else:
		handlers = [logging.StreamHandler()]

	""" add '%(filename)s' to format show source file """
	logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers = handlers)


if __name__ == '__main__':
	epoch = 1
	pred_i_epoch = [0] * 120
	for epoch in range(1, 121):
		# set args
		args = parser.parse_args()
		args = autofill(args)

		set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
		logging.info("Start evaluation with args:\n" + json.dumps(vars(args), indent=4, sort_keys=True))

		# set device states
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
		assert torch.cuda.is_available(), "CUDA is not available"

		# load dataset related configuration
		dataset_cfg = dataset.get_config(name=args.dataset)

		# creat model
		sym_net, input_config = get_symbol(name=args.network, **dataset_cfg)

		# network
		if torch.cuda.is_available():
			cudnn.benchmark = True
			sym_net = torch.nn.DataParallel(sym_net).cuda()
			criterion = torch.nn.CrossEntropyLoss().cuda()
		else:
			sym_net = torch.nn.DataParallel(sym_net)
			criterion = torch.nn.CrossEntropyLoss()

		net = static_model(net=sym_net, criterion=criterion, model_prefix=args.model_prefix)
		net.load_checkpoint(epoch=epoch)

		# data iterator:
		# data_root = "../dataset/{}".format(args.dataset)
		data_root = args.data_root
		video_location = os.path.join(data_root, 'raw', 'validate')
		# video_location = os.path.join(data_root, 'raw', 'test')

		normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])

		val_sampler = sampler.RandomSampling(num=args.clip_length, interval=args.frame_interval, speed=[1.0, 1.0], seed=1)
		val_loader = VideoIter(video_prefix=video_location, csv_list=os.path.join(data_root, 'raw', 'list_cvt', args.list_file), sampler=val_sampler,
						  force_color=True, video_transform=transforms.Compose([transforms.Resize((256,256)), transforms.RandomCrop((224,224)), transforms.ToTensor(), normalize]),
						  name='predict', return_item_subpath=True,)

		eval_iter = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

		# main loop
		net.net.eval()
		sum_batch_elapse = 0.
		softmax = torch.nn.Softmax(dim=1)
		field_names = ['VideoID', 'Video', 'ClassID']
		pred_rows = []
		pred_acc = 0
		pred_file = 'track1_pred.csv'

		i_batch = 0
		for datas, targets, video_subpaths in eval_iter:

			batch_start_time = time.time()

			outputs_all, _ = net.forward(datas, target=None)

			sum_batch_elapse += time.time() - batch_start_time

			# recording
			outputs = softmax(outputs_all[0]).data.cpu()
			for i_item in range(0, outputs.shape[0]):
				output_i = outputs[i_item,:].view(1, -1)
				target_i = targets[i_item]
				# assert target_i == -1, "Target not -1, label should not be contained in validation/testing data"
				video_subpath_i = video_subpaths[i_item]
				_, pred_class_i = torch.topk(output_i, 1)
				class_id_i = pred_class_i.numpy()[0][0]
				video_id_i = int(video_subpath_i.split('.')[0])
				pred_row = {field_names[0]: video_id_i, field_names[1]: video_subpath_i, field_names[2]: class_id_i}
				if class_id_i == target_i:
					pred_acc += 1
				pred_rows.append(pred_row)

			# show progress
			if (i_batch % 10) == 0:
				logging.info("{:.1f}% \t| Batch [{}]    \t".format(float(100*i_batch) / eval_iter.__len__(),  i_batch))
			i_batch += 1

		# finished
		pred_acc = pred_acc/3.2
		logging.info("Prediction Finished!")
		logging.info("Total time cost: {:.1f} sec".format(sum_batch_elapse))
		logging.info("Prediction accuracy: {:.2f} %".format(pred_acc))

		# write answer to prediction csv file and zipped
		with open(pred_file, 'w', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=field_names)
			writer.writeheader()
			for pred_row in pred_rows:
				writer.writerow(pred_row)

		csvfile.close()

		with ZipFile(args.zip_file, 'w') as zip:
			zip.write(pred_file)

		zip.close()
		logging.info("Prediction file written and zipped, ready for submission!")
		pred_i_epoch[epoch-1] = pred_acc
		epoch += 1

	print(np.array(pred_i_epoch))
	# print(size(pred_i_epoch))

