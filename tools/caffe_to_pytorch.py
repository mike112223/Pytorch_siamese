import argparse

def main(args):
	import caffe

	proto = args.proto
	weight = args.weight
	net = caffe.Net(proto, weight, caffe.TEST)

	params = net.params
	a_tran = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
	b_tran = {'a': 1, 'b': 2, 'c': 3}
	py_params = {}
	py_params['module.base.conv1.weight'] = params['conv1'][0].data
	py_params['module.base.conv1.bias'] = params['conv1'][1].data
	py_params['module.base.bn1.weight'] = params['scale_conv1'][0].data
	py_params['module.base.bn1.bias'] = params['scale_conv1'][1].data
	py_params['module.base.bn1.running_mean'] = params['bn_conv1'][0].data/params['bn_conv1'][2].data[0]
	py_params['module.base.bn1.running_var'] = params['bn_conv1'][1].data/params['bn_conv1'][2].data[0]

	for i in range(3, len(params.keys())//2):
		key = params.keys()[i]
		a_b = key.split('_')
		if len(a_b) == 2:
			a = a_b[0]
			b = a_b[1]
			layer_type = a[:-2]
			branch_type = b
			layer = str(int(a[-2])-1)+'.'+str(a_tran[a[-1]])

			if b == 'branch1':
				if layer_type == 'res':
					py_params['module.base.layer{}.downsample.0.weight'.format(layer)] = params[key][0].data
				elif layer_type == 'scale':
					py_params['module.base.layer{}.downsample.1.weight'.format(layer)] = params[key][0].data
					py_params['module.base.layer{}.downsample.1.bias'.format(layer)] = params[key][1].data
				else:
					py_params['module.base.layer{}.downsample.1.running_mean'.format(layer)] = params[key][0].data/params[key][2].data[0]
					py_params['module.base.layer{}.downsample.1.running_var'.format(layer)] = params[key][1].data/params[key][2].data[0]
			else:
				l = b_tran[b[-1]]
				if layer_type == 'res':
					py_params['module.base.layer{}.conv{}.weight'.format(layer,l)] = params[key][0].data
				elif layer_type == 'scale':
					py_params['module.base.layer{}.bn{}.weight'.format(layer,l)] = params[key][0].data
					py_params['module.base.layer{}.bn{}.bias'.format(layer,l)] = params[key][1].data
				else:
					py_params['module.base.layer{}.bn{}.running_mean'.format(layer,l)] = params[key][0].data/params[key][2].data[0]
					py_params['module.base.layer{}.bn{}.running_var'.format(layer,l)] = params[key][1].data/params[key][2].data[0]

		else:
			py_params['module.base.{}.weight'.format(a_b[0])] = params[key][0].data
			py_params['module.base.{}.bias'.format(a_b[0])] = params[key][1].data

	from siamese_resnet import res50_sia
	import torch
	import torch.nn as nn

	for i in py_params:
		py_params[i] = torch.from_numpy(py_params[i]).float()

	model = res50_sia(pretrained=False, siamese=True)
	del model.base.fc
	model = nn.DataParallel(model,device_ids=[0]).cuda()
	model.load_state_dict(py_params)
	torch.save(model.state_dict(), '{}'.format(args.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Siamese")
    parser.add_argument('--proto', type=str, default=None)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)

    main(parser.parse_args())


