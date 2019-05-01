# File to return the Deep VO model.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V


# DeepVO model
class DeepVO(nn.Module):

	def __init__(self, imageWidth, imageHeight, activation = 'relu', parameterization = 'default', batchnorm = False, \
		dropout = 0.0, flownet_weights_path = None, numLSTMCells = 1, hidden_units_LSTM = None, \
		numFC = 2, FC_dims = None):

		super(DeepVO, self).__init__()

		# Check if input image width and height are feasible
		self.imageWidth = int(imageWidth)
		self.imageHeight = int(imageHeight)
		if self.imageWidth < 64 or self.imageHeight < 64:
			raise ValueError('The width and height for an input image must be at least 64 px.')

		# Compute the size of the LSTM input feature vector.
		# There are 6 conv stages (some stages have >1 conv layers), which effectively reduce an 
		# image to 1/64 th of its initial dimensions. Further, the final conv layer has 1024
		# filters, hence, numConcatFeatures = 1024 * (wd/64) * (ht/64) = (wd * ht) / 4
		self.numConcatFeatures = int((self.imageWidth*self.imageHeight)/4)

		# Activation functions to be used in the network
		self.activation = activation

		# Parameterization representing the transform output by the network
		self.parameterization = parameterization
		if parameterization == 'quaternion':
			self.rotationDims = 4
		else:
			self.rotationDims = 3
		self.translationDims = 3

		# Whether or not batchnorm is required
		self.batchnorm = batchnorm

		# Whether or not dropout is required
		if dropout <= 0.0:
			self.dropout = False
		else:
			# Specify the drop_ratio
			self.dropout = True
			self.drop_ratio = dropout

		self.numLSTMCells = numLSTMCells
		self.hidden_units_LSTM = hidden_units_LSTM

		# Path to FlowNet weights
		if flownet_weights_path is not None:
			self.use_flownet = True
			self.flownet_weights_path = flownet_weights_path
		else:
			self.use_flownet = False

		"""
		Initialize variables required for the network
		"""

		# If we're using batchnorm, do not use bias for the conv layers
		self.bias = not self.batchnorm

		self.conv1   = nn.Conv2d(6, 64, 7, 2, 3, bias = self.bias)
		self.conv2   = nn.Conv2d(64, 128, 5, 2, 2, bias = self.bias)
		self.conv3   = nn.Conv2d(128, 256, 5, 2, 2, bias = self.bias)
		self.conv3_1 = nn.Conv2d(256, 256, 3, 1, 1, bias = self.bias)
		self.conv4   = nn.Conv2d(256, 512, 3, 2, 1, bias = self.bias)
		self.conv4_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = self.bias)
		self.conv5   = nn.Conv2d(512, 512, 3, 2, 1, bias = self.bias)
		self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = self.bias)
		self.conv6   = nn.Conv2d(512, 1024, 3, 2, 1, bias = self.bias)

		if self.batchnorm:
			self.conv1_bn = nn.BatchNorm2d(64)
			self.conv2_bn = nn.BatchNorm2d(128)
			self.conv3_bn = nn.BatchNorm2d(256)
			self.conv3_1_bn = nn.BatchNorm2d(256)
			self.conv4_bn = nn.BatchNorm2d(512)
			self.conv4_1_bn = nn.BatchNorm2d(512)
			self.conv5_bn = nn.BatchNorm2d(512)
			self.conv5_1_bn = nn.BatchNorm2d(512)
			self.conv6_bn = nn.BatchNorm2d(1024)

		self.conv7   = nn.Conv2d(4, 64, 7, 2, 3, bias = self.bias)
		self.conv8   = nn.Conv2d(64, 128, 5, 2, 2, bias = self.bias)
		self.conv9   = nn.Conv2d(128, 256, 5, 2, 2, bias = self.bias)
		self.conv9_1 = nn.Conv2d(256, 256, 3, 1, 1, bias = self.bias)
		self.conv10   = nn.Conv2d(256, 512, 3, 2, 1, bias = self.bias)
		self.conv10_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = self.bias)
		self.conv11   = nn.Conv2d(512, 512, 3, 2, 1, bias = self.bias)
		self.conv11_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = self.bias)
		self.conv12   = nn.Conv2d(512, 1024, 3, 2, 1, bias = self.bias)

		if self.batchnorm:
			self.conv7_bn = nn.BatchNorm2d(64)
			self.conv8_bn = nn.BatchNorm2d(128)
			self.conv9_bn = nn.BatchNorm2d(256)
			self.conv9_1_bn = nn.BatchNorm2d(256)
			self.conv10_bn = nn.BatchNorm2d(512)
			self.conv10_1_bn = nn.BatchNorm2d(512)
			self.conv11_bn = nn.BatchNorm2d(512)
			self.conv11_1_bn = nn.BatchNorm2d(512)
			self.conv12_bn = nn.BatchNorm2d(1024)

		self.conv13 = nn.Conv2d(2048, 1024, 3, 1, 1, bias = self.bias)
		self.conv14 = nn.Conv2d(1024, 1024, 3, 1, 1, bias = self.bias)

		if self.numLSTMCells == 1:
			self.lstm1 = nn.LSTMCell(self.numConcatFeatures, self.hidden_units_LSTM[0])
			self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
			self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
		else:
			self.lstm1 = nn.LSTMCell(self.numConcatFeatures, self.hidden_units_LSTM[0])
			self.lstm2 = nn.LSTMCell(self.hidden_units_LSTM[0], self.hidden_units_LSTM[1])
			self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
			self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
			self.h2 = torch.zeros(1, self.hidden_units_LSTM[1])
			self.c2 = torch.zeros(1, self.hidden_units_LSTM[1])

		# FC layers
		
		self.fc1 = nn.Linear(self.hidden_units_LSTM[self.numLSTMCells-1], 128)
		
		# self.fc1 = nn.Linear(1024, 128)
		self.fc2 = nn.Linear(128, 32)

		if self.parameterization != 'mahalanobis':
			if self.parameterization == 'quaternion':
				self.fc_rot = nn.Linear(32, 4)
			else:
				self.fc_rot = nn.Linear(32, 3)
			self.fc_trans = nn.Linear(32,3)
		else:
			# Mahalanobis norm parameterization, where loss is computed as Mahalanobis distance
			self.fc_out = nn.Linear(32, 6)


	def forward(self, x, reset_hidden = False):
		x_img = x[:,:6,:,:]
		# print (x_img.shape)

		x_event = x[:,6:,:,:]
		# print (x_event.shape)

		if not self.batchnorm:
			x = (F.leaky_relu(self.conv1(x_img)))
			x = (F.leaky_relu(self.conv2(x)))
			x = (F.leaky_relu(self.conv3(x)))
			x = (F.leaky_relu(self.conv3_1(x)))
			x = (F.leaky_relu(self.conv4(x)))
			x = (F.leaky_relu(self.conv4_1(x)))
			x = (F.leaky_relu(self.conv5(x)))
			x = (F.leaky_relu(self.conv5_1(x)))
			
			x = ((self.conv6(x)))

			# Stacking the output from the final conv layer
			# x = x.view(-1, self.numConcatFeatures)

			y = (F.leaky_relu(self.conv7(x_event)))
			y = (F.leaky_relu(self.conv8(y)))
			y = (F.leaky_relu(self.conv9(y)))
			y = (F.leaky_relu(self.conv9_1(y)))
			y = (F.leaky_relu(self.conv10(y)))
			y = (F.leaky_relu(self.conv10_1(y)))
			y = (F.leaky_relu(self.conv11(y)))
			y = (F.leaky_relu(self.conv11_1(y)))
			
			y = ((self.conv12(y)))

			# y = y.view(-1, self.numConcatFeatures)

			feat = torch.cat((x,y), 1)
			# print (feat.shape)

			x = self.conv13(F.leaky_relu(feat))
			x = self.conv14(x)
			# print (x.shape)
			x = x.view(-1, self.numConcatFeatures)
			
			if reset_hidden is True:
				if self.numLSTMCells == 1:
					self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
					self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
				else:
					self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
					self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
					self.h2 = torch.zeros(1, self.hidden_units_LSTM[1])
					self.c2 = torch.zeros(1, self.hidden_units_LSTM[1])
			
			if self.numLSTMCells == 1:
				self.h1, self.c1 = self.lstm1(x, (self.h1, self.c1))
			else:
				self.h1, self.c1 = self.lstm1(x, (self.h1, self.c1))
				self.h2, self.c2 = self.lstm2(self.h1, (self.h2, self.c2))

			# Forward pass through the FC layers
			if self.activation == 'relu':
				"""
				output_fc1 = F.relu(self.fc1(lstm_final_output))
				"""
				if self.numLSTMCells == 1:
					output_fc1 = F.relu(self.fc1(self.h1))
				else:
					output_fc1 = F.relu(self.fc1(self.h2))
				# output_fc1 = F.relu(self.fc1(self.h2))
				if self.dropout is True:
					output_fc2 = F.dropout(F.relu(self.fc2(output_fc1)), p = self.drop_ratio, \
						training = self.training)
				else:
					output_fc2 = F.relu(self.fc2(output_fc1))
			elif self.activation == 'selu':
				"""
				output_fc1 = F.selu(self.fc1(lstm_final_output))
				"""
				if self.numLSTMCells == 1:
					output_fc1 = F.selu(self.fc1(self.h1))
				else:
					output_fc1 = F.selu(self.fc1(self.h2))
				# output_fc1 = F.selu(self.fc1(self.h2))
				if self.dropout is True:
					output_fc2 = F.dropout(F.selu(self.fc2(output_fc1)), p = self.drop_ratio, \
						training = self.training)
				else:
					output_fc2 = F.selu(self.fc2(output_fc1))

			if self.parameterization == 'mahalanobis':
				output_ = self.fc_out(output_fc2)
				return output_, None

			output_rot = self.fc_rot(output_fc2)
			output_trans = self.fc_trans(output_fc2)

			return output_rot, output_trans

	# Initialize the weights of the network
	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				# print('# Linear')
				nn.init.xavier_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.Conv2d):
				# print('$ Conv2d')
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.LSTMCell):
				# print('% LSTMCell')
				for name, param in m.named_parameters():
					if 'weight' in name:
						nn.init.orthogonal(param)
						# nn.init.xavier_normal_(param)
					elif 'bias' in name:
						nn.init.constant_(param, 0.)
						bias = getattr(m, name)
						n = bias.size(0)
						start, end = n // 4, n // 2
						bias.data[start:end].fill_(10.)

		# Special weight_init for rotation FCs
		if self.parameterization == 'mahalanobis':
			pass
		else:
			self.fc_rot.weight.data = self.fc_rot.weight.data / 1000.
			# self.fc_trans.weight.data = self.fc_trans.weight.data * 100.


	# Detach LSTM hidden state (i.e., output) and cellstate variables to free up the
	# computation graph. Gradients will NOT flow backward through the timestep where a
	# detach is performed.
	def detach_LSTM_hidden(self):
		
		if self.numLSTMCells == 1:
			self.h1 = self.h1.detach()
			self.c1 = self.c1.detach()
		else:
			self.h1 = self.h1.detach()
			self.c1 = self.c1.detach()
			self.h2 = self.h2.detach()
			self.c2 = self.c2.detach()


	def reset_LSTM_hidden(self):
		if self.numLSTMCells == 1:
			self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
			self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
		else:
			self.h1 = torch.zeros(1, self.hidden_units_LSTM[0])
			self.c1 = torch.zeros(1, self.hidden_units_LSTM[0])
			self.h2 = torch.zeros(1, self.hidden_units_LSTM[1])
			self.c2 = torch.zeros(1, self.hidden_units_LSTM[1])


	def load_flownet_weights(self):
		if self.use_flownet is True:
			flownet = torch.load(self.flownet_weights_path)
			cnn = flownet['state_dict']
			if self.batchnorm is False:

				self.conv1.weight.data = cnn["conv1.0.weight"]
				self.conv1.bias.data = cnn["conv1.0.bias"]

				self.conv2.weight.data = cnn["conv2.0.weight"]
				self.conv2.bias.data = cnn["conv2.0.bias"]

				self.conv3.weight.data = cnn["conv3.0.weight"]
				self.conv3.bias.data = cnn["conv3.0.bias"]

				self.conv3_1.weight.data = cnn["conv3_1.0.weight"]
				self.conv3_1.bias.data = cnn["conv3_1.0.bias"]

				self.conv4.weight.data = cnn["conv4.0.weight"]
				self.conv4.bias.data = cnn["conv4.0.bias"]

				self.conv4_1.weight.data = cnn["conv4_1.0.weight"]
				self.conv4_1.bias.data = cnn["conv4_1.0.bias"]

				self.conv5.weight.data = cnn["conv5.0.weight"]
				self.conv5.bias.data = cnn["conv5.0.bias"]

				self.conv5_1.weight.data = cnn["conv5_1.0.weight"]
				self.conv5_1.bias.data = cnn["conv5_1.0.bias"]

				self.conv6.weight.data = cnn["conv6.0.weight"]
				self.conv6.bias.data = cnn["conv6.0.bias"]

			else:
				self.conv1.weight.data = cnn["conv1.0.weight"]
				self.conv1_bn.weight.data = cnn["conv1.1.weight"]
				self.conv1_bn.bias.data = cnn["conv1.1.bias"]
				self.conv1_bn.running_mean.data = cnn["conv1.1.running_mean"]
				self.conv1_bn.running_var.data = cnn["conv1.1.running_var"]

				self.conv2.weight.data = cnn["conv2.0.weight"]
				self.conv2_bn.weight.data = cnn["conv2.1.weight"]
				self.conv2_bn.bias.data = cnn["conv2.1.bias"]
				self.conv2_bn.running_mean.data = cnn["conv2.1.running_mean"]
				self.conv2_bn.running_var.data = cnn["conv2.1.running_var"]

				self.conv3.weight.data = cnn["conv3.0.weight"]
				self.conv3_bn.weight.data = cnn["conv3.1.weight"]
				self.conv3_bn.bias.data = cnn["conv3.1.bias"]
				self.conv3_bn.running_mean.data = cnn["conv3.1.running_mean"]
				self.conv3_bn.running_var.data = cnn["conv3.1.running_var"]

				self.conv3_1.weight.data = cnn["conv3_1.0.weight"]
				self.conv3_1_bn.weight.data = cnn["conv3_1.1.weight"]
				self.conv3_1_bn.bias.data = cnn["conv3_1.1.bias"]
				self.conv3_1_bn.running_mean.data = cnn["conv3_1.1.running_mean"]
				self.conv3_1_bn.running_var.data = cnn["conv3_1.1.running_var"]

				self.conv4.weight.data = cnn["conv4.0.weight"]
				self.conv4_bn.weight.data = cnn["conv4.1.weight"]
				self.conv4_bn.bias.data = cnn["conv4.1.bias"]
				self.conv4_bn.running_mean.data = cnn["conv4.1.running_mean"]
				self.conv4_bn.running_var.data = cnn["conv4.1.running_var"]

				self.conv4_1.weight.data = cnn["conv4_1.0.weight"]
				self.conv4_1_bn.weight.data = cnn["conv4_1.1.weight"]
				self.conv4_1_bn.bias.data = cnn["conv4_1.1.bias"]
				self.conv4_1_bn.running_mean.data = cnn["conv4_1.1.running_mean"]
				self.conv4_1_bn.running_var.data = cnn["conv4_1.1.running_var"]

				self.conv5.weight.data = cnn["conv5.0.weight"]
				self.conv5_bn.weight.data = cnn["conv5.1.weight"]
				self.conv5_bn.bias.data = cnn["conv5.1.bias"]
				self.conv5_bn.running_mean.data = cnn["conv5.1.running_mean"]
				self.conv5_bn.running_var.data = cnn["conv5.1.running_var"]

				self.conv5_1.weight.data = cnn["conv5_1.0.weight"]
				self.conv5_1_bn.weight.data = cnn["conv5_1.1.weight"]
				self.conv5_1_bn.bias.data = cnn["conv5_1.1.bias"]
				self.conv5_1_bn.running_mean.data = cnn["conv5_1.1.running_mean"]
				self.conv5_1_bn.running_var.data = cnn["conv5_1.1.running_var"]

				self.conv6.weight.data = cnn["conv6.0.weight"]
				self.conv6_bn.weight.data = cnn["conv6.1.weight"]
				self.conv6_bn.bias.data = cnn["conv6.1.bias"]
				self.conv6_bn.running_mean.data = cnn["conv6.1.running_mean"]
				self.conv6_bn.running_var.data = cnn["conv6.1.running_var"]

		return cnn