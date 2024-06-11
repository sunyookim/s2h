import argparse

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--id', type=str, default='temp', help='a name for identifying the model')
		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
		self.parser.add_argument('--data_dir', type=str, default='/home/sunyoo/data/dataset/audiovisual', help='root dir')
		self.parser.add_argument('--dataset', type=str, default='MUSIC', help='dataset')
		self.parser.add_argument('--detr_dataset', type=str, default='openimages_instruments', help='detr dataset')
		self.parser.add_argument('--pretrain_detr', action='store_true', help='whether to use DETR pretrained on OpenImages or Roboflow')
		self.parser.add_argument('--pretrain_encoders', action='store_true', help='whether to use pretrained visual/audio encoders')
		self.parser.add_argument('--mode', type=str, default='train', help='which model to use.')
		self.parser.add_argument('--model_size', type=str, default='large', help='ast_detr model size.')
		self.parser.add_argument('--weights', type=str, default='', help='which model to use.')
		self.parser.add_argument('--seed', default=1027, type=int, help='random seed')
		self.parser.add_argument('--num_workers', default=16, type=int, help='number of data loading workers')
		
		#audio arguments
		self.parser.add_argument('--audio_window', default=65535, type=int, help='audio segment length')
		self.parser.add_argument('--audRate', default=11025, type=int, help='sound sampling rate')
		self.parser.add_argument('--audLen', default=65535, type=int, help='sound length for MUSIC')
		self.parser.add_argument('--stft_frame', default=1022, type=int, help="stft frame length")
		self.parser.add_argument('--stft_hop', default=256, type=int, help="stft hop length")
		
		#training arguments
		self.parser.add_argument('--epochs', type=int, default=200, help='total epochs')
		self.parser.add_argument('--dup_trainset', default=40, type=int, help='duplicate so that one epoch has more iters')
		self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
		self.parser.add_argument('--num_mix', default=2, type=int, help='number of video clips to mix')		
		self.parser.add_argument('--num_frames', type=int, default=3, help='number of frames to sample for detection with features')
		self.parser.add_argument('--max_width', type=int, default=256)
		self.parser.add_argument('--frameRate', default=1, type=float,help='video frame sampling rate') 
		self.parser.add_argument('--stride_frames', default=1, type=int, help='sampling stride of frames')
		self.parser.add_argument('--disp_iter', type=int, default=10, help='frequency to display')
		
		#model arguments
		self.parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
		self.parser.add_argument('--freeze_encoders', action='store_true', help='whether to freeze AST-DETR encoders')

		#optimizer arguments
		self.parser.add_argument('--coseparation_loss_weight', default=1, type=float, help='weight for reconstruction loss')
		self.parser.add_argument('--bbox_loss_weight', default=0.1, type=float, help='weight for reconstruction loss')
		self.parser.add_argument('--lr_backbone', type=float, default=0.00001, help='learning rate for visual encoder backbone')
		self.parser.add_argument('--lr_visual', type=float, default=0.00002, help='learning rate for visual stream')
		self.parser.add_argument('--lr_audio', type=float, default=0.00005, help='learning rate for audiovisual stream')
		self.parser.add_argument('--lr_audiovisual', type=float, default=0.0001, help='learning rate for audiovisual stream')
		self.parser.add_argument('--lr_unet', type=float, default=0.0001, help='learning rate for audiovisual stream')
		self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[80, 100], help='steps to drop LR in epochs')
		
		#other arguments
		self.parser.add_argument('--output_visuals', action='store_true', help='whether to output visualization')

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')
		return self.opt
