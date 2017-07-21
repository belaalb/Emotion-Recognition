from __future__ import print_function
import argparse
import torch
import models_zoo
import data_load
from train_loop import TrainLoop

# Training settings
parser = argparse.ArgumentParser(description = 'Online transfer learning for emotion recognition tasks')
parser.add_argument('--batch-size', type = int, default = 64, metavar = 'N', help = 'input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type = int, default = 1000, metavar = 'N', help = 'input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type = int, default = 500, metavar = 'N', help = 'number of epochs to train (default: 200)')
parser.add_argument('--patience', type = int, default = 30, metavar = 'N', help = 'number of epochs without improvement to wait before stopping training (default: 30)')
parser.add_argument('--lr', type = float, default = 0.001, metavar = 'LR', help = 'learning rate (default: 0.001)')
parser.add_argument('--l2', type = float, default = 0.001, metavar = 'lambda', help = 'L2 wheight decay coefficient (default: 0.001)')
parser.add_argument('--no-cuda', action = 'store_true', default = False, help = 'disables CUDA training')
parser.add_argument('--checkpoint-epoch', type = int, default = None, metavar = 'N', help = 'epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type = str, default = None, metavar = 'Path', help = 'Path for checkpointing')
parser.add_argument('--seed', type = int, default = 1, metavar = 'S', help = 'random seed (default: 1)')
parser.add_argument('--save-every', type = int, default = None, metavar = 'N', help = 'how many batches to wait before logging training status. If None, cp is done every epoch')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

train_loader = data_load.train_generator(args.batch_size)
valid_loader = data_load.valid_generator(args.valid_batch_size)

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = models_zoo.model_im()

if args.cuda:
	model.cuda()

optimizer = optim.Adam(lr = args.lr, weight_decay = args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, checkpoint_path = args.checkpoint_path, checkpoint_epoch = args.checkpoint_epoch, cuda = args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs = args.epochs, patience = args.patience, tl_delay = args.tl_delay save_every=args.save_every)