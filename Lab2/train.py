'''Train and evaluate the optic disc segmentation network (Practical 2, Step 3).

Example:
    python3 train.py --run_name adam_1e-3 --optimizer adam --lr 1e-3 --epochs 50
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import json
import random
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from disc_segmentor import OpticDiscSegmenter
from metrics import dice_coefficient


parser = argparse.ArgumentParser(description='Optic disc segmentation training')
parser.add_argument('--run_name', default='default', type=str, help='name of output folder in runs/')
parser.add_argument('--data_root', default='../Messidor_Processed', type=str,
                    help='output folder of proc_messidor.py')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--seed', default=42, type=int, help='seed for training')
parser.add_argument('--num_workers', default=2, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
run_dir = os.path.join('runs', args.run_name)
os.makedirs(run_dir, exist_ok=True)
ckpt_path = os.path.join(run_dir, 'best.pth')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Data
class MessidorSegDataset(Dataset):
    '''Preprocessed MESSIDOR split written by proc_messidor.py.

    Expected layout:
        <split_dir>/images/<name>.png
        <split_dir>/masks/<name>_mask.png
    '''
    def __init__(self, split_dir):
        img_dir = os.path.join(split_dir, 'images')
        self.image_paths = [os.path.join(img_dir, f)
                            for f in sorted(os.listdir(img_dir)) if f.endswith('.png')]
        self.mask_paths = []
        for p in self.image_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            self.mask_paths.append(os.path.join(split_dir, 'masks', f'{stem}_mask.png'))
        if not self.image_paths:
            raise RuntimeError(f'No images found in {img_dir}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.image_paths[idx]).convert('RGB'), dtype=np.float32) / 255.
        mask = (np.asarray(Image.open(self.mask_paths[idx]).convert('L')) > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)        # [1, H, W]

        return image, mask


print('==> Preparing data..')
trainset = MessidorSegDataset(os.path.join(args.data_root, 'train'))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

valset = MessidorSegDataset(os.path.join(args.data_root, 'val'))
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

testset = MessidorSegDataset(os.path.join(args.data_root, 'test'))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print(f'train={len(trainset)} | val={len(valset)} | test={len(testset)}')

# Model
print('==> Building model..')
net = OpticDiscSegmenter().to(device)
if device == 'cuda':
    cudnn.benchmark = True
print(f'trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}')

criterion = nn.BCEWithLogitsLoss()  # model outputs logits
if args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Training
def train_one_epoch():
    net.train()
    total_loss = 0.0
    for inputs, masks in trainloader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = net(inputs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(trainloader.dataset)


def evaluate(loader):
    '''Returns (mean loss, mean Dice, list of per-image Dice).'''
    net.eval()
    total_loss = 0.0
    dices = []
    with torch.no_grad():
        for inputs, masks in loader:
            inputs, masks = inputs.to(device), masks.to(device)
            logits = net(inputs)
            total_loss += criterion(logits, masks).item() * inputs.size(0)
            dices += dice_coefficient(torch.sigmoid(logits), masks).tolist()
    return total_loss / len(loader.dataset), float(np.mean(dices)), dices


history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'lr': []}
best_dice, best_epoch = -1.0, -1

for epoch in range(args.epochs):
    lr = optimizer.param_groups[0]['lr']
    train_loss = train_one_epoch()
    val_loss, val_dice, _ = evaluate(valloader)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice)
    history['lr'].append(lr)

    saved = ''
    if val_dice > best_dice:
        best_dice, best_epoch = val_dice, epoch
        torch.save({'net': net.state_dict(), 'epoch': epoch, 'val_dice': val_dice}, ckpt_path)
        saved = ' | saved'
    print(f'Epoch {epoch:03d} | lr={lr:.4g} | train_loss={train_loss:.4f} | '
          f'val_loss={val_loss:.4f} | val_dice={val_dice:.4f}{saved}', flush=True)


# Test with the best checkpoint
print('==> Evaluating best checkpoint on test set..')
net.load_state_dict(torch.load(ckpt_path, map_location=device)['net'])
test_loss, test_dice, test_dices = evaluate(testloader)
print(f'best_epoch={best_epoch} | best_val_dice={best_dice:.4f} | '
      f'test_loss={test_loss:.4f} | test_dice={test_dice:.4f}')

with open(os.path.join(run_dir, 'results.json'), 'w') as f:
    json.dump({
        'args': vars(args),
        'history': history,
        'best_epoch': best_epoch,
        'best_val_dice': best_dice,
        'test_loss': test_loss,
        'test_dice': test_dice,
        'test_dice_per_image': test_dices,
    }, f, indent=2)
