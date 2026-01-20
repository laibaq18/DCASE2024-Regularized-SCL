import torch
from torch import nn
from tqdm import tqdm
from losses import SupConLoss
from augmentations import RandomCrop, Resize, Compander, GaussNoise, FreqShift, MixRandom, SpecAugment
from models import ResNet
from torchinfo import summary
from args import args
import math
import h5py
import os

def train_scl(encoder, train_loader, transform1, transform2, transform3, args):

    print(f"Training starting on {args.device}")
    
    loss_fn = SupConLoss(temperature=args.tau, device=args.device, usetcr=args.usetcr)
    
    optim = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    num_epochs = args.epochs

    ckpt_dir = os.path.join(args.traindir, '../model/')
    os.makedirs(ckpt_dir, exist_ok=True) 
    last_model_path = os.path.join(ckpt_dir, f'{args.model_name}.pth')

    encoder = encoder.to(args.device)
    
    for epoch in range(1, num_epochs+1):
        tr_loss = 0.
        print("Epoch {}".format(epoch))
        adjust_learning_rate(optim, args.lr, epoch, num_epochs+1)
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            
            # build a list of transforms (V views)
            transforms = [transform1, transform2] # by default 2 views with 2 transforms

            if args.nTrainingViews == 4:
                transforms.append(transform1)  # view3: reuse transform1
                transforms.append(transform2)  # view4: reuse transform2

            if (args.nTransforms == 3) and (args.nTrainingViews == 3):
                transforms.append(transform3)  # add new view for transform3

            # generate V augmented views
            views = [t(x) for t in transforms[:args.nTrainingViews]]  # each is [B, 1, nmels, T]

            # encode each view -> embeddings [B, D]
            outs = []
            for v in views:
                _, z = encoder(v)    # z: [B, D]
                outs.append(z)

            # stack -> [B, V, D]
            z_feats = torch.stack(outs, dim=1)

            # compute loss for scl
            loss = loss_fn(z_feats, y)

            tr_loss += loss.item()

            loss.backward()
            optim.step()

        tr_loss = tr_loss/len(train_iterator)
        print('Average train loss: {}'.format(tr_loss))

    torch.save({'encoder':encoder.state_dict()}, last_model_path)

    return encoder

def adjust_learning_rate(optimizer, init_lr, epoch, tot_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / tot_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == "__main__":

    # Load data
    hdf_tr = os.path.join(args.traindir, 'train.h5')
    hdf_train = h5py.File(hdf_tr, 'r+')
    X = hdf_train['data'][:]
    Y = hdf_train['label'][:]

    # Create dataset
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).unsqueeze(1), torch.tensor(Y.squeeze(), dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    # Data augmentation
    time_steps = int(args.sr / (1000/args.len) / args.hoplen)
    rc = RandomCrop(n_mels=args.nmels, time_steps=time_steps, tcrop_ratio=args.tratio)
    resize = Resize(n_mels=args.nmels, time_steps=time_steps)
    awgn = GaussNoise(stdev_gen=args.noise)
    comp = Compander(comp_alpha=args.comp)
    mix = MixRandom()
    fshift = FreqShift(Fshift=args.fshift)

    # Prepare views
    time_mask_param = max(1, int(time_steps * 0.2))
    freq_mask_param = max(1, int(args.nmels * 0.1))
    spec_aug = SpecAugment(time_mask_param=time_mask_param, freq_mask_param=freq_mask_param)
    
    transform1 = nn.Sequential(mix, fshift, rc, resize, comp, awgn) # only one branch has mixing with a background sound
    transform2 = nn.Sequential(fshift, rc, resize, comp, awgn)
    # adding a new view - transform3 with specaug
    transform3 = nn.Sequential(fshift, rc, resize, comp, spec_aug, awgn)
    
    # Prepare model
    encoder = ResNet(method=args.method)
    print(summary(encoder))

    # Launch training
    train_scl(encoder, train_loader, transform1, transform2, transform3, args)
