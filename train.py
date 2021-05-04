# -*- coding: utf-8 -*-
from __future__ import print_function, division

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle, random, json
from evaluation import *

use_aws = True

if use_aws:
    # S3 and SageMaker
#     import sagemaker
#     from sagemaker import get_execution_role
    import io
    import boto3

#     sagemaker_session = sagemaker.Session()
#     role = get_execution_role()
#     print(role)
    s3 = boto3.resource('s3')
    bucket_name = 'schen5-bucket01'
    bucket = s3.Bucket(bucket_name)
    prefix = 'train'

plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# Data augmentation and normalization for training
# Just normalization for validation

def make_labels():
    def mean(scores):
        return np.mean([v for v in scores.values()])
    def mean2(scores):
        out = []
        for v in scores.values():
            out += v
        return np.mean(out)
    if use_aws:
        response = s3.Object(bucket_name, 'labels.json').get()['Body'].read()
        labels_dic = json.loads(response)
    else:
        labels_path = os.path.join(data_dir, 'labels.json')
        labels_dic = json.load(open(labels_path, 'rb'))
    out = {}
    for file_name, scores in labels_dic.items():
        pinch = mean(scores['pinch'])
        clench = mean(scores['clench'])
        poke = mean(scores['poke'])
        palm = mean(scores['palm'])
        familiarity = mean2(scores['familiarity'])
        size = scores['size']
        out[file_name] = [pinch, clench, poke, palm, familiarity, size]
    return out

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    if use_aws:
        img_bytes = s3.Object(bucket_name, path).get()['Body'].read()
        # e.g. path: 'train/21JBED3YWoL._AC_.png'
        return Image.open(io.BytesIO(img_bytes)).convert('RGB')
    else:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return accimage_loader(path)
        else:
            return pil_loader(path)


def make_dataset(directory, labels_dic):
    images = []
    if use_aws:
        for object_summary in bucket.objects.filter(Prefix=directory):
            path = object_summary.key
            fname = object_summary.key[len(directory)+1:-4]
            try:
                item = (path, labels_dic[fname])
                images.append(item)
            except:
                continue
    else:
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in fnames:
                path = os.path.join(root, fname)
                try:
                    item = (path, labels_dic[fname[:-4]]) # remove .png
                    images.append(item)
                except:
                    continue
    print(len(images))
    random.Random(4).shuffle(images)
    return images

class DatasetFolder():
    def __init__(self, directory, transform):
        samples = make_dataset(directory, make_labels())
        self.loader = default_loader
        self.transform = transform
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        target = torch.Tensor(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

data_transforms = {
    'train': transforms.Compose([
        transforms.Pad(512),
        transforms.CenterCrop(512),
        transforms.Resize(224),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Pad(512),
        transforms.CenterCrop(512),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def train_val_idx(n, l):
    assert n >= 0
    start = int((l*0.2*n)%l)
    end = int((l*0.2*(n+1))%l)
    return start, end

def make_splitted_images(image_datasets, n):
    s, e = train_val_idx(n, len(image_datasets))

    if s < e:
        train = copy.deepcopy(image_datasets)
        train.samples = image_datasets.samples[:s]+image_datasets.samples[e:]
        train.targets = image_datasets.targets[:s]+image_datasets.targets[e:]

        val = copy.deepcopy(image_datasets)
        val.samples = image_datasets.samples[s:e]
        val.targets = image_datasets.targets[s:e]
    else:
        train = copy.deepcopy(image_datasets)
        train.samples = image_datasets.samples[e:s]
        train.targets = image_datasets.targets[e:s]

        val = copy.deepcopy(image_datasets)
        val.samples = image_datasets.samples[:e]+image_datasets.samples[s:]
        val.targets = image_datasets.samples[:e]+image_datasets.targets[s:]

    return {'train': train, 'val': val}

class affordance_model(nn.Module):
    def __init__(self, originalModel):
        super(affordance_model, self).__init__()
        self.features = nn.Sequential(*list(originalModel.features))
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout()
        )
        self.pinch = nn.Linear(4096, 10)
        self.clench = nn.Linear(4096, 10)
        self.poke = nn.Linear(4096, 10)
        self.palm = nn.Linear(4096, 10)
        w = torch.FloatTensor([[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w = w.to(device)

    def weighted_sum(self, x):
        return ((self.w * x).sum(-1, keepdim=True)-1)*100/9

    def forward(self, x, size):
        x = self.features(x)
        x = x.view(-1, 512*7*7)
        size = size.view(-1, 1)
        size = size / 512.
        #x = torch.cat((x, size), 1)
        x = self.classifier(x)

        pinch = F.softmax(self.pinch(x), dim=-1)
        clench = F.softmax(self.clench(x), dim=-1)
        poke = F.softmax(self.poke(x), dim=-1)
        palm = F.softmax(self.palm(x), dim=-1)

        pinch = self.weighted_sum(pinch)
        clench = self.weighted_sum(clench)
        poke = self.weighted_sum(poke)
        palm = self.weighted_sum(palm)

        return pinch, clench, poke, palm


data_dir = '/Users/sharon/data/EE148/affordance/Data/'
ckpt_dir = './ckpt/'

def run(n): # n is the CV fold idx
    if use_aws:
        images = DatasetFolder('train', data_transforms['train'])
    else:
        images = DatasetFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    print('sharon')
    print(len(images))
    image_datasets = make_splitted_images(images, n)
    #dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=1)
    if use_aws:
        num_workers = 4
    else:
        num_workers = 0
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ######################################################################
    # Testing the model predictions
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    #

    def test_model(model):
        was_training = model.training
        model.eval()

        with torch.no_grad():
            count = 0
            total_loss = 0
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                obj_size = labels[:, -1].to(device)
                labels = labels[:, :-2].to(device)

                pinch, clench, poke, palm = model(inputs, obj_size)

                pinch = pinch.cpu().numpy()
                clench = clench.cpu().numpy()
                poke = poke.cpu().numpy()
                palm = palm.cpu().numpy()

                if i == 0:
                    all_preds = np.concatenate((pinch, clench, poke, palm), axis=-1)
                    all_labels = labels.cpu().numpy()
                else:
                    preds = np.concatenate((pinch, clench, poke, palm), axis=-1)
                    all_preds = np.concatenate((all_preds, preds), axis=0)
                    labels = labels.cpu().numpy()
                    all_labels = np.concatenate((all_labels, labels), axis=0)


            mse, corr, acc = score_evaluation_from_np_batches(all_labels, all_preds)
            model.train(mode=was_training)
            return mse, corr, acc


    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        #iteration = 0
        no_update_count = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                sharon = 0
                # Iterate over data.
                print(len(dataloaders[phase]))
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    obj_size = labels[:, -1].to(device)
                    labels = labels[:, :-2].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    print(sharon)
                    sharon += 1
                    with torch.set_grad_enabled(phase == 'train'):
                        pinch, clench, poke, palm = model(inputs, obj_size)

                        loss1 = criterion(pinch, labels[:, 0:1])
                        loss2 = criterion(clench, labels[:, 1:2])
                        loss3 = criterion(poke, labels[:, 2:3])
                        loss4 = criterion(palm, labels[:, 3:4])

                        loss = loss1 + loss2 + loss3 + loss4

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    iter_loss = loss.item() * inputs.size(0)
                    running_loss += iter_loss

                    #if phase == 'train':
                    #    print('{} Iter: {}, IterLoss: {:.4f}'.format(phase, iteration, iter_loss))
                    #    iteration += 1

                epoch_loss = running_loss / dataset_sizes[phase]

                print('{} EpochLoss: {:.4f}'.format(phase, epoch_loss))

                if phase == 'train':
                    scheduler.step()
                # deep copy the model
                else:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        torch.save(model, os.path.join(ckpt_dir, '{}_fold_best.ckpt'.format(n)))
                        best_model_wts = copy.deepcopy(model.state_dict())
                        test_model(model)
                        no_update_count = 0

                    else:
                        no_update_count += 1

            print()

            if no_update_count >= 20:
                break
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #

    model_ft = models.vgg16(pretrained=True)

    model_ft = affordance_model(model_ft)

    model_ft = model_ft.to(device)

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False

    #set_parameter_requires_grad(model_ft.features)

    criterion = nn.SmoothL1Loss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=200)

    ######################################################################
    #

    mse, corr, acc  = test_model(model_ft)

    plt.ioff()
    plt.show()

    return mse, corr, acc

mse_list = []
corr_list = []
acc_list = []
for i in range(5):
    print(str(i)+'th fold')
    mse, corr, acc = run(i)
    mse_list.append(mse)
    corr_list.append(corr)
    acc_list.append(acc)

print("=====FINAL RESULT=====")
print("MSE: ", np.mean(mse_list), "Corr: ", np.mean(corr_list), "Acc: ", np.mean(acc_list))
