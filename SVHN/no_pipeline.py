import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import time
import numpy as np
import shutil
import os
import argparse
import pdb
from collections import OrderedDict
import svhn


def test(valloader):
    #validation accuracy
    count1 = 0
    total1 = 0
    times = []
    for data, target in valloader:
        start = time.time()
        count1+=1
        path_taken = []
        model = getattr(svhn, "get_root_model")()
        #model.load_state_dict(torch.load('models/root.pth', map_location=torch.device('cpu')))
        while 1:
            model.eval()
            data, net_out = model(data)
            child = net_out.max(1, keepdim=True)[1]
            path_taken.append(child.item())
            if tuple(path_taken) not in paths:
                break
            next_dnn = paths[tuple(path_taken)]
            model = getattr(svhn, "get_"+next_dnn+"_model")()
            #model.load_state_dict(torch.load('models/'+next_dnn+'.pth', map_location=torch.device('cpu')))
        times.append(time.time() - start)
        if count1 >= 100:
            print(times)
            break

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='path to dataset', default='SavedSVHNImages')
args = parser.parse_args()


transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
testset = datasets.ImageFolder(root=args.data, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


paths = {
    (1,): 'SG1',
    (3,): 'SG2',
}
start = time.time()
test(testloader)
end = time.time()
print("Total elapsed time: %d msec" % ((end-start)*1000))

