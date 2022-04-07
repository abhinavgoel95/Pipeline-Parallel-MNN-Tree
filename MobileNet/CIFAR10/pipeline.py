import sys
import os
import time
import zmq
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import mobilenetv2
import pickle
from communication import communication


def process_data():
    if args.index == 1:
        for frame_number, (image, label) in enumerate(testloader):
            for colab in range(2, args.N+1):
                if not frame_number%colab:
                    data = [frame_number, image]
                    data = pickle.dumps(data)
                    senders[colab].send(data)
                    continue
            model = mobilenetv2.MobileNetV2()
            model.eval()
            model.load_state_dict(torch.load("mobilenetV2.pth"))
            net_out = model(image)
            output = net_out.max(1, keepdim=True)[1].item()
            data = [frame_number, output]
            data = pickle.dumps(data)
            send_to_sink.send(data)


    else:
        while True:
            socks = dict(poller.poll())
            for receiver in receivers:
                if receiver in socks:
                    data = receiver.recv()
                    data = pickle.loads(data)
                    frame_number, image = data[0], data[1]
                    model = mobilenetv2.MobileNetV2()
                    model.eval()
                    model.load_state_dict(torch.load("mobilenetV2.pth"))
                    net_out = model(image)
                    output = net_out.max(1, keepdim=True)[1].item()
                    data = [frame_number, output]
                    data = pickle.dumps(data)
                    send_to_sink.send(data)
                    

parser = argparse.ArgumentParser(description='Arguements')
parser.add_argument('--N', type=int, help='Number of devices')
parser.add_argument('--index', type=int, help='My device index (one indexing)')
parser.add_argument('--ips', nargs='+', help='list of children device IP addresses', default=[])
parser.add_argument('--data', help='path to dataset', default='SavedCifarImages')
parser.add_argument('--num_frames', help='number of frames to process (100,300,500, 1000)', default='500')

args = parser.parse_args()

comm = communication(args.N)
args.data = os.path.join(args.data, args.num_frames)
context = zmq.Context()

send_to_sink = context.socket(zmq.PUSH)
send_to_sink.connect("tcp://"+args.ips[-1]+":5558")


if args.index == 1:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    senders = {}
    for i in range(2, args.N+1):
        sender = context.socket(zmq.PUSH)
        port = comm.getPort(i)
        sender.bind("tcp://"+args.ips[args.index-1]+":"+str(port))
        print("tcp://"+args.ips[args.index-1]+":"+str(port))
        senders[i] = sender
    testset = datasets.ImageFolder(root=args.data, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    print("Press Enter when all devices are ready: ")
    _ = input()


else:
    receivers = []
    poller = zmq.Poller() #use poll becuase we may have multiple senders to one device
    port = comm.getPort(args.index)
    receivers.append(context.socket(zmq.PULL))
    print("tcp://"+args.ips[0]+":"+str(port))
    receivers[-1].connect("tcp://"+args.ips[0]+":"+str(port))
    poller.register(receivers[-1], zmq.POLLIN)

process_data()
