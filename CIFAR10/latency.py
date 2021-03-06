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
import CifarDataset
import pdb
import pickle
from configuration import configuration
from hierarchy import hierarchy_structure
from communication import communication
import random
import cifar

def process_data():
    if args.index == 1:
        while 1:
            data = recv_from_src.recv()
            data = pickle.loads(data)
            frame_number, current_DNN, image = data[0], data[1], data[2]
            current_DNN = 'root'
            model = models[current_DNN]
            model.eval()
            image, net_out = model(image)
            output = net_out.max(1, keepdim=True)[1].item()
            current_DNN, next_device = hierarchy.getNext(current_DNN, output)
            while next_device == args.index:
                model = models[current_DNN]
                model.eval()
                image, net_out = model(image)
                output = net_out.max(1, keepdim=True)[1].item()
                current_DNN, next_device = hierarchy.getNext(current_DNN, output)
            
            if current_DNN == None:
                data = [frame_number, output]
                data = pickle.dumps(data)
                send_to_sink.send(data)
            else:
                data = [frame_number, current_DNN, image]
                data = pickle.dumps(data)
                senders[next_device].send(data)
    else:
        while True:
            socks = dict(poller.poll())
            for receiver in receivers:
                if receiver in socks:
                    data = receiver.recv()
                    data = pickle.loads(data)
                    frame_number, current_DNN, image = data[0], data[1], data[2]
                    #print("\n\nrecieved: ", frame_number)
                    model = models[current_DNN]
                    model.eval()
                    image, net_out = model(image)
                    output = net_out.max(1, keepdim=True)[1].item()
                    current_DNN, next_device = hierarchy.getNext(current_DNN, output)
                    while next_device == args.index:
                        model = models[current_DNN]
                        model.eval()
                        image, net_out = model(image)
                        output = net_out.max(1, keepdim=True)[1].item()
                        current_DNN, next_device = hierarchy.getNext(current_DNN, output)
                    
                    if current_DNN == None:
                        #print("frame number: ", frame_number, " output: ", output)
                        data = [frame_number, output]
                        data = pickle.dumps(data)
                        send_to_sink.send(data)
                    else:
                        data = [frame_number, current_DNN, image]
                        data = pickle.dumps(data)
                        senders[next_device].send(data)
                        #print("frame number: ", frame_number, " output: ", output, " sent to: ", next_device)
                    

parser = argparse.ArgumentParser(description='Arguements')
parser.add_argument('--N', type=int, help='Number of devices')
parser.add_argument('--index', type=int, help='My device index (one indexing)')
parser.add_argument('--ips', nargs='+', help='list of children device IP addresses', default=[])
parser.add_argument('--data', help='path to dataset', default='SavedCifarImages')
parser.add_argument('--num_frames', help='number of frames to process (100,300,500, 1000)', default='100')

args = parser.parse_args()


config = configuration(args.N, args.index, args.ips)
recv_list, send_list, allocations = config.getAssignment()
hierarchy = hierarchy_structure(config)
comm = communication(args.N)
args.data = os.path.join(args.data, args.num_frames)
context = zmq.Context()

send_to_sink = context.socket(zmq.PUSH)
send_to_sink.connect("tcp://"+args.ips[-1]+":5558")

if args.index == 1:
    recv_from_src = context.socket(zmq.PULL)
    recv_from_src.bind("tcp://*:5557")



models = {}
for DNN in allocations:
    models[DNN] = getattr(cifar, "get_"+DNN+"_model")()
    models[DNN].load_state_dict(torch.load('models/'+DNN+'.pth', map_location=torch.device('cpu')))


senders = {}
for child in set(send_list):
    sender = context.socket(zmq.PUSH)
    port = comm.getPort(args.index, child[0])
    sender.bind("tcp://"+args.ips[args.index-1]+":"+str(port))
    senders[child[0]] = sender

receivers = []
poller = zmq.Poller() #use poll becuase we may have multiple senders to one device
for parent in set(recv_list):
    port = comm.getPort(args.index, parent[0])
    receivers.append(context.socket(zmq.PULL))
    print("tcp://"+parent[1]+":"+str(port))
    receivers[-1].connect("tcp://"+parent[1]+":"+str(port))
    poller.register(receivers[-1], zmq.POLLIN)
process_data()
