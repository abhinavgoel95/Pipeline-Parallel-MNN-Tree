import sys
import time
import zmq
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pickle
import statistics

parser = argparse.ArgumentParser(description='Arguements')
parser.add_argument('--num_frames', type=int, help='number of frames to process (100,300,500, 1000)', default=100)
parser.add_argument('--ip', type=str, help='number of frames to process (100,300,500, 1000)', default='192.168.1.4')
args = parser.parse_args()

context = zmq.Context()



# Socket to receive messages on
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5558")

sender = context.socket(zmq.PUSH)
sender.connect("tcp://"+args.ip+":5557")

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
testset = datasets.ImageFolder(root='SavedCifarImages', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
print("Press Enter when all devices are ready: ")
_ = input()

# Wait for start of batch

times = []
count = 0
for frame_number, (image, label) in enumerate(testloader):
    # Start our clock now
    current_DNN = 'root'
    tstart = time.time()
    data = [frame_number, current_DNN, image]
    data = pickle.dumps(data)
    sender.send(data)

    # Process 100 confirmations
    s = receiver.recv()
    data = pickle.loads(data)
    frame_number1, image = data[0], data[1]
    print(frame_number1)
    # Calculate and report duration of batch
    tend = time.time()
    print("Total elapsed time: %d msec" % ((tend-tstart)*1000))
    times.append(tend - tstart)
    count+=1
    if count > 100:
        break

print(sum(times))
print(statistics.mean(times), "+-", statistics.stdev(times))