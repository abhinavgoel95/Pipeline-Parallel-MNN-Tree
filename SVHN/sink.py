import sys
import time
import zmq
import argparse

parser = argparse.ArgumentParser(description='Arguements')
parser.add_argument('--num_frames', type=int, help='number of frames to process (100,300,500, 1000)', default=100)
args = parser.parse_args()

context = zmq.Context()

# Socket to receive messages on
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5558")

# Wait for start of batch
s = receiver.recv()

# Start our clock now
tstart = time.time()

# Process 100 confirmations
for task_nbr in range(args.num_frames - 1):
    s = receiver.recv()

# Calculate and report duration of batch
tend = time.time()
print("Total elapsed time: %d msec" % ((tend-tstart)*1000))
