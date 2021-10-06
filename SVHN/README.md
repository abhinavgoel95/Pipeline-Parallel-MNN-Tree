# MNN-Tree for SVHN Dataset:

## Directory Structure
`svhn/`: Contains PyTorch DNN models for each node of the MNN-Tree.

`models/`: Contains Trained PyTorch weights and parameters for each node.

`pipeline.py`: Performs pipeline parallel inference.

`no_pipeline.py`: Performs regular single-device inference.

`sink.py`: Collects all outputs and calculates throughput.

`source_sink.py`: Collects all outputs and calculates latency.

#### Helper Functions:

`SVHNDataset.py`: Contains custom dataloader for MNN-Tree.

`communication.py`: Assigns port numbers to enable inter-device communication.

`configuration.py`: Contains information on how to partition MNN-Tree.

`hierarhy.py`: Contains hierarchy structure for the MNN-Tree.


## Running Pipeline Parallel MNN-Tree

Example for 3 devices:

Device 1:
`python pipeline.py --ips <device 1 IP> <device 2 IP> <device 3 IP> <sink device IP> --N 3 --index 1`

Device 2:
`python pipeline.py --ips <device 1 IP> <device 2 IP> <device 3 IP> <sink device IP> --N 3 --index 2`

Device 3:
`python pipeline.py --ips <device 1 IP> <device 2 IP> <device 3 IP> <sink device IP> --N 3 --index 3`

Sink Device (Can be run on any device. This is where all outputs are collected):
`python sink.py`

Please note: Images for inference must be downloaded and stored in a folder called `SavedSVHNImages`.


## Running Single Device MNN-Tree

`python no_pipeline.py`
