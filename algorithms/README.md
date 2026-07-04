# Algorithms

Each algorithm gets its own folder with code, run notes, and method-specific
configuration.

The intended progression is:

1. `fedavg_softmax_sanity`: lightweight sanity baseline already used to verify
   frozen manifest loading, FedAvg, communication accounting, and metrics.
2. `1d_cnn_fedavg`: first neural FedAvg baseline using raw 3-axis accelerometer
   windows.
3. `fedprox`: FedProx extension of the tuned 1D-CNN FedAvg baseline.
4. Later folders for SCAFFOLD, FedRep, Ditto, FedBN, FedProto, and
   compression variants.
