# SHMnet
Python Code for SHMnet, a 1D CNN algorithm for struactual health mornitoring
SHMnet: Condition Assessment of Bolted Connection with Beyond Human-level Performance
Abstract:
Deep learning algorithms are transforming a variety of research areas with accuracy levels that the traditional methods cannot compete with. Recently, increasingly more research efforts have been put into the structural health monitoring (SHM) domain. In this work, we propose a new deep convolutional neural network, namely SHMnet, for a challenging structural condition identification case, i.e. steel frame with bolted connection damage. We perform systematic studies on the optimization of network architecture and the preparation of the training data. In the laboratory, repeated impact hammer tests are conducted on a steel frame with different bolted connection damage scenarios, as small as one bolt loosened. The time-domain monitoring data from a single accelerometer are used for training. We conduct parametric studies on different sensor locations, the quantity of the training data sets, and noise levels. The results show that the proposed SHMnet is effective and reliable with at least four independent training data sets and by avoiding vibration node points as sensor locations. Under up to 60% addictive Gaussian noise, the average identification accuracy is over 98%. In comparison, the traditional methods based on the identified modal parameters inevitably fail due to the unnoticeable changes of identified natural frequencies and mode shapes. The results provide confidence in using the developed method as an effective structural condition identification framework. It has the potential to transform the SHM practice.

# Getting Started
Running train.py to repeat the training, and test_noise.py for addictive noise testing procesure reported in SHMnet.

It requires the Python 3.6 and Pytorch 0.4 versions and numpy, scipy, and matplotlib.

Once everything is set up you can simply run:

python train.py//
python test_noise.py
# Authors & Citation
Tong Zhang,
Suryakanta Biswal,
Ying Wang

