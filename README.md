# Time-Contrastive Networks

An pytorch implementation a time-contrastive networks model as presented in the paper "Time-Contrastive Networks: Self-Supervised Learning from Multi-View Observation" (Sermanet Et al. 2017). The original Tensorflow code can be found [here](https://github.com/tensorflow/models/tree/master/research/tcn)

Videos are added to the `data/train` and `data/validation` directories.

Training the model is done by running the `train_tcn.py` script.

Evaluating the model is done by running the `evalnn_tcn.py` script.

### Todo list
- [ ] Test the code to get the results mentioned in the paper
- [ ] Merge the SingleView and MultiView triplet builders to inherit from a common super class
- [ ] Work on transferring learnt representation to Robot Actions