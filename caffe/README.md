# transfer-caffe

This is a caffe repository for transfer learning. We fork the repository with version ID `29cdee7` from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Add `mmd layer` described in paper "Learning Transferable Features with Deep Adaptation Networks".
- Add `jmmd layer` described in paper "Deep Transfer Learning with Joint Adaptation Networks".
- Add `entropy layer` and `outerproduct layer` described in paper "Unsupervised Domain Adaptation with Residual Transfer Networks".
- Copy `grl layer` and `messenger.hpp` from repository [Caffe](https://github.com/ddtm/caffe/tree/grl).
- Emit `SOLVER_ITER_CHANGE` message in `solver.cpp` when `iter_` changes.

Data Preparation
---------------
In `data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset.

We have published the Image-Clef dataset we use [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view?usp=sharing).

Training Model
---------------

In `models/DAN/amazon_to_webcam`, we give an example model based on Alexnet to show how to transfer from `amazon` to `webcam`. In this model, we insert mmd layers after fc7 and fc8 individually.

The [bvlc\_reference\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) is used as the pre-trained model. If the Office dataset and pre-trained caffemodel are prepared, the example can be run with the following command:
```
"./build/tools/caffe train -solver models/DAN/amazon_to_webcam/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
```

Resnet pre-trainded model is [here](https://github.com/KaimingHe/deep-residual-networks). We use Resnet-50.

In `models/JAN`, we give example models based on Alexnet and Resnet to show how to transfer from `amazon` to `webcam`, according to "Deep Transfer Learning with Joint Adaptation Networks". The shell scripts to train JAN model are the same with the above command, except that the paths of `solver.prototxt` and pretrained `caffemodel` are different.

As JAN uses the joint distribution of features (e.g. fc7) and softmax output (fc8) which are randomized at the beginning of training, one should stablize training by adding `grl layer` before `JMMD Loss Layer` to increase the loss weight from zero gradually, as suggested in the `train_val.prototxt`. 

Parameter Tuning
---------------
In mmd-layer and jmmd-layer, parameter `loss_weight` can be tuned to give mmd/jmmd loss different weights.

