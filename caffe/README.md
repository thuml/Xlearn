# transfer-caffe

This is a caffe repository for transfer learning. We fork the repository with version ID `29cdee7` from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Add `mmd layer` described in paper "Learning Transferable Features with Deep Adaptation Networks".
- Add `jmmd layer` described in paper "Deep Transfer Learning with Joint Adaptation Networks".
- Add `entropy layer` and `outerproduct layer` described in paper "Unsupervised Domain Adaptation with Residual Transfer Networks".
- Copy `grl layer` and `messenger.hpp` from repository [Caffe](https://github.com/ddtm/caffe/tree/grl).
- Emit `SOLVER_ITER_CHANGE` message in `solver.cpp` when `iter_` changes.

If you have any problem about this code, feel free to concact us with the following email:
- zhuhan10@gmail.com
- caozhangjie14@gmail.com
- liushichen95@gmail.com
- longmingsheng@gmail.com

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

Citation
---------------
    @inproceedings{DBLP:conf/icml/LongC0J15,
      author    = {Mingsheng Long and
                   Yue Cao and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Learning Transferable Features with Deep Adaptation Networks},
      booktitle = {Proceedings of the 32nd International Conference on Machine Learning,
                   {ICML} 2015, Lille, France, 6-11 July 2015},
      pages     = {97--105},
      year      = {2015},
      crossref  = {DBLP:conf/icml/2015},
      url       = {http://jmlr.org/proceedings/papers/v37/long15.html},
      timestamp = {Tue, 12 Jul 2016 21:51:15 +0200},
      biburl    = {http://dblp2.uni-trier.de/rec/bib/conf/icml/LongC0J15},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
    
    @inproceedings{DBLP:conf/nips/LongZ0J16,
      author    = {Mingsheng Long and
                   Han Zhu and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Unsupervised Domain Adaptation with Residual Transfer Networks},
      booktitle = {Advances in Neural Information Processing Systems 29: Annual Conference
                   on Neural Information Processing Systems 2016, December 5-10, 2016,
                   Barcelona, Spain},
      pages     = {136--144},
      year      = {2016},
      crossref  = {DBLP:conf/nips/2016},
      url       = {http://papers.nips.cc/paper/6110-unsupervised-domain-adaptation-with-residual-transfer-networks},
      timestamp = {Fri, 16 Dec 2016 19:45:58 +0100},
      biburl    = {http://dblp.uni-trier.de/rec/bib/conf/nips/LongZ0J16},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
    
    @inproceedings{DBLP:conf/icml/LongZ0J17,
      author    = {Mingsheng Long and
                   Han Zhu and
                   Jianmin Wang and
                   Michael I. Jordan},
      title     = {Deep Transfer Learning with Joint Adaptation Networks},
      booktitle = {Proceedings of the 34th International Conference on Machine Learning,
               {ICML} 2017, Sydney, NSW, Australia, 6-11 August 2017},
      pages     = {2208--2217},
      year      = {2017},
      crossref  = {DBLP:conf/icml/2017},
      url       = {http://proceedings.mlr.press/v70/long17a.html},
      timestamp = {Tue, 25 Jul 2017 17:27:57 +0200},
      biburl    = {http://dblp.uni-trier.de/rec/bib/conf/icml/LongZ0J17},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }

