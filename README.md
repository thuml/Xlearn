# Xlearn
**Transfer Learning Library**<br>
<br>
The transfer learning library for the following paper:<br>
* [Learning Transferable Features with Deep Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) **(DAN)**
* [Unsupervised Domain Adaptation with Residual Transfer Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/residual-transfer-network-nips16.pdf) **(JAN)**
* [Deep Transfer Learning with Joint Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf) **(RTN)** <br>

This package will include **Caffe,PyTorch and TensorFlow** implementations for transfer learning.<br> 
The pytorch and tensorflow versions are under developing.<br>
The code was written by [Han Zhu](https://github.com/zhuhan1236), [Zhangjie Cao](https://github.com/caozhangjie), [Shichen Liu](https://github.com/ShichenLiu) and [Mingsheng Long](https://github.com/longmingsheng).<br>
## Applications
## Getting Started
Refer to [Xlearn on Caffe](https://github.com/thuml/Xlearn/blob/master/caffe/README.md) to get started with our [Caffe version](https://github.com/thuml/Xlearn/tree/master/caffe).<br>
## Datasets
In `caffe/data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset.<br>
<br>
We have published the Image-Clef dataset we use [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view?usp=sharing).<br>
## Citation
If you use this code for your research, please consider citing:<br>
```
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
```
## Contact
If you have any problem about our code, feel free to contact<br>

* zhuhan10@gmail.com
* caozhangjie14@gmail.com
* liushichen95@gmail.com
* longmingsheng@gmail.com<br>

or describe your problem in Issues.
