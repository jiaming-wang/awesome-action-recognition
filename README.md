# Awesome Action Recognition: [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of action recognition and related area (e.g. object recognition, pose estimation) resources, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

### Video Representation

* [Representation Flow for Action Recognition](https://arxiv.org/pdf/1810.01455.pdf) - AJ. Piergiovanni and M. S. Ryoo et al., CVPR2019.
* [Action Recognition Zoo](https://github.com/coderSkyChen/Action_Recognition_Zoo) - 
Codes for popular action recognition models, written based on pytorch, verified on the something-something dataset.
* [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf) - K. Hara et al., CVPR2019. [[code]](https://github.com/kenshohara/3D-ResNets-PyTorch) 
* [ConvNet Architecture Search for Spatiotemporal Feature Learning](https://arxiv.org/abs/1708.05038) - D. Tran et al, arXiv2017. Note: Aka Res3D. [[code]](https://github.com/facebook/C3D): In the repository, C3D-v1.1 is the Res3D implementation.
* [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf) - J. Carreira et al, CVPR2017. [[code]](https://github.com/deepmind/kinetics-i3d)[[PyTorch code]](https://github.com/hassony2/kinetics_i3d_pytorch), [[another PyTorch code]](https://github.com/piergiaj/pytorch-i3d)
* [Temporal Convolutional Networks: A Unified Approach to Action Segmentation and Detection](https://arxiv.org/pdf/1611.05267.pdf) - C. Lea et al, CVPR 2017. [[code]](https://github.com/colincsl/TemporalConvolutionalNetworks)
* [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf) - L. Wang et al, arXiv 2016. [[code]](https://github.com/yjxiong/temporal-segment-networks)


#### Useful Code Repos on Video Representation Learning
* [[3D ResNet PyTorch]](https://github.com/kenshohara/3D-ResNets-PyTorch)
* [[PyTorch Video Research]](https://github.com/gsig/PyVideoResearch)
* [[M-PACT: Michigan Platform for Activity Classification in Tensorflow]](https://github.com/MichiganCOG/M-PACT)
* [[Inflated models on PyTorch]](https://github.com/hassony2/inflated_convnets_pytorch)
* [[I3D models transfered from Tensorflow to PyTorch]](https://github.com/hassony2/kinetics_i3d_pytorch)
* [[A Two Stream Baseline on Kinectics dataset]](https://github.com/gurkirt/2D-kinectics)
* [[MMAction]](https://github.com/open-mmlab/mmaction)

### Action Classification
* [Attentional Pooling for Action Recognition](https://arxiv.org/abs/1711.01467) - R. Girdhar and D. Ramanan, NIPS2017. [[code]](https://github.com/rohitgirdhar/AttentionalPoolingAction)
* [Hidden Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/pdf/1704.00389.pdf) - Y. Zhu et al, arXiv2017. [[code]](https://github.com/bryanyzhu/Hidden-Two-Stream)
* [Dynamic Image Networks for Action Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/bilen16a.pdf) - H. Bilen et al, CVPR2016. [[code]](https://github.com/hbilen/dynamic-image-nets) [[project web]](http://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/)

### Temporal Action Detection
* [Cascaded Boundary Regression for Temporal Action Detection](https://arxiv.org/abs/1705.01180) - Jiyang Gao et al., BMVC 2017 [[code](https://github.com/jiyanggao/CBR)]
* [Temporal Context Network for Activity Localization in Videos](https://arxiv.org/pdf/1708.02349.pdf) - X. Dai et al., ICCV2017.
* [Detecting the Moment of Completion: Temporal Models for Localising Action Completion](https://arxiv.org/abs/1710.02310) - F. Heidarivincheh et al., arXiv2017.
* [CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos](https://arxiv.org/abs/1703.01515/) - Z. Shou et al, CVPR2017. [[code]](https://bitbucket.org/columbiadvmm/cdc)
* [SST: Single-Stream Temporal Action Proposals](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) - S. Buch et al, CVPR2017. [[code]](https://github.com/shyamal-b/sst)
* [DAPs: Deep Action Proposals for Action Understanding](https://ivul.kaust.edu.sa/Documents/Publications/2016/DAPs%20Deep%20Action%20Proposals%20for%20Action%20Understanding.pdf) - V. Escorcia et al, ECCV2016. [[code]](https://github.com/escorciav/daps) [[raw data]](https://github.com/escorciav/daps)
* [Temporal Action Localization in Untrimmed Videos via Multi-stage CNNs](http://dvmmweb.cs.columbia.edu/files/dvmm_scnn_paper.pdf) - Z. Shou et al, CVPR2016. [[code]](https://github.com/zhengshou/scnn) Note: Aka S-CNN.
* [Learning Activity Progression in LSTMs for Activity Detection and Early Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Ma_Learning_Activity_Progression_CVPR_2016_paper.pdf) - S. Ma et al, CVPR2016.
* [End-to-end Learning of Action Detection from Frame Glimpses in Videos](http://vision.stanford.edu/pdf/yeung2016cvpr.pdf) - S. Yeung et al, CVPR2016. [[code]](https://github.com/syyeung/frameglimpses) [[project web]](http://ai.stanford.edu/~syyeung/frameglimpses.html) Note: This method uses reinforcement learning
* [Bag-of-fragments: Selecting and encoding video fragments for event detection and recounting](https://staff.fnwi.uva.nl/t.e.j.mensink/publications/mettes15icmr.pdf) - P. Mettes et al, ICMR2015.


### Spatio-Temporal Action Detection
* [Spatio-Temporal Object Detection Proposals](https://hal.inria.fr/hal-01021902/PDF/proof.pdf) - D. Oneata et al, ECCV2014. [[code]](https://bitbucket.org/doneata/proposals) [[project web]](http://lear.inrialpes.fr/~oneata/3Dproposals/)
* [Action localization with tubelets from motion](http://isis-data.science.uva.nl/cgmsnoek/pub/jain-tubelets-cvpr2014.pdf) - M. Jain et al, CVPR2014.


## Licenses
License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Jinwoo Choi](https://sites.google.com/site/jchoivision/) has waived all copyright and related or neighboring rights to this work.


## Contributing
Please read the [contribution guidelines](contributing.md). Then please feel free to send me [pull requests](https://github.com/jinwchoi/Action-Recognition/pulls) or email (jinchoi@vt.edu) to add links. 
