# Awesome Action Recognition: [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of action recognition and related area (e.g. object recognition, pose estimation) resources, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

## Contents
 - [Action Recognition and Video Understanding](#action-recognition-and-video-understanding)
 - [Object Recognition](#object-recognition)
 - [Pose Estimation](#pose-estimation)

## Action Recognition and Video Understanding

### Summary posts
* [Deep Learning for Videos: A 2018 Guide to Action Recognition](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review) - Summary of major landmark action recognition research papers till 2018


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
* [Neural Graph Matching Networks for Fewshot 3D Action Recognition](http://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Neural_Graph_Matching_ECCV_2018_paper.pdf) - M. Guo et al., ECCV2018.
* [Temporal 3D ConvNets using Temporal Transition Layer](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w19/Diba_Temporal_3D_ConvNets_CVPR_2018_paper.pdf) - A. Diba et al., CVPRW2018.
* [Temporal 3D ConvNets: New Architecture and Transfer Learning for Video Classification](https://arxiv.org/abs/1711.08200) - A. Diba et al., arXiv2017.
* [Attentional Pooling for Action Recognition](https://arxiv.org/abs/1711.01467) - R. Girdhar and D. Ramanan, NIPS2017. [[code]](https://github.com/rohitgirdhar/AttentionalPoolingAction)
* [Fully Context-Aware Video Prediction](https://arxiv.org/pdf/1710.08518v1.pdf) - Byeon et al, arXiv2017.
* [Hidden Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/pdf/1704.00389.pdf) - Y. Zhu et al, arXiv2017. [[code]](https://github.com/bryanyzhu/Hidden-Two-Stream)
* [Dynamic Image Networks for Action Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/bilen16a.pdf) - H. Bilen et al, CVPR2016. [[code]](https://github.com/hbilen/dynamic-image-nets) [[project web]](http://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/)
* [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf) - J. Donahue et al, CVPR2015. [[code]](https://github.com/LisaAnne/lisa-caffe-public/tree/lstm_video_deploy) [[project web]](http://jeffdonahue.com/lrcn/)
* [Describing Videos by Exploiting Temporal Structure](http://arxiv.org/pdf/1502.08029v4.pdf) - L. Yao et al, ICCV2015. [[code]](https://github.com/yaoli/arctic-capgen-vid) note: from the same group of RCN paper “Delving Deeper into Convolutional Networks for Learning Video Representations"
* [Two-Stream SR-CNNs for Action Recognition in Videos](http://wanglimin.github.io/papers/ZhangWWQW_CVPR16.pdf) - L. Wang et al, BMVC2016.
* [Real-time Action Recognition with Enhanced Motion Vector CNNs](http://arxiv.org/abs/1604.07669) - B. Zhang et al, CVPR2016. [[code]](https://github.com/zbwglory/MV-release)
* [Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf) - L. Wang et al, CVPR2015. [[code]](https://github.com/wanglimin/TDD)


### Temporal Action Detection
* [Rethinking the Faster R-CNN Architecture for Temporal Action Localization](https://arxiv.org/pdf/1804.07667v1.pdf) - Yu-Wei Chao et al., CVPR2018
* [Weakly Supervised Action Localization by Sparse Temporal Pooling Network](https://arxiv.org/pdf/1712.05080) - Phuc Nguyen et al., CVPR 2018
* [Temporal Deformable Residual Networks for Action Segmentation in Videos](http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr18_TDRN.pdf) - P. Lei and S. Todrovic., CVPR2018.
* [End-to-End, Single-Stream Temporal Action Detection in Untrimmed Videos](http://vision.stanford.edu/pdf/buch2017bmvc.pdf) - Shayamal Buch et al., BMVC 2017 [[code]](https://github.com/shyamal-b/ss-tad)
* [Cascaded Boundary Regression for Temporal Action Detection](https://arxiv.org/abs/1705.01180) - Jiyang Gao et al., BMVC 2017 [[code](https://github.com/jiyanggao/CBR)]
* [Temporal Tessellation: A Unified Approach for Video Analysis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Kaufman_Temporal_Tessellation_A_ICCV_2017_paper.pdf) - Kaufman et al., ICCV2017. [[code]](https://github.com/dot27/temporal-tessellation) 
* [Temporal Action Detection with Structured Segment Networks](http://cn.arxiv.org/pdf/1704.06228v2) - Y. Zhao et al., ICCV2017. [[code]](https://github.com/yjxiong/action-detection) [[project web]](http://yjxiong.me/others/ssn/)
* [Temporal Context Network for Activity Localization in Videos](https://arxiv.org/pdf/1708.02349.pdf) - X. Dai et al., ICCV2017.
* [Detecting the Moment of Completion: Temporal Models for Localising Action Completion](https://arxiv.org/abs/1710.02310) - F. Heidarivincheh et al., arXiv2017.
* [CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos](https://arxiv.org/abs/1703.01515/) - Z. Shou et al, CVPR2017. [[code]](https://bitbucket.org/columbiadvmm/cdc)
* [SST: Single-Stream Temporal Action Proposals](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) - S. Buch et al, CVPR2017. [[code]](https://github.com/shyamal-b/sst)
* [R-C3D: Region Convolutional 3D Network for Temporal Activity Detection](https://arxiv.org/abs/1703.07814) - H. Xu et al, arXiv2017. [[code]](https://github.com/VisionLearningGroup/R-C3D) [[project web]](http://ai.bu.edu/r-c3d/) [[PyTorch]](https://github.com/sunnyxiaohu/R-C3D.pytorch)
* [DAPs: Deep Action Proposals for Action Understanding](https://ivul.kaust.edu.sa/Documents/Publications/2016/DAPs%20Deep%20Action%20Proposals%20for%20Action%20Understanding.pdf) - V. Escorcia et al, ECCV2016. [[code]](https://github.com/escorciav/daps) [[raw data]](https://github.com/escorciav/daps)
* [Online Action Detection using Joint Classification-Regression Recurrent Neural Networks](https://arxiv.org/abs/1604.05633) - Y. Li et al, ECCV2016. Noe: RGB-D Action Detection
* [Temporal Action Localization in Untrimmed Videos via Multi-stage CNNs](http://dvmmweb.cs.columbia.edu/files/dvmm_scnn_paper.pdf) - Z. Shou et al, CVPR2016. [[code]](https://github.com/zhengshou/scnn) Note: Aka S-CNN.
* [Fast Temporal Activity Proposals for Efficient Detection of Human Actions in Untrimmed Videos](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Heilbron_Fast_Temporal_Activity_CVPR_2016_paper.pdf) - F. Heilbron et al, CVPR2016. [[code]](https://github.com/cabaf/sparseprop) Note: Depends on [C3D](http://vlg.cs.dartmouth.edu/c3d/), aka SparseProp.
* [Actionness Estimation Using Hybrid Fully Convolutional Networks](https://arxiv.org/abs/1604.07279) - L. Wang et al, CVPR2016. [[code]](https://github.com/wanglimin/actionness-estimation/) Note: The code is not a complete verision. It only contains a demo, not training. [[project web]](http://wanglimin.github.io/actionness_hfcn/index.html)
* [Learning Activity Progression in LSTMs for Activity Detection and Early Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Ma_Learning_Activity_Progression_CVPR_2016_paper.pdf) - S. Ma et al, CVPR2016.
* [End-to-end Learning of Action Detection from Frame Glimpses in Videos](http://vision.stanford.edu/pdf/yeung2016cvpr.pdf) - S. Yeung et al, CVPR2016. [[code]](https://github.com/syyeung/frameglimpses) [[project web]](http://ai.stanford.edu/~syyeung/frameglimpses.html) Note: This method uses reinforcement learning
* [Fast Action Proposals for Human Action Detection and Search](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yu_Fast_Action_Proposals_2015_CVPR_paper.pdf) - G. Yu and J. Yuan, CVPR2015. Note: code for FAP is NOT available online. Note: Aka FAP.
* [Bag-of-fragments: Selecting and encoding video fragments for event detection and recounting](https://staff.fnwi.uva.nl/t.e.j.mensink/publications/mettes15icmr.pdf) - P. Mettes et al, ICMR2015.
* [Action localization in videos through context walk](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Soomro_Action_Localization_in_ICCV_2015_paper.pdf) - K. Soomro et al, ICCV2015.

### Spatio-Temporal Action Detection
* [A Better Baseline for AVA](https://arxiv.org/pdf/1807.10066.pdf) - R. Girdhar et al., ActivityNet Workshop, CVPR2018. 
* [Real-Time End-to-End Action Detection with Two-Stream Networks](https://arxiv.org/abs/1802.08362) - A. El-Nouby and G. Taylor, arXiv2018.
* [Human Action Localization with Sparse Spatial Supervision](https://arxiv.org/pdf/1605.05197.pdf) - P. Weinzaepfel et al., arXiv2017. 
* [Unsupervised Action Discovery and Localization in Videos](http://openaccess.thecvf.com/content_ICCV_2017/papers/Soomro_Unsupervised_Action_Discovery_ICCV_2017_paper.pdf) - K. Soomro and M. Shah, ICCV2017.
* [Spatial-Aware Object Embeddings for Zero-Shot Localization and Classification of Actions](https://arxiv.org/pdf/1707.09145.pdf) - P. Mettes and C. G. M. Snoek, ICCV2017.
* [Action Tubelet Detector for Spatio-Temporal Action Localization](https://arxiv.org/abs/1705.01861) - V. Kalogeiton et al, ICCV2017. [[code]](https://github.com/vkalogeiton/caffe/tree/act-detector) [[project web]](http://thoth.inrialpes.fr/src/ACTdetector/)
* [Tube Convolutional Neural Network (T-CNN) for Action Detection in Videos](https://arxiv.org/pdf/1703.10664.pdf) - [R. Hou](http://www.cs.ucf.edu/~rhou/) et al, ICCV2017. [[project web]](http://crcv.ucf.edu/projects/TCNN/)
* [Chained Multi-stream Networks Exploiting Pose, Motion, and Appearance for Action Classification and Detection](https://arxiv.org/abs/1704.00616) - M. Zolfaghari et al, ICCV2017. [[project web]](https://lmb.informatik.uni-freiburg.de/projects/action_chain/)
* [TORNADO: A Spatio-Temporal Convolutional Regression Network for Video Action Proposal](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_TORNADO_A_Spatio-Temporal_ICCV_2017_paper.pdf) - H. Zhu et al., ICCV2017. 
* [Online Real time Multiple Spatiotemporal Action Localisation and Prediction](https://arxiv.org/pdf/1611.08563v3.pdf) - [G. Singh](http://gurkirt.github.io/) et al, ICCV2017. [[code]](https://github.com/gurkirt/realtime-action-detection)
* [AMTnet: Action-Micro-Tube regression by end-to-end trainable deep architecture](https://arxiv.org/pdf/1704.04952.pdf) - S. Saha et al, ICCV2017.
* [Am I Done? Predicting Action Progress in Videos](https://arxiv.org/abs/1705.01781) - F. Becattini et al, BMVC2017.
* [Generic Tubelet Proposals for Action Localization](https://arxiv.org/abs/1705.10861) - J. He et al, arXiv2017.
* [Incremental Tube Construction for Human Action Detection](https://arxiv.org/pdf/1704.01358.pdf) - H. S. Behl et al, arXiv2017.
* [Multi-region two-stream R-CNN for action detection](https://www.robots.ox.ac.uk/~vgg/rg/papers/peng16eccv.pdf) - [X. Peng](http://xjpeng.weebly.com/) and C. Schmid. ECCV2016. [[code]](https://github.com/pengxj/action-faster-rcnn)
* [Spot On: Action Localization from Pointly-Supervised Proposals](http://jvgemert.github.io/pub/spotOnECCV16.pdf) - P. Mettes et al, ECCV2016.
* [Deep Learning for Detecting Multiple Space-Time Action Tubes in Videos](https://arxiv.org/abs/1608.01529) - S. Saha et al, BMVC2016. [[code]](https://bitbucket.org/sahasuman/bmvc2016_code) [[project web]](http://sahasuman.bitbucket.org/bmvc2016/)
* [Learning to track for spatio-temporal action localization](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Weinzaepfel_Learning_to_Track_ICCV_2015_paper.pdf) - P. Weinzaepfel et al. ICCV2015.
* [Action detection by implicit intentional motion clustering](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Chen_Action_Detection_by_ICCV_2015_paper.pdf) - W. Chen and J. Corso, ICCV2015.
* [Finding Action Tubes](https://people.eecs.berkeley.edu/~gkioxari/ActionTubes/action_tubes.pdf) - G. Gkioxari and J. Malik CVPR2015. [[code]](https://github.com/gkioxari/ActionTubes) [[project web]](https://people.eecs.berkeley.edu/~gkioxari/ActionTubes/)
* [APT: Action localization proposals from dense trajectories](http://jvgemert.github.io/pub/gemertBMVC15APTactionProposals.pdf) - J. Gemert et al, BMVC2015. [[code]](https://github.com/jvgemert/apt)
* [Spatio-Temporal Object Detection Proposals](https://hal.inria.fr/hal-01021902/PDF/proof.pdf) - D. Oneata et al, ECCV2014. [[code]](https://bitbucket.org/doneata/proposals) [[project web]](http://lear.inrialpes.fr/~oneata/3Dproposals/)
* [Action localization with tubelets from motion](http://isis-data.science.uva.nl/cgmsnoek/pub/jain-tubelets-cvpr2014.pdf) - M. Jain et al, CVPR2014.
* [Spatiotemporal deformable part models for action detection](http://crcv.ucf.edu/papers/cvpr2013/cvpr2013-sdpm.pdf) - [Y. Tian](http://www.cs.ucf.edu/~ytian/index.html) et al, CVPR2013. [[code]](http://www.cs.ucf.edu/~ytian/sdpm.html)
* [Action localization in videos through context walk](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Soomro_Action_Localization_in_ICCV_2015_paper.pdf) - K. Soomro et al, ICCV2015.
* [Fast Action Proposals for Human Action Detection and Search](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yu_Fast_Action_Proposals_2015_CVPR_paper.pdf) - G. Yu and J. Yuan, CVPR2015. Note: code for FAP is NOT available online. Note: Aka FAP.

### Ego-Centric Action Recognition
* [Actor and Observer: Joint Modeling of First and Third-Person Videos](https://arxiv.org/pdf/1804.09627.pdf) - G. Sigurdsson et al., CVPR2018. [[code]](https://github.com/gsig/actor-observer)

### Miscellaneous
* [What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/pdf/1904.04346.pdf) - P. Parma and B. T. Morris. CVPR2019.
* [PathTrack: Fast Trajectory Annotation with Path Supervision](http://openaccess.thecvf.com/content_ICCV_2017/papers/Manen_PathTrack_Fast_Trajectory_ICCV_2017_paper.pdf) - S. Manen et al., ICCV2017.
* [CortexNet: a Generic Network Family for Robust Visual Temporal Representations](https://arxiv.org/pdf/1706.02735.pdf) A. Canziani and E. Culurciello - arXiv2017. [[code]](https://github.com/atcold/pytorch-CortexNet) [[project web]](https://engineering.purdue.edu/elab/CortexNet/)
* [Slicing Convolutional Neural Network for Crowd Video Understanding](http://www.ee.cuhk.edu.hk/~jshao/papers_jshao/jshao_cvpr16_scnn.pdf) - J. Shao et al, CVPR2016. [[code]](https://github.com/amandajshao/Slicing-CNN)
* [Two-Stream (RGB and Flow) pretrained model weights](https://github.com/craftGBD/caffe-GBD/tree/master/models/action_recognition)


### Video Annotation
* [Efficiently scaling up crowdsourced video annotation](http://cvrr.ucsd.edu/ece285/Spring2014/papers/Vondrick_IJCV2013.pdf) - C. Vondrick et. al, IJCV2013. [[code]](https://github.com/cvondrick/vatic)
* [The Design and Implementation of ViPER](https://www.cs.umd.edu/grad/scholarlypapers/papers/davidm-viper.pdf) - D. Mihalcik and D. Doermann, Technical report.
* [VTT: Visual Object Tagging Tool](https://github.com/microsoft/VoTT). Modern app to annotate objects in videos and images. It facilitates the development of an end-to-end machine learning pipeline encompassing the annotation/export/import of assets. Moreover, it could run as a native app or via web.
* [VIA: VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/). Simple and standalone manual annotation web-app for image, audio and video. It runs in the web browser and does not require any installation or setup.



## Licenses
License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Jinwoo Choi](https://sites.google.com/site/jchoivision/) has waived all copyright and related or neighboring rights to this work.


## Contributing
Please read the [contribution guidelines](contributing.md). Then please feel free to send me [pull requests](https://github.com/jinwchoi/Action-Recognition/pulls) or email (jinchoi@vt.edu) to add links. 
