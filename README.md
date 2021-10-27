# TANGO: Commonsense Generalization in Predicting Tool Interactions for Mobile Manipulators

This repository contains code implementation of the paper "TANGO: Commonsense Generalization in Predicting Tool Interactions for Mobile Manipulators".

**Shreshth Tuli, Rajas Bansal, Rohan Paul and Mausam**. Department of Computer Science and Engineering, Indian Institute of Techonology Delhi.

To appear in **International Joint Conference on Artificial Intelligence (IJCAI) 2021**.

## Abstract

A robot working in a physical environment (like home or factory) must learn  to  make  use  of  objects  as  tools  to  perform  tasks,  for  instance,  a  tray  for carrying objects. The number of possible tools is large and it may not be feasible to demonstrate usage of each individual tool during training. Can a robot learn commonsense knowledge and adapt to novel settings where some known tools are missing, but alternative unseen tools are present? We present a recurrent neural model that predicts the likely sequence of tool interactions from the available objects for achieving an intended goal conveyed by a human.  This model is trained by user demonstrations,  which we crowd-source through humans instructing a robot in a physics simulator. This dataset maintains user plans involving multi-step object interactions (such as containment, transfer, attachment etc.) along with symbolic state changes.  The proposed model combines a graph neural network to encode the current environment state, and goal-conditioned spatial attention to predict the sequence of tool interactions for the robot to execute.  We find that providing metric and semantic properties of objects, and pre-trained object embeddings derived from a commonsense knowledge repository, significantly improves the model’s ability to generalize to predicting use of unseen tools. When compared to a graph neural network baseline, it achieves 69% improvement in prediction accuracy in novel settings and 70% improvement in predicting feasible symbolic plans in unseen settings for a simulated mobile manipulator.

## Supplementary video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lUWU3rK1Gno/0.jpg)](https://www.youtube.com/watch?v=lUWU3rK1Gno)

## Getting Started

This implementation contains all the models mentioned in the paper for next-tool prediction along with next-action prediction. This readme gives a broad idea of the work that has been accomplished. The code start point is `app.py`. For more details on replicating results, running the data collection platform and visualizing the collected dataset, refer to this [wiki](https://github.com/reail-iitd/tango/wiki).

For our ToolTango model, use $MODEL_NAME as **GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Tool_Action**.

## Arxiv preprint
https://arxiv.org/abs/2105.04556.

## Cite this work
```
@article{tuli2021tango,
  title={TANGO: Commonsense Generalization in Predicting Tool Interactions for Mobile Manipulators},
  author={Tuli, Shreshth and Bansal, Rajas and Paul, Rohan and Mausam},
  journal={arXiv preprint arXiv:2105.04556},
  year={2021}
}
```

## License

BSD-2-Clause. 
Copyright (c) 2021, Shreshth Tuli, Rajas Basal, Rohan Paul, Mausam
All rights reserved.

See License file for more details.
