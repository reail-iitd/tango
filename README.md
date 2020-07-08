# Human-Guided Acquisition of Commonsense Knowledge for Robot Task Execution: An Imitation Learning Approach

This implementation contains all the models mentioned in the paper for next-tool prediction along with next-action prediction. This README gives a broad idea of the work that has been accomplished. For more details on replicating results, running the data collection platform and visualizing the collected dataset, refer to this [wiki](https://github.com/reail-iitd/Robot-task-planning/wiki).

<img src="/figures/screenshot.png" width="900" align="middle">

A robot working in a physical environment (like home or factory) must learn  to  make  use  of  objects  as  tools  to  perform  tasks,  for  instance,  a  tray  for carrying objects. The number of possible tools is large and it may not be feasible to demonstrate usage of each individual tool during training. Can a robot learn commonsense knowledge and adapt to novel settings where some known tools are missing, but alternative unseen tools are present? We present a recurrent neural model that predicts the likely sequence of tool interactions from the available objects for achieving an intended goal conveyed by a human.  This model is trained by user demonstrations,  which we crowd-source through humans instructing a robot in a physics simulator. This dataset maintains user plans involving multi-step object interactions (such as containment, transfer, attachment etc.) along with symbolic state changes.  The proposed model combines a graph neural network to encode the current environment state, and goal-conditioned spatial attention to predict the sequence of tool interactions for the robot to execute.  We find that providing metric and semantic properties of objects, and pre-trained object embeddings derived from a commonsense knowledge repository, significantly improves the model’s ability to generalize to predicting use of unseen tools. When compared to a graph neural network baseline, it achieves 69% improvement in prediction accuracy in novel settings and 70% improvement in predicting feasible symbolic plans in unseen settings for a simulated mobile manipulator.

## Dataset

A dataset of 1500 plans was collected from human teachers in a simulated environment containing a robot. The humans were given a goal, and were tild to provide the robot instructions in order for it to complete the task completely and efficiently. We release the platform used to collect the data. PyBullet was used for the physics simulation of the environment with a Flask back end in order to allow the collectection of data.

We also constructed a novel dataset called the **GenTest** dataset in order to test the generalization ability of our model. This dataset has also been released and can be found in the `dataset/` folder.

## Results

We use two evaluation metrics:
1. **Action Prediction Accuracy:** This is the fraction of actions predicted by the model, which matched the human demonstrated action `a` for a given state `s`.
2. **Plan Execution Accuracy:** This is the fraction of estimated plans that are successful, i.e., can be executed by the robot in simulation and attain the intended goal (with an upper bound of 50 actions in the plan). 

The results for *TANGO*, as compared to the baseline along with the individual contribution of each component, are shown below.

<img src="https://github.com/reail-iitd/Robot-task-planning/blob/master/figures/tango_table.png" width="900" align="middle">

## License

BSD-2-Clause. 
Copyright (c) 2020, Rajas Basal, Shreshth Tuli, Rohan Paul, Mausam
All rights reserved.

See License file for more details.
