# Human-Guided Acquisition of Commonsense Knowledge for Robot Task Execution: An Imitation Learning Approach

This implementation contains all the models mentioned in the paper for next-tool prediction along with next-action prediction.

## Contents 
- Introduction
- Environment Setup
- Directory Structure
- Setup Web Interface for Data Collection
- Dataset
- Training models

## Introduction

<img src="/screenshot.png" width="900" align="middle">

In order to effectively perform activities in realistic environments like a home or a factory, robots often require tools. Determining if a tool could be effective in accomplishing a task can be helpful in scoping the search for feasible plans. This work aims at learning such "commonsense" to generalize and adapt to novel settings where one or more known tools are missing with the presence of unseen tools and world scenes. Towards this objective, we crowd-source a data set of humans instructing a robot in a Physics simulator perform tasks involving multi-step object interactions with symbolic state changes. The data set is used to supervise a learner that predicts the use of an object as a tool in a plan achieving the agent’s goal. The model encodes the agent’s environment, including metric and semantic properties, using gated graph convolutions and incorporates goal- conditioned spatial attention to predict the optimal tool to use. We demonstrate generalization for predicting tool use for objects unseen in training data by conditioning the model on pre-trained embeddings derived from a relational knowledge source such as ConceptNet. Experiments show that the learned model can accurately predict common use of tools based on spatial context, semantic attribute of objects and goals specifications and effectively generalize to novel scenarios with world instances not encountered in training.

## Environment Setup

```bash
$ python3 -m venv commonsense_tool
$ source commonsense_tool/bin/activate
(commonsense_tool) $ git clone https://github.com/reail-iitd/commonsense-task-planning.git
(commonsense_tool) $ cd Robot-task-planning
(commonsense_tool) $ git checkout release
(commonsense_tool) $ pip3 install -r requirements.txt
```

We used a system with 32gb RAM CPU and 8GB graphics RAM to run our simulator. The models were trained using the same system. The simulator and training pipeline has been tested on Windows, MacOS and Ubuntu 16.04.

## Directory Structure
| Folder/File                       | Utility 		              
| --------------------------------- | --------------------------- 
| **app.py** 				        | This is the main file to run the data collection platform. This file starts the web server at the default port 5000 on localhost. It starts an instance of PyBullet on the system and exposes the simulator to the user at the appropriate website.
| **train.py**                      | This is the main file used to train and evaluate all the models as mentioned in the paper.
| **husky_ur5.py**                  | This is the main file for the PyBullet simulator. It is responsible for loading the PyBullet simulator, running the appropriate action and sending the appropriate exceptions wherever applicable. It also checks if the goal specified has been completed.
| **src/GNN/CONSTANTS.py**			| This file contains the global constant used by the training pipeline like the number of epochs, hidden size used etc.
| **src/GNN/dataset_utils.py**      | This file contains the Dataset class, which can be used to process the dataset in any form as required by the training pipeline.
| **src/GNN/\*models** 	            | These contain the different PyTorch models that were used for training the system.
| **src/datapoint.py** 				| This contains the datapoint class. All datapoints found in the dataset are an instance of this class.
| **jsons/embeddings**              | These contain the files corresponding to [fasttext](https://fasttext.cc/docs/en/english-vectors.html) and [conceptnet](https://github.com/commonsense/conceptnet-numberbatch) embeddings.
| **jsons/\*\_goals**               | These contain the goals which can be completed by the robot in the factory and the home domain.
| **jsons/\*\_worlds**      	    | These contain the different world instances in the home and factory domain.
| **jsons/\*.json**					| These are configuration files for the simulator and the webapp. These define the different actions possible in the simulator, the different objects present in the simulator, states possible, readable form of actions to show on the webapp etc.
| **models/\***                     | This folder contains the stl and urdf files for all models which are loaded by the physics simulator used i.e Pybullet.
| **templates/\***				    | These are the templates that are used by the webapp to load the different tutorial webpages along with the actual data collection platform.
| **dataste/\***                    | These are files corresponding to training, test and generalization datasets.

## Setup Web Interface for Data Collection
To execute the website that is needed for data collection, use the following command:
```bash
$ python3 app.py --randomize
```
The website can now be accessed by using the link (http://0.0.0.0:5000/).
For all the arguments that can be provided look at the help provided,

```bash
$ python3 app.py --help
```

## Dataset

Here is how the dataset structure should look like:

```
dataset
├── home
├── factory
└── test
    ├── home
    └── factory
```

The dataset is organized as follows. We have 8 different goals and 10 different world instances for both the domains, home and factory. Each domain has 8 directories corresponding to the goals possible for the domain. These goals itself, contain directories for the 10 different world instances. Each goal for each world instance in a particular domain thus has a number of different human demonstrations, and these are saved in the form of a .datapoint file for each plan. This is a pickled instance of the Datapoint class found in `src/datapoint.py` and contains all the information needed about the plan. Refer to the [class](src/datapoint.py) for more information.

## Training

All the trained models of ToolNet can be found [here](https://drive.google.com/open?id=1Kw65B55DehnteO1hwLUk0k1rWw2eCTl0).
All the trained models of Tango can be found [here](https://drive.google.com/file/d/1nhhvoJqhktsoDnvgI0vO0kQizs-yV1tO/view?usp=sharing).

All the models mentioned in the paper can be trained through the command

```bash
$ python3 train.py $DOMAIN $TRAINING_TYPE $MODEL_NAME $EXEC_TYPE
```
Here DOMAIN can be home/factory.

TRAINING_TYPE can be as follows:

| TRAINING_TYPE                     | Meaning                     
| --------------------------------- | --------------------------- 
| **gcn**                           | Tool prediction model predicting most probable tool using inital state  
| **gcn_seq**                       | Tool sequence prediction model, which predicts the sequence of tools that will be used in the plan      
| **action**                        | Action prediction model which does not use the trained tool prediction model

MODEL_NAME specifies the specific PyTorch model that you want to train. Look at `src/GNN/models.py` (ToolNet) or `src/GNN/action_models.py` (Tango) to specify the name. They are specified here for reference.


| MODEL_NAME                     | Name in papern (Tango)                     
| -------------------------------| --------------------------- 
| **GGCN_Auto_Action**               | GGCN+Auto (Baseline)
| **GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Action**                    | Tango
| **Final_GGCN_Action**              | - GGCN
| **Final_Attn_Action**              | - Goal-Conditioned Attn
| **Final_Cons_Action**              | - Constraints
| **Final_Auto_Action**              | - Autoregression
| **Final_Aseq_Action**              | - Temporal Action History
| **Final_L_Action**                 | - Factored Likelihood

EXEC_TYPE can be as follows:

| EXEC_TYPE                         | Meaning                     
| --------------------------------- | --------------------------- 
| **train**                         | Train the model 
| **accuracy**                      | Determine the prediction accuracy for tool/action prediction model on the given dataset     
| **generalization**                | Calculate accuracies of all models on generalization test set
| **policy**                        | Run the action model for the given dataset and determine percentage task completion using the model as a policy in approximate simulated environment.

## Sample Commands

To train the best model in **home** domain:
```
python3 train.py home action GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Action train
```

To train the best model in **factory** domain:
```
python3 train.py factory action GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Action train
```

To train the ablated **- GGCN** model in **home** domain:
```
python3 train.py home action Final_GGCN_Action train
```

To find the action prediction accuracy of trained **- GGCN** model:
```
python3 train.py home action Final_GGCN_Action accuracy
```

To find the plan execution accuracy on Test Set of trained **- GGCN** model:
```
python3 train.py home action Final_GGCN_Action policy
```

To find the plan execution accuracy on GenTest Set of trained **- GGCN** model:
```
python3 train.py home action Final_GGCN_Action generalization
```

