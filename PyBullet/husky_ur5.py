import os
import time
import pdb
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools
import math


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

print(pybullet_data.getDataPath())

print("Loading Plane")
p.loadURDF("plane.urdf")

print("Loading Husky")
husky = p.loadURDF("husky/husky.urdf", [0,0, 0.1],
                   [0,0,0,0.1])

print("Loading ur5 arm")  
robotID = p.loadURDF("pybullet-playground-master/urdf/sisbot.urdf", [0,0,0.320208])
for jointIndex in range(p.getNumJoints(robotID)):
  p.resetJointState(robotID, jointIndex, 0)

print("Husky num joints: ", p.getNumJoints(husky))
print("Husky Joints info:")
for i in range(p.getNumJoints(husky)):
  print(p.getJointInfo(husky, i))
print("UR5 num joints: ", p.getNumJoints(robotID))
print("UR5 Joints info:")
for i in range(p.getNumJoints(robotID)):
  print(p.getJointInfo(robotID, i))

#put kuka on top of husky

print("Reset done!")

cid = p.createConstraint(husky, -1, robotID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0., 0., -.1],
                         [0, 0, 0, 1])

# p.setGravity(0,0,-10)

#######
controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                 "elbow_joint", "wrist_1_joint",
                 "wrist_2_joint", "wrist_3_joint",
                 "robotiq_85_left_knuckle_joint"]
robotStartPos = [0,0,0]
robotStartOrn = p.getQuaternionFromEuler([0,0,0])
jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
numJoints = p.getNumJoints(robotID)
jointInfo = namedtuple("jointInfo", 
                       ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
joints = AttrDict()
for i in range(numJoints):
    info = p.getJointInfo(robotID, i)
    jointID = info[0]
    jointName = info[1].decode("utf-8")
    jointType = jointTypeList[info[2]]
    jointLowerLimit = info[8]
    jointUpperLimit = info[9]
    jointMaxForce = info[10]
    jointMaxVelocity = info[11]
    controllable = True if jointName in controlJoints else False
    info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                     jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
    if info.type=="REVOLUTE": # set revolute joint to static
        p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    joints[info.name] = info
# explicitly deal with mimic joints
def controlGripper(robotID, parent, children, mul, **kwargs):
    controlMode = kwargs.pop("controlMode")
    if controlMode==p.POSITION_CONTROL:
        pose = kwargs.pop("targetPosition")
        # move parent joint
        p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose, 
                                force=parent.maxForce, maxVelocity=parent.maxVelocity) 
        # move child joints
        for name in children:
            child = children[name]
            childPose = pose * mul[child.name]
            p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose, 
                                    force=child.maxForce, maxVelocity=child.maxVelocity) 
    else:
        raise NotImplementedError("controlGripper does not support \"{}\" control mode".format(controlMode))
    # check if there 
    if len(kwargs) is not 0:
        raise KeyError("No keys {} in controlGripper".format(", ".join(kwargs.keys())))
mimicParentName = "robotiq_85_left_knuckle_joint"
mimicChildren = {"robotiq_85_right_knuckle_joint":      1,
                 "robotiq_85_right_finger_joint":       1,
                 "robotiq_85_left_inner_knuckle_joint": 1,
                 "robotiq_85_left_finger_tip_joint":    1,
                 "robotiq_85_right_inner_knuckle_joint":1,
                 "robotiq_85_right_finger_tip_joint":   1}
parent = joints[mimicParentName] 
children = AttrDict((j, joints[j]) for j in joints if j in mimicChildren.keys())
controlRobotiqC2 = functools.partial(controlGripper, robotID, parent, children, mimicChildren)

# start simulation
try:
    flag = True
    userParams = dict()
    for name in controlJoints:
        joint = joints[name]
        userParam = p.addUserDebugParameter(name, joint.lowerLimit, joint.upperLimit, 0)
        userParams[name] = userParam

    x = p.addUserDebugParameter('X', -20, 20, 0)
    y = p.addUserDebugParameter('Y', -20, 20, 0)
    o = p.addUserDebugParameter('Omega', -1, 1, 0)
    while(flag):
        x1 = p.readUserDebugParameter(x)
        y1 = p.readUserDebugParameter(y)
        o1 = p.readUserDebugParameter(o)
        for name in controlJoints:
            joint = joints[name]
            pose = p.readUserDebugParameter(userParams[name])
            if name==mimicParentName:
                controlRobotiqC2(controlMode=p.POSITION_CONTROL, targetPosition=pose)
            else:
                p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce, 
                                        maxVelocity=joint.maxVelocity)
        q=p.getQuaternionFromEuler((0,0,o1))
        if p.getBasePositionAndOrientation(robotID)[0] != ((x1, y1, 0.220208), (q)):
            p.resetBasePositionAndOrientation(robotID, [x1, y1, 0.220208], q)
        if p.getBasePositionAndOrientation(husky)[0] != ((x1, y1, 0.0), (q)):
          p.resetBasePositionAndOrientation(husky, [x1, y1, 0], q)
        p.stepSimulation()     
    p.disconnect()
except Exception as e: 
    print(e)
    p.disconnect()

