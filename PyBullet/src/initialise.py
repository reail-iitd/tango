import pybullet as p
import pybullet_data
import json
import os
import time

def loadObject(name, position, orientation, obj_list):
    """
    Load an object based on its specified position and orientation
    Generate constraints for the object as specified
    :param: names, positions and orientation of objects
    :return: object index
    """
    urdf = ''
    obj = 0
    object_id = 0
    for obj in obj_list:
      if obj['name'] == name:
        urdf = obj['urdf']
        break
    if orientation == []:
      object_id = p.loadURDF(urdf, position)
    else:
      object_id = p.loadURDF(urdf, position, orientation)
    return (object_id, 
            ("horizontal" in obj['constraints']), 
            obj["tolerance"], 
            obj["constraint_pos"], 
            obj["constraint_link"],
            obj["ur5_dist"])


def loadWorld(objects, object_file):
    """
    Load all objects specified in the world and create a user friendly dictionary with body
    indexes to be used by pybullet parser at the time of loading the urdf model. 
    :param objects: List containing names of objects in the world with positions and orientations.
    :return: Dictionary of object name -> object index and object index -> name
    """
    object_list = []
    horizontal = []
    object_lookup = {}
    id_lookup = {}
    cons_pos_lookup = {}
    cons_link_lookup = {}
    ur5_dist = {}
    tolerances = {}
    with open(object_file, 'r') as handle:
        object_list = json.load(handle)['objects']
    for obj in objects:
        (object_id, 
            horizontal_cons, 
            tol, 
            pos, 
            link, 
            dist) = loadObject(obj['name'], obj['position'], obj['orientation'], object_list)
        if horizontal_cons:
            horizontal.append(object_id)
        object_lookup[object_id] = obj['name']
        id_lookup[obj['name']] = object_id
        cons_pos_lookup[obj['name']] = pos
        cons_link_lookup[obj['name']] = link
        ur5_dist[obj['name']] = dist
        tolerances[obj['name']] = tol
        print(obj['name'], object_id)
    return object_lookup, id_lookup, horizontal, tolerances, cons_pos_lookup, cons_link_lookup, ur5_dist

def initWingPos(wing_file):
    wings = dict()
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "robotiq_85_left_knuckle_joint"]
    with open(wing_file, 'r') as handle:
        poses = json.load(handle)["poses"]
        for pose in poses:
            wings[pose["name"]] = dict(zip(controlJoints, pose["pose"]))
    return wings

def initHuskyUR5(world_file, object_file):
    """
    Load Husky and Ur5 module
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    with open(world_file, 'r') as handle:
        world = json.load(handle)
    (object_lookup, 
        id_lookup, 
        horizontal_list, 
        tolerances, 
        cons_pos_lookup, 
        cons_link_lookup,
        ur5_dist) = loadWorld(world['entities'], object_file)
    base = id_lookup['husky']
    arm = id_lookup['ur5']
    return base, arm, object_lookup, id_lookup, horizontal_list, tolerances, cons_pos_lookup, cons_link_lookup, ur5_dist
 