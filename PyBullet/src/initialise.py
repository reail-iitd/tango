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
    return object_id, ("horizontal" in obj['constraints'])


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
    with open(object_file, 'r') as handle:
        object_list = json.load(handle)['objects']
    for obj in objects:
        object_id, horizontal_cons = loadObject(obj['name'], obj['position'], obj['orientation'], object_list)
        if horizontal_cons:
            horizontal.append(object_id)
        object_lookup[object_id] = obj['name']
        id_lookup[obj['name']] = object_id
        print(obj['name'], object_id)
    return object_lookup, id_lookup, horizontal


def initHuskyUR5(world_file, object_file):
    """
    Load Husky and Ur5 module
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    with open(world_file, 'r') as handle:
        world = json.load(handle)
    object_lookup, id_lookup, horizontal_list = loadWorld(world['entities'], object_file)
    base = id_lookup['husky']
    arm = id_lookup['ur5']
    return base, arm, object_lookup, id_lookup, horizontal_list
 