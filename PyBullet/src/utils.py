import pybullet as p
import math
import operator 
import json
from scipy.spatial import distance

names = {}

def keepHorizontal(object_list):
    """
    Keep the objects horizontal
    """
    for obj_id in object_list:
        p.resetBasePositionAndOrientation(obj_id,
                                          p.getBasePositionAndOrientation(obj_id)[0],
                                          p.getQuaternionFromEuler((0,0,0)))

def keepOnGround(object_list):
    """
    Keep the objects on ground
    """
    for obj_id in object_list:
        p.resetBasePositionAndOrientation(obj_id,
                                          (p.getBasePositionAndOrientation(obj_id)[0][0],
                                          p.getBasePositionAndOrientation(obj_id)[0][1], 0.05),
                                          p.getBasePositionAndOrientation(obj_id)[1])

def moveKeyboard(x1, y1, o1, object_list):
    """
    Move robot based on keyboard inputs
    """
    flag = False
    keys = p.getKeyboardEvents()
    if 65297 in keys:
        x1 += math.cos(o1)*0.001
        y1 += math.sin(o1)*0.001
        flag= True
    if 65298 in keys:
        x1 -= math.cos(o1)*0.001
        y1 -= math.sin(o1)*0.001
        flag= True
    if 65295 in keys:
        o1 += 0.005
        flag= True
    if 65296 in keys:
        o1 -= 0.005
        flag= True
    q=p.getQuaternionFromEuler((0,0,o1))
    for obj_id in object_list:
        z = p.getBasePositionAndOrientation(obj_id)[0][2]
        if p.getBasePositionAndOrientation(obj_id)[0] != ((x1, y1, z), (q)):
            p.resetBasePositionAndOrientation(obj_id, [x1, y1, z], q)
    return x1, y1, o1, flag

def moveUR5Keyboard(robotID, wings, gotoWing):
    """
    Change UR5 arm position based on keyboard input
    """
    keys = p.getKeyboardEvents()
    if ord(b'h') in keys:
        gotoWing(robotID, wings["home"])
        return
    if ord(b'u') in keys:
        gotoWing(robotID, wings["up"])
        return
    if ord(b'n') in keys:
        gotoWing(robotID, wings["down"])
    return

def mentionNames(id_lookup):
    """
    Add labels of all objects in the world
    """
    if len(names.keys()) == 0:
        for obj in id_lookup.keys():
            id = p.addUserDebugText(obj, 
                            (0, 0, 0.2),
                            parentObjectUniqueId=id_lookup[obj])


def restoreOnKeyboard(world_states, x1, y1, o1):
    """
    Restore to last saved state when 'r' is pressed
    """
    keys = p.getKeyboardEvents()
    if ord(b'r') in keys:
        if len(world_states) != 0:
            id1 = world_states.pop()
            p.restoreState(stateId=id1)
        return 0, 0, 0, world_states
    return x1, y1, o1, world_states


def checkGoal(goal_file, constraints, states, id_lookup):
    """
    Check if goal conditions are true for the current state
    """
    with open(goal_file, 'r') as handle:
        file = json.load(handle)
    goals = file['goals']
    success = True

    for goal in goals:
        obj = goal['object']
        if goal['target'] != "":
            constrained = False
            for obj1 in constraints.keys():
                if obj1 == obj and constraints[obj][0] == goal["target"]:
                    constrained = True
            success = success and constrained

        if goal['state'] != "":
            positionAndOrientation = states[obj][goal['state']]
            q=p.getQuaternionFromEuler(positionAndOrientation[1])
            ((x1, y1, z1), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[obj])
            ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
            done = (True and 
                abs(x2-x1) <= 0.01 and 
                abs(y2-y1) <= 0.01 and 
                abs(a2-a1) <= 0.01 and 
                abs(b2-b2) <= 0.01 and 
                abs(c2-c1) <= 0.01 and 
                abs(d2-d2) <= 0.01)
            success = success and done

        if goal['position'] != []:
            pos = p.getBasePositionAndOrientation(id_lookup[obj])[0]
            goal_pos = goal['position']
            if abs(distance.euclidean(pos, goal_pos)) > abs(goal['tolerance']):
                success = False
    return success

def checkUR5constrained(constraints):
    """
    Check if UR5 gripper is already holding something
    """
    for obj in constraints.keys():
        if constraints[obj][0] == 'ur5':
            return True
    return False

def checkInside(constraints, states, id_lookup, obj, enclosures):
    """
    Check if object is inside cupboard or fridge
    """
    for obj in constraints.keys():
        for enclosure in enclosures:
            if constraints[obj][0] == enclosure:
                positionAndOrientation = states[enclosure]["close"]
                q=p.getQuaternionFromEuler(positionAndOrientation[1])
                ((x1, y1, z1), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[enclosure])
                ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
                closed = (abs(x2-x1) <= 0.01 and 
                        abs(y2-y1) <= 0.01 and 
                        abs(a2-a1) <= 0.01 and 
                        abs(b2-b2) <= 0.01 and 
                        abs(c2-c1) <= 0.01 and 
                        abs(d2-d2) <= 0.01)
                if closed:
                    return True
    return False

def isClosed(enclosure, states, id_lookup):
    """
    Check if enclosure is closed or not
    """
    positionAndOrientation = states[enclosure]["closed"]
    q=p.getQuaternionFromEuler(positionAndOrientation[1])
    ((x1, y1, z1), (a1, b1, c1, d1)) = p.getBasePositionAndOrientation(id_lookup[enclosure])
    ((x2, y2, z2), (a2, b2, c2, d2)) = (positionAndOrientation[0], q)
    closed = (abs(x2-x1) <= 0.01 and 
            abs(y2-y1) <= 0.01 and 
            abs(a2-a1) <= 0.01 and 
            abs(b2-b2) <= 0.01 and 
            abs(c2-c1) <= 0.01 and 
            abs(d2-d2) <= 0.01)
    return closed