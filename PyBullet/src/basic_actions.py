import pybullet as p
import math
from scipy.spatial import distance

def move(x1, y1, o1, object_list, target_coordinates, keyboard, speed, tolerance=0):
    """
    Move robot towards target coordinate location
    :params: 
        x1 - current x coordinate of objects in object_list
        y1 - current y coordinate of objects in object_list
        o1 - current angle of objects in object_list
        object_list - list of object ids to be moved
        target_coordinates - coordinates of target location
        moved - move operation complete or not
        tolerance - how close to reach target location
    :return:
        x1 - updated x coordinate of objects in object_list
        y1 - updated y coordinate of objects in object_list
        o1 - updated angle of objects in object_list
        moved - if currently moving via keyboard
    """
    if keyboard:
        return x1, y1, o1, False
    x2 = target_coordinates[0]; y2 = target_coordinates[1]
    diff = math.atan2((y2-y1),(x2-x1))%(2*math.pi) - (o1%(2*math.pi))
    if abs(diff) > 0.05:
        o1 = o1 + 0.001*speed if diff > 0 else o1 - 0.001*speed
    elif abs(distance.euclidean((x1, y1), (x2, y2))) > tolerance + 0.1: 
        x1 += math.cos(o1)*0.001*speed
        y1 += math.sin(o1)*0.001*speed
    else:
        return x1, y1, o1, True
    q=p.getQuaternionFromEuler((0,0,o1))
    for obj_id in object_list:
        z = p.getBasePositionAndOrientation(obj_id)[0][2]
        if p.getBasePositionAndOrientation(obj_id) != ((x1, y1, z), (q)):
            p.resetBasePositionAndOrientation(obj_id, [x1, y1, z], q)
    return x1, y1, o1, False


def moveTo(x1, y1, o1, object_list, target, tolerance, keyboard, speed):
    """
    Move robot towards a target object
    :params: 
        x1 - current x coordinate of objects in object_list
        y1 - current y coordinate of objects in object_list
        o1 - current angle of objects in object_list
        object_list - list of object ids to be moved
        target - object id of target to which the objects need to be moved to
        keyboard - if currently moving via keyboard
    :return:
        x1 - updated x coordinate of objects in object_list
        y1 - updated y coordinate of objects in object_list
        o1 - updated angle of objects in object_list
        moved - move operation complete or not
    """
    if keyboard:
        return x1, y1, o1, False
    y2 = p.getBasePositionAndOrientation(target)[0][1]
    x2 = p.getBasePositionAndOrientation(target)[0][0]
    target_coordinates = [x2, y2]
    return move(x1, y1, o1, object_list, target_coordinates, keyboard, speed, tolerance)


def constrain(obj1, obj2, link, pos, id_lookup, constraints, ur5_dist):
    if obj1 in constraints.keys():
        p.removeConstraint(constraints[obj1][1])
    count = 0
    for obj in constraints.keys():
        if constraints[obj][0] == obj2:
            count += 1
    print("New constraint=", obj1, " on ", obj2)
    # parent is the target, child is the object
    if obj2 == "ur5":
        cid = p.createConstraint(id_lookup[obj2], link[obj2], id_lookup[obj1], link[obj1], p.JOINT_POINT2POINT, [0, 0, 0], 
                                parentFramePosition=ur5_dist[obj1],
                                childFramePosition=pos[obj1][0],
                                childFrameOrientation=[0,0,0,0])
    else:
        cid = p.createConstraint(id_lookup[obj2], link[obj2], id_lookup[obj1], link[obj1], p.JOINT_POINT2POINT, [0, 0, 0], 
                                parentFramePosition=pos[obj2][count],
                                childFramePosition=pos[obj1][0],
                                childFrameOrientation=[0,0,0,0])
    return cid