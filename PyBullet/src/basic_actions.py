import pybullet as p
import math
from scipy.spatial import distance

def moveTo(x1, y1, o1, object_list, target, tolerance, moved):
    """
    Move robot towards a target object
    """
    if moved:
        return x1, y1, o1, False
    y2 = p.getBasePositionAndOrientation(target)[0][1]
    x2 = p.getBasePositionAndOrientation(target)[0][0]
    diff = math.atan2((y2-y1),(x2-x1))%(2*math.pi) - (o1%(2*math.pi))
    # print(math.atan2((y2-y1),(x2-x1))%(2*math.pi), (o1%(2*math.pi)), diff)
    if abs(diff) > 0.05:
        o1 = o1 + 0.001 if diff > 0 else o1 - 0.001
    elif abs(distance.euclidean((x1, y1), (x2, y2)) - tolerance) > 0.2: 
        x1 += math.cos(o1)*0.001
        y1 += math.sin(o1)*0.001
    else:
        return x1, y1, o1, True
    q=p.getQuaternionFromEuler((0,0,o1))
    for obj_id in object_list:
        z = p.getBasePositionAndOrientation(obj_id)[0][2]
        if p.getBasePositionAndOrientation(obj_id) != ((x1, y1, z), (q)):
            p.resetBasePositionAndOrientation(obj_id, [x1, y1, z], q)
    return x1, y1, o1, False

def constrain(obj1, obj2, link, pos, id_lookup, constraints):
    if obj1 in constraints.keys():
        p.removeConstraint(constraints[obj1][1])
    count = 0
    for obj in constraints.keys():
        if constraints[obj][0] == obj2:
            count += 1
    print(obj1, obj2, link[obj1], link[obj2], pos[obj1], pos[obj2], count)
    cid = p.createConstraint(id_lookup[obj2], link[obj2], id_lookup[obj1], link[obj1], p.JOINT_POINT2POINT, [0, 0, 0], 
                            parentFramePosition=pos[obj2][count],
                            childFramePosition=pos[obj1][0],
                            childFrameOrientation=[0,0,0,0])
    print(cid)
    return cid