import pybullet as p
import math
import operator 

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
                                          p.getBasePositionAndOrientation(obj_id)[0][1], 0.2),
                                          p.getQuaternionFromEuler((0,0,0)))

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
    if len(names.keys()) == 0:
        for obj in id_lookup.keys():
            id = p.addUserDebugText(obj, 
                            tuple(map(operator.add, p.getBasePositionAndOrientation(id_lookup[obj])[0], 
                            (0.2, 0.2, 0.2))))
            names[obj] = id
    # else:
    #     for name in id_lookup.keys():
    #         p.addUserDebugText(name, 
    #                             tuple(map(operator.add, p.getBasePositionAndOrientation(id_lookup[name])[0], 
    #                             (0, 0, 0.2))), 
    #                             replaceItemUniqueId=names[name])

