import pybullet as p
import math

def keepHorizontal(object_list):
    """
    Keep the objetcts horizontal
    """
    for obj_id in object_list:
        p.resetBasePositionAndOrientation(obj_id,
                                          p.getBasePositionAndOrientation(obj_id)[0],
                                          p.getQuaternionFromEuler((0,0,0)))

def moveKeyboard(x1, y1, o1, object_list):
    """
    Move robot based on keyboard inputs
    """
    keys = p.getKeyboardEvents()
    if 65297 in keys:
        x1 += math.cos(o1)*0.001
        y1 += math.sin(o1)*0.001
    if 65298 in keys:
        x1 -= math.cos(o1)*0.001
        y1 -= math.sin(o1)*0.001
    if 65295 in keys:
        o1 += 0.005
    if 65296 in keys:
        o1 -= 0.005
    q=p.getQuaternionFromEuler((0,0,o1))
    for obj_id in object_list:
        z = p.getBasePositionAndOrientation(obj_id)[0][2]
        if p.getBasePositionAndOrientation(obj_id)[0] != ((x1, y1, z), (q)):
            p.resetBasePositionAndOrientation(obj_id, [x1, y1, z], q)
    return x1, y1, o1

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

