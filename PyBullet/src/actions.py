import json

def convertActions(action_file):
    inp = None
    with open(action_file, 'r') as handle:
        inp = json.load(handle)
    action_list = []

    for high_level_action in inp['actions']:
        args = high_level_action['args']
        print(args)
        if high_level_action['name'] == 'pickNplaceAonB':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveTo", args[1]],
                ["constrain", args[0], args[1]]
            ])

        if high_level_action['name'] == 'changeWing':
            action_list.extend([
                ["changeWing", args[0]]
            ])

        elif high_level_action['name'] == 'moveAToB':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveTo", args[1]]
            ])

        elif high_level_action['name'] == 'push':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["move", args[1]],
                ["removeConstraint", args[0], "ur5"]
            ])

        elif high_level_action['name'] == 'pushTo':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveTo", args[1]],
                ["removeConstraint", args[0], "ur5"]
            ])


        elif high_level_action['name'] == 'moveTo':
            action_list.extend([
                ["moveTo", args[0]]
            ])

        elif high_level_action['name'] == 'move':
            action_list.extend([
                ["move", args[0]]
            ])

        elif high_level_action['name'] == 'moveUp':
            action_list.extend([
                ["move", [0.5, -0.5, 0]],
                ["moveZ", [-1.5, 1.5, 1]]
            ])
        
        elif high_level_action['name'] == 'moveDown':
            action_list.extend([
                ["move", [-0.6, 0.6, 1]],
                ["moveZ", [0.5, -0.5, 0]]
            ])

        elif high_level_action['name'] == 'pick':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"]
            ])

        elif high_level_action['name'] == 'place':
            action_list.extend([
                ["moveTo", args[1]],
                ["changeWing", "up"],
                ["constrain", args[0], args[1]]
            ])

        elif high_level_action['name'] == 'dropTo':
            for obj in args[0]:
                action_list.append(["constrain", obj, args[1]])
        
        elif high_level_action['name'] == 'drop':
            action_list.append(["removeConstraint", args[0], "ur5"])

        elif high_level_action['name'] == 'changeState':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["changeState", args[0], args[1]]
            ])
        
        elif high_level_action['name'] == 'placeRamp':
            action_list.extend([
                ["moveTo", "ramp"],
                ["changeWing", "up"],
                ["constrain", "ramp", "ur5"],
                ["move", [0.5,-0.5,0]],
                ["constrain", "ramp", "floor_warehouse"]
            ])

        action_list.append(["saveBulletState"])

    return action_list