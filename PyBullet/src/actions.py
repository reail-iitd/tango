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
                ["move", args[1]]
            ])

        elif high_level_action['name'] == 'pushTo':
            action_list.extend([
                ["moveTo", args[0]],
                ["changeWing", "up"],
                ["constrain", args[0], "ur5"],
                ["moveTo", args[1]]
            ])


        elif high_level_action['name'] == 'moveTo':
            action_list.extend([
                ["moveTo", args[0]]
            ])

        elif high_level_action['name'] == 'move':
            action_list.extend([
                ["move", args[0]]
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

        elif high_level_action['name'] == 'drop':
            for obj in args[0]:
                action_list.append(["constrain", obj, args[1]])
    return action_list