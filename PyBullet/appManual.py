#!/usr/bin/env python3
from importlib import import_module
import os
import json
from flask import Flask, render_template, Response, request
from camera import Camera
from base_camera import BaseCamera
import multiprocessing as mp
import time
from src.parser import *
from os import listdir

args = initParser()

queue_from_webapp_to_simulator = mp.Queue()
queue_from_simulator_to_webapp = mp.Queue()
queue_for_error = mp.Queue()
workerId = None
app = Flask(__name__)
moves_to_show = []

dict_of_predicates = {
		# "Move object to destination":{"source-object" : "dropdown-objects", "destination (near object)": "dropdown-objects"},
		"Push object to destination": {"Object to push" : "dropdown-objects", "Destination (near this object)": "dropdown-objects"},
		"Pick source and place on destination": {"Object to pick": "dropdown-objects", "Object to place on": "dropdown-objects"},
        "Move robot to object" : {"Destination (near this object)": "dropdown-objects"},
        "Open/Close object" : {"Object to open or close": "dropdown-objects", "Open or Close it": "dropdown-states"},
        "Pick Object": {"Object to pick": "dropdown-objects"},
        "Drop Object on destination": {"Object to drop": "dropdown-objects", "Object to drop on": "dropdown-objects"},
        "Climb up an object": {"Object to climb on": "dropdown-objects"},
        "Climb down an object": {"Object to climb down from": "dropdown-objects"},
        "Apply object on another object": {"Object to apply": "dropdown-objects", "Object to apply on": "dropdown-objects"},
        "Stick object to destination": {"Object to stick": "dropdown-objects", "Destination object to stick on": "dropdown-objects"},
        "Clean object": {"Object to clean": "dropdown-objects"},
        "Switch Object on/off": {"Object to switch state": "dropdown-objects", "On or off": "dropdown-states"}
    }

dict_predicate_to_action = {
    # "Move object to destination": "moveAToB",
    "Push object to destination": "pushTo",
    "Pick source and place on destination": "pickNplaceAonB",
    "Move robot to object": "moveTo",
    "Open/Close object" : "changeState",
    "Pick Object": "pick",
    "Drop Object on destination": "dropTo",
    "Climb up an object": "climbUp",
    "Climb down an object": "climbDown",
    "Apply object on another object": "apply",
    "Stick object to destination": "stick",
    "Clean object": "clean",
    "Switch Object on/off": "changeState"
}

# Unnecessary (can be removed)
d = json.load(open(args.world))["entities"]
world_objects = []
renamed_objects = {}
constraints_dict = json.load(open("jsons/constraints.json"))
dropdown_states = ["open", "close", "off"]
for obj in d:
    if (("ignore" in obj) and (obj["ignore"] == "true")):
        continue
    if ("rename" in obj):
        world_objects.append(obj["rename"])
        renamed_objects[obj["rename"]] = obj["name"]
    else:
        world_objects.append(obj["name"])
world_objects.sort()

def convertActionsFromFile(action_file):
    inp = None
    with open(action_file, 'r') as handle:
        inp = json.load(handle)
    return(inp)

def simulator(queue_from_webapp_to_simulator, queue_from_simulator_to_webapp, queue_for_error):
    import husky_ur5
    import src.actions
    import sys
    queue_from_simulator_to_webapp.put(True)
    print ("Waiting")
    husky_ur5.firstImage()
    goal_file = None
    while True:
        inp = queue_from_webapp_to_simulator.get()
        if ("rotate" in inp or "zoom" in inp or "toggle" in inp):
            husky_ur5.changeView(inp["rotate"])
        elif "undo" in inp:
            husky_ur5.undo()
            if (len(moves_to_show) > 0):
                moves_to_show.pop(-1)
        elif "showObject" in inp:
            husky_ur5.showObject(inp["showObject"])
        elif "restart" in inp:
            goal_file = inp["restart"]
            print ("hello")
            husky_ur5.destroy()
            del sys.modules["husky_ur5"]
            del sys.modules["src.actions"]
            import husky_ur5
            import src.actions
            husky_ur5.firstImage()
            queue_from_simulator_to_webapp.put(True)
        else:
            try:
                done = husky_ur5.execute(inp, goal_file)
                print("Done: ", done)
            except Exception as e:
                print (str(e))
                queue_for_error.put(str(e))
                done = False
            if (done):
                foldername = 'dataset/home/' + goal_file.split("\\")[3].split(".")[0] + '/' + args.world.split('\\')[3].split(".")[0]
                try:   
                    a = len(listdir(foldername))
                except Exception as e:
                    os.makedirs(foldername)
                if len(listdir(foldername)) == 0:
                    husky_ur5.saveDatapoint(foldername + '/' + '0')
                else:    
                    husky_ur5.saveDatapoint(foldername + '/' + str(int(listdir(foldername)[-1].split('.')[0]) + 1))
                queue_for_error.put("You have completed this tutorial.")
                queue_from_webapp_to_simulator.put({"restart": args.goal})
            called_undo_before = False

@app.route('/', methods = ["GET"])
def index():
    queue_from_webapp_to_simulator.put({"restart": args.goal})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    if (request.method == "GET"):
        return render_template('index.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/tutorial/1', methods = ["GET"])
def show_tutorial1():
    return render_template('tutorial1.html')

@app.route('/tutorial/2', methods = ["GET"])
def show_tutorial2():
    queue_from_webapp_to_simulator.put({"restart": None})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial2.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/tutorial/3', methods = ["GET"])
def show_tutorial3():
    return render_template('tutorial3.html')

@app.route('/tutorial/4', methods = ["GET"])
def show_tutorial4():
    queue_from_webapp_to_simulator.put({"restart": "jsons/home_goals/goal0-tut1.json"})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial4.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/tutorial/5', methods = ["GET"])
def show_tutorial5():
    return render_template('tutorial5.html')

@app.route('/tutorial/6', methods = ["GET"])
def show_tutorial6():
    queue_from_webapp_to_simulator.put({"restart": "jsons/home_goals/goal0-tut2.json"})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial6.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/tutorial/7', methods = ["GET"])
def show_tutorial7():
    return render_template('tutorial7.html')

@app.route('/tutorial/8', methods = ["GET"])
def show_tutorial8():
    queue_from_webapp_to_simulator.put({"restart": "jsons/home_goals/goal0-tut3.json"})
    should_webapp_start = queue_from_simulator_to_webapp.get()
    return render_template('tutorial8.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route('/workerId', methods = ["POST"])
def addworkerid():
    global workerId
    workerId = request.form["workerId"]
    print (workerId)
    return ""
    # show_tutorial1()
    # return render_template('index.html', list_of_predicates = dict_of_predicates.keys(), workerId = workerId, world_objects = world_objects)

@app.route("/arguments")
def return_arguments_for_predicate():
	text = request.args.get('predicate')
	return render_template("arguments.html", arguments_list = list(enumerate(dict_of_predicates[text].items())), world_objects = world_objects, constraints_dict = constraints_dict[text], dropdown_states = dropdown_states)
    
def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/simulator_state')
def get_simulator_state():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/execute_move", methods = ["POST"])
def execute_move():
    print (request.form)
    predicate = request.form["predicate"]
    l = []
    front_end_objects = []
    i = 0
    while True:
        if ("arg" + str(i) in request.form):
            front_end_object = request.form["arg" + str(i)]
            front_end_objects.append(front_end_object)
            if front_end_object in renamed_objects:
                l.append(renamed_objects[front_end_object])
            else:
                l.append(front_end_object)
            i += 1
        else:
            break
    d = {
        'actions': [
        {
            'name': str(dict_predicate_to_action[predicate]),
            'args': list(l)
        }
        ]
    }
    print (d)
    move_string = predicate + " ( " + str(front_end_objects[0])
    for i in range(1,len(front_end_objects)):
        move_string += " ," + str(front_end_objects[i])
    move_string += " )"
    print (move_string)
    moves_to_show.append(move_string)
    queue_from_webapp_to_simulator.put(d)
    return move_string

@app.route("/showObject", methods = ["POST"])
def showObject():
    object_to_show = request.form["object"]
    if object_to_show in renamed_objects:
        object_to_show = renamed_objects[object_to_show]
    print (object_to_show)
    queue_from_webapp_to_simulator.put({"showObject": object_to_show})
    return ""

@app.route("/rotateCameraLeft", methods = ["POST"])
def rotateCameraL():
    queue_from_webapp_to_simulator.put({"rotate": "left"})
    return ""

@app.route("/rotateCameraRight", methods = ["POST"])
def rotateCameraR():
    queue_from_webapp_to_simulator.put({"rotate": "right"})
    return ""

@app.route("/zoomIn", methods = ["POST"])
def zoomIn():
    queue_from_webapp_to_simulator.put({"rotate": "in"})
    return ""

@app.route("/zoomOut", methods = ["POST"])
def zoomOut():
    queue_from_webapp_to_simulator.put({"rotate": "out"})
    return ""

@app.route("/toggle", methods = ["POST"])
def toggle():
    queue_from_webapp_to_simulator.put({"rotate": None})
    return ""

@app.route("/undo_move", methods = ["GET"])
def undo_move():
    queue_from_webapp_to_simulator.put({"undo": True})
    return ""

@app.route("/check_error", methods = ["GET"])
def is_error():
    try:
        err_string = queue_for_error.get(block = False)
        return err_string
    except:
        return ""

if __name__ == '__main__':
    inp = "jsons/input_home.json"
    p = mp.Process(target=simulator, args=(queue_from_webapp_to_simulator,queue_from_simulator_to_webapp,queue_for_error))
    p.start()
    should_webapp_start = queue_from_simulator_to_webapp.get()
    app.run(host='0.0.0.0', threaded=True)