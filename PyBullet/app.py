#!/usr/bin/env python3
from importlib import import_module
import os
import json
from flask import Flask, render_template, Response, request
from camera import Camera
from base_camera import BaseCamera
import multiprocessing as mp
import time

queue_from_webapp_to_simulator = mp.Queue()
queue_from_simulator_to_webapp = mp.Queue()

app = Flask(__name__)

dict_of_predicates = {
		# "Move object to destination":{"source-object" : "dropdown-objects", "destination (near object)": "dropdown-objects"},
		"Push object to destination": {"Object to push" : "dropdown-objects", "Destination (near this object)": "dropdown-objects"},
		"Pick source and place on destination": {"Object to pick": "dropdown-objects", "Object to place on": "dropdown-objects"},
        "Move robot to object" : {"Destination (near this object)": "dropdown-objects"},
        "Open/Close object" : {"Object to open or close": "dropdown-objects", "Open or Close it": "dropdown-states"}
	}

dict_predicate_to_action = {
    # "Move object to destination": "moveAToB",
    "Push object to destination": "pushTo",
    "Pick source and place on destination": "pickNplaceAonB",
    "Move robot to object": "moveTo",
    "Open/Close object" : "changeState"
}

# Unnecessary (can be removed)
d = json.load(open("jsons/world_home.json"))["entities"]
world_objects = []
renamed_objects = {}
constraints_dict = json.load(open("jsons/constraints.json"))
dropdown_states = ["open", "close"]
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

def simulator(queue_from_webapp_to_simulator, queue_from_simulator_to_webapp):
    import husky_ur5
    import src.actions
    queue_from_simulator_to_webapp.put(True)
    print ("Waiting")
    husky_ur5.firstImage()
    called_undo_before = False
    while True:
        inp = queue_from_webapp_to_simulator.get()
        if ("rotate" in inp or "zoom" in inp or "toggle" in inp):
            husky_ur5.changeView(inp["rotate"])
        elif "undo" in inp:
            if (not called_undo_before):
                husky_ur5.undo()
                husky_ur5.undo()
                called_undo_before = True
            else:
                husky_ur5.undo()
        else:
            husky_ur5.execute(inp)
            called_undo_before = False

@app.route('/')
def index():
    return render_template('index.html', list_of_predicates = dict_of_predicates.keys())

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
    i = 0
    while True:
        if ("arg" + str(i) in request.form):
            front_end_object = request.form["arg" + str(i)]
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
    queue_from_webapp_to_simulator.put(d)
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
if __name__ == '__main__':
    inp = "jsons/input_home.json"
    p = mp.Process(target=simulator, args=(queue_from_webapp_to_simulator,queue_from_simulator_to_webapp))
    p.start()
    should_webapp_start = queue_from_simulator_to_webapp.get()
    app.run(host='0.0.0.0', threaded=True)