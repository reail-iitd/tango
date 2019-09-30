#!/usr/bin/env python3
from importlib import import_module
import os
import json
from flask import Flask, render_template, Response, request
from camera import Camera
from base_camera import BaseCamera
import multiprocessing as mp
import time

q = mp.Queue()

app = Flask(__name__)

dict_of_predicates = {
		"Move object to destination":{"source-object" : "dropdown-objects", "destination (near object)": "dropdown-objects"},
		"Push object to destination": {"source-object" : "dropdown-objects", "destination-place": "dropdown-places"},
		"Pick source and place on destination": {"source-object": "dropdown-objects", "destination-place": "dropdown-objects"},
        "Move robot to object" : {"destination-object": "dropdown-objects"}
	}

dict_predicate_to_action = {
    "Move object to destination": "moveAtoB",
    "Push object to destination": "pushTo",
    "Pick source and place on destination": "pickNplaceAonB",
    "Move robot to object": "moveTo"
}

d = json.load(open("../PyBullet/jsons/world_home.json"))["entities"]
world_objects = []
renamed_objects = {}
for obj in d:
    if (("ignore" in obj) and (obj["ignore"] == "true")):
        continue
    if ("rename" in obj):
        world_objects.append(obj["rename"])
        renamed_objects[obj["rename"]] = obj["name"]
    else:
        world_objects.append(obj["name"])

def convertActionsFromFile(action_file):
    inp = None
    with open(action_file, 'r') as handle:
        inp = json.load(handle)
    return(inp)

def simulator(q):
    import husky_ur5
    import src.actions
    print ("Waiting")
    while True:
        inp = q.get()
        husky_ur5.execute(inp)

@app.route('/')
def index():
    return render_template('index.html', list_of_predicates = dict_of_predicates.keys())

@app.route("/arguments")
def return_arguments_for_predicate():
	text = request.args.get('predicate')
	return render_template("arguments.html", arguments_list = list(enumerate(dict_of_predicates[text].items())), world_objects = world_objects)
    
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
        "actions": [
        {
            "name": dict_predicate_to_action[predicate],
            "args": l
        }
        ]
    }
    print (d)
    d = {'actions': [{'name': 'pickNplaceAonB', 'args': ['book', 'box']}]}
    q.put(d)
    return ""

if __name__ == '__main__':
    inp = "jsons/input_home.json"
    p = mp.Process(target=simulator, args=(q,))
    p.start()
    app.run(host='0.0.0.0', threaded=True)