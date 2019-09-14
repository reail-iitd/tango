#!/usr/bin/env python3
from importlib import import_module
import os
from flask import Flask, render_template, Response, request
from camera import Camera
from base_camera import BaseCamera

app = Flask(__name__)

dict_of_predicates = {
		"Move":{"argument1" : "dropdown-objects", "argument2": "dropdown-objects", "argument3":"dropdown-objects"},
		"Grasp": {"argument1" : "dropdown-objects", "argument2": "dropdown-places"},
		"Pick": {"argument1": "dropdown-objects"}
	}
world_objects = ["apple", "banana", "table"]

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
	print (request.form["arg0"])
	return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)