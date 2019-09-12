#!/usr/bin/env python3
from importlib import import_module
import os
from flask import Flask, render_template, Response, request

app = Flask(__name__)

dict_of_predicates = {
		"move":{"argument1" : "dropdown", "argument2": "dropdown", "argument3":"dropdown"},
		"grasp": {"argument1" : "dropdown", "argument2": "dropdown"}
	}

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route("/arguments")
def return_arguments_for_predicate():
	text = request.args.get('predicate').lower()
	return render_template("arguments.html", arguments_list = dict_of_predicates[text].keys())


