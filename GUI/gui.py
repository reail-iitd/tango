import sys
import json

from PyQt5.QtWidgets import QApplication, QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton
from PyQt5.QtGui import QIcon, QFont

tasksfile = "input/tasks.txt"
worldmodelfile = "input/world_model.json"
outputFile = "output/move.json"

class WorldModel:
    def __init__(self):
        self.dir = json.load(open(worldmodelfile, "r"))
    def getObjects(self):
        return self.dir["objects"]

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.worldModel = WorldModel()
        self.title = 'Learning Common Sense Knowledge for Robust Robot Autonomy'
        # Set the location of the window
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        # Set parameters and show the GUI
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.overlying_structure()
        self.setLayout(self.main_layout)
        f = open(tasksfile, "r")
        tasks = f.readlines()
        self.task_description_label.setText(tasks[0])
        self.adjustSize()
        self.show()
    
    def overlying_structure(self):
        task_description_group = QGroupBox("Task Description (Goal to achieve)")
        instruction_input_group = QGroupBox("Instruction Input")
        feedback_label = QLabel("Robot Response: ")
        self.debug_group = QGroupBox("Debugging Information")
        show_debug_check = QCheckBox("Show Debug Information")

        self.task_description_init(task_description_group)
        self.instruction_input_init(instruction_input_group)
        show_debug_check.stateChanged.connect(self.show_debug_group)

        self.main_layout = QVBoxLayout()
        for group_box in [task_description_group, instruction_input_group, feedback_label, show_debug_check,    self.debug_group]:
            if group_box in [task_description_group, instruction_input_group, self.debug_group]:
                group_box.setStyleSheet("font-weight: bold;")
            elif group_box in [feedback_label]:
                group_box.setStyleSheet("font-weight: bold; color: green;")
            self.main_layout.addWidget(group_box)
        self.debug_group.setHidden(True)

    def task_description_init(self, task_description_group):
        self.task_description_label = QLabel("")
        self.task_description_label.setStyleSheet("color: rgb(60,60,60);")
        self.task_description_label.setWordWrap(True)
        taskLayout = QVBoxLayout()
        taskLayout.addWidget(self.task_description_label)
        task_description_group.setLayout(taskLayout)

    def instruction_input_init(self, instruction_input_group):
        instruction_input_layout = QVBoxLayout()

        instructionBoxesLayout = QHBoxLayout()
        self.predicate_input = QLineEdit()
        self.openingBracket = QLabel("(")
        self.arg1_input = QComboBox()
        self.arg1_input.setEditable(True)
        self.comma_1 = QLabel(",")
        self.arg2_input = QLineEdit()
        self.comma_2 = QLabel(",")
        self.arg3_input = QLineEdit()
        self.closingBracket = QLabel(")")
        for i_widget in [self.predicate_input, self.openingBracket, self.arg1_input, self.comma_1, self.arg2_input, self.comma_2, self.arg3_input, self.closingBracket]:
            instructionBoxesLayout.addWidget(i_widget)
        for i_widget in [self.comma_2, self.arg3_input]:
            i_widget.setHidden(True)
        instruction_input_layout.addLayout(instructionBoxesLayout)

        doneLayout = QHBoxLayout()
        self.num_arguments = QCheckBox("Number of arguments is 3")
        self.num_arguments.stateChanged.connect(self.change_num_arguments)
        self.execute_move = QPushButton("Execute Move")
        self.execute_move.clicked.connect(self.writeAction)
        for i_widget in [self.num_arguments, self.execute_move]:
            doneLayout.addWidget(i_widget)
        instruction_input_layout.addLayout(doneLayout)

        instruction_input_group.setLayout(instruction_input_layout)

        object_list = self.worldModel.getObjects()
        for i in object_list:
            self.arg1_input.addItem(i)

    def show_debug_group(self):
        self.debug_group.setHidden(not self.debug_group.isHidden())
    
    def change_num_arguments(self):
        self.comma_2.setHidden(not self.num_arguments.isChecked())
        self.arg3_input.setHidden(not self.num_arguments.isChecked())

    def writeAction(self):
        d = {}
        d["name"] = self.predicate_input.text()
        if (self.num_arguments.isChecked()):
            d["args"] = [self.arg1_input.currentText(), self.arg2_input.text(), self.arg3_input.text()]
        else:
            d["args"] = [self.arg1_input.currentText(), self.arg2_input.text()]
        json.dump(d, open(outputFile, "w"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())