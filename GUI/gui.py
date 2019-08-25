import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGroupBox, QVBoxLayout
from PyQt5.QtGui import QIcon

class App(QWidget):

    def __init__(self):
        super().__init__()
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
        self.show()
    
    def overlying_structure(self):
        task_description_group = QGroupBox("Task Description (Goal to achieve)")
        instruction_input_group = QGroupBox("Instruction Input")
        feedback_group = QGroupBox("Robot Response")
        debug_group = QGroupBox("Debugging Information")

        self.main_layout = QVBoxLayout()
        for group_box in [task_description_group, instruction_input_group, feedback_group, debug_group]:
            group_box.setStyleSheet("font-weight: bold;")
            self.main_layout.addWidget(group_box)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())