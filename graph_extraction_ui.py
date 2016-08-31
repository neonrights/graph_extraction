'''
graph_extraction_ui.py
Sha Lai
8/6/2016

This program does the same job as the original program does but interacts with
the user via a UI instead.
'''

import sys
import numpy as np
import imutils
import cv2
from common import *
from graph_extraction import *
from math import sqrt, inf, fabs
from os import listdir
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# This class helps redirecting stdout to the text area.
class EmittingStream(QObject):
   _stdout = None
   _stderr = None
   textWritten = pyqtSignal(str)
   
   def flush(self):
      pass
      
   def fileno(self):
      return -1
      
   def write(self, text):
      #if (not self.signalsBlocked):
         #self.textWritten.emit(str(text))
      self.textWritten.emit(str(text))

class Window(QWidget):
   def __init__(self):
      super(Window, self).__init__()
      sys.stdout = EmittingStream(textWritten = self.outputText)
      self.input = ''
      self.state = 0
      self.initUI()
   
   def initUI(self):
      # set up the basic window
      self.window = QWidget()
      self.window.setWindowTitle("Graph Extraction GUI")
      
      # create image area
      self.label = QLabel()
      
      # create a text area for outputs
      self.outputConsole = QTextEdit(self)
      self.outputConsole.setReadOnly(True)
      pal = self.outputConsole.palette()
      pal.setColor(QPalette.Base, QColor(255, 255, 255))
      self.outputConsole.setPalette(pal)
      self.outputConsole.setStyleSheet("QTextEdit {color:black}")
      font = self.outputConsole.font()
      font.setPointSize(OUTPUT_FONT)
      self.outputConsole.setFont(font)
      self.outputConsole.setFixedSize(400, OUTPUT_CONSOLE_HEIGHT)
      
      
      print("Welcome!")
      self.dir_path = GRAPH_PATH
      self.input_dir = listdir(self.dir_path)
      print("Files in the input directory:")
      print_list(self.input_dir)
      print("Please provide the file by index of the graph: ")
      
      
      # create a text area for inputs
      self.inputConsole = QLineEdit(self)
      self.inputConsole.returnPressed.connect(self.onChanged)
      
      
      # set up the layout
      grid = QGridLayout()
      grid.addWidget(self.label, 1, 1)
      grid.setAlignment(self.label, Qt.AlignHCenter)
      
      vBox = QVBoxLayout()
      vBox.addLayout(grid)
      vBox.addWidget(self.outputConsole)
      vBox.addWidget(self.inputConsole)
      
      self.window.setLayout(vBox)
      
      # display the window
      self.window.show()
   
   # Takes an openCV image as a parameter, shrinks the image if necessary, and then
   # displays the image to the target area in the window.
   def loadImage(self, cvImage):
      h, w, channel = cvImage.shape
      bytesPerLine = 3 * w
      self.pixmap = QPixmap(QImage(cvImage.data, w, h, bytesPerLine, QImage.Format_RGB888))
      if w >= GRAPH_SIZE_MAX[0]:
         self.pixmap = self.pixmap.scaledToWidth(GRAPH_SIZE_MAX[0], Qt.KeepAspectRatio)
      elif h >= GRAPH_SIZE_MAX[1]:
         self.pixmap = self.pixmap.scaledToHeight(GRAPH_SIZE_MAX[1], Qt.KeepAspectRatio)
      self.label.setPixmap(self.pixmap)
      self.outputConsole.setFixedSize(self.pixmap.size().width(), OUTPUT_CONSOLE_HEIGHT)
      
   def loadImageGray(self, cvImage_gray):
      h, w = cvImage.shape
      bytesPerLine = 3 * w
      self.pixmap = QPixmap(QImage(cvImage.data, w, h, bytesPerLine, QImage.Format_RGB888))
      if w >= GRAPH_SIZE_MAX[0]:
         self.pixmap = self.pixmap.scaledToWidth(GRAPH_SIZE_MAX[0], Qt.KeepAspectRatio)
      elif h >= GRAPH_SIZE_MAX[1]:
         self.pixmap = self.pixmap.scaledToHeight(GRAPH_SIZE_MAX[1], Qt.KeepAspectRatio)
      self.label.setPixmap(self.pixmap)
   
   # Writes the passed in text to the output console.
   def outputText(self, text):
      cursor = self.outputConsole.textCursor()
      cursor.movePosition(QTextCursor.End)
      cursor.insertText(text)
      self.outputConsole.setTextCursor(cursor)
      #self.textEdit.ensureCursorVisible()
   
   # Examines the input, and performs the corresponding work. This function
   # operates based on state. Each state, probably excluding the last one,
   # consists of three sub-state: takes user input, perform corresponding
   # work, and then provide the starting information of the next state.
   def onChanged(self):
      self.input = self.inputConsole.text()
      self.inputConsole.clear()
      print(self.input)
      
      # get the graph
      if self.state == 0:
         self.graph, self.graph_gray = self.get_image(self.input, "graph")
         if not self.graph is None and not self.graph_gray is None:
            self.state += 1
            self.break_point = get_threshold(self.graph_gray)
            self.loadImage(self.graph)
            
            # ask the user for the template
            self.dir_path = TEMPLATE_PATH
            self.input_dir = listdir(self.dir_path)
            print("Files in the input directory:")
            print_list(self.input_dir)
            print("Please provide the file by index of the template: ")
      
      # get the template
      elif self.state == 1:
         self.template, self.template_gray = self.get_image(self.input, "template")
         if not self.template is None and not self.template_gray is None:
            self.state = 2
            self.template, (self.tH, self.tW), self.radius = process_template(self.template)
            
            # start to find the vertices
            self.nodes = []
            self.nodes_center = []
            self.graph_display = self.graph.copy()
            self.graph_work = self.graph_gray.copy()
            print("How many vertices are you looking for?(0 means done) ")
            
      # whlie looking for the vertices, keep asking the user for more
      elif self.state == 2:  
         if self.input == '0':
            self.state = 2.5
            
            # start to ask the user to sort the vertices
            print("Please indicate non-vertex elements in the list in a sequence of indices or \"done\" to proceed to next step:")
         else:
            if not is_valid_type(self.input, int, "Please provide an integer!"):
               pass
            elif int(self.input) < 0:
               print("Please provide a positive integer!")
            else:
               
               locate_vertices(int(self.input), self.graph_work, self.template, self.tW, self.tH, self.nodes)
               print("Current vertices:")
               print_list(self.nodes)
               draw_vertices(self.graph_display, self.nodes, self.tW, self.tH, True, False)
               self.loadImage(self.graph_display)
               print("How many vertices are you looking for?(0 means done) ")
      
      # remove the false vertices
      elif self.state == 2.5:
         if self.input == DONE:
            self.state = 3
            print("Do you want to sort the vertices?(y/n)")
         else:
            indices = self.input.split()
            valid = True
            for i in indices:
               if not is_valid_type(i, int, "Invalid input detected!"):
                  valid = False
               elif int(i) < BASE or int(i) >= BASE + len(self.nodes):
                  print("Error: index out of bound!\n")
                  valid = False
            if valid == True:
               self.nodes = remove_indices(self.input, self.nodes)
               graph_display = self.graph.copy()
               draw_vertices(graph_display, self.nodes, self.tW, self.tH, True, False)
               self.loadImage(graph_display)
               print("Current vertices:")
               print_list(self.nodes)
               print("Please indicate non-vertex elements in the list in a sequence of indices or \"done\" to proceed to next step:")
      
      # ask the user if they want to sort the vertices
      elif self.state == 3:
         if self.input[0] == 'y' or self.input[0] == 'Y':
            self.state = 3.5
            
            # ask the user for the method
            print("Please indicate the method by index you want to help" +
                  " sorting:")
            print("1. One-by-one,")
            print("2. Once-for-all.")
         elif self.input[0] == 'n' or self.input[0] == 'N':
            self.end_sorting()
      
      # the user wants to sort
      elif self.state == 3.5:
         self.index_list = []
         if self.input == '1': # one-by-one
            self.state = 3.51
            self.current_index = 0
            print("What's the correct index value of the vertex " + str(self.current_index + BASE) + ". " + str(self.nodes[self.current_index]) + "?")
         elif self.input == '2': # once-for-all
            self.state = 3.52
            print("Please provide a sequence of correct indices for each vertex or \"done\" to proceed to next step:")
         else:
            print("Invalid input, please try again!")
         
      # the user wants to sort and chooses method 1
      elif self.state == 3.51:
         index = self.input
         if is_valid_type(index, int, "Please provide a valid integer."):
            index = int(index)
            if index < BASE or index >= BASE + len(self.nodes):
               print("Error: index out of bound!\n")
            elif index in self.index_list:
               print("Duplicate index detected, please provide another one.")
            else:
               self.index_list.append(index)
               self.current_index += 1
            if len(self.index_list) == len(self.nodes): # when done sorting
               self.end_sorting()
            else:
               print("What's the correct index value of the vertex " + str(self.current_index + BASE) + ". " + str(self.nodes[self.current_index]) + "?")
      # once-for-all
      elif self.state == 3.52:
         indices = self.input
         try:
            indices = indices.split()
            if not len(indices) == len(self.nodes):
               print("Not enough integers or too many of them, please try again.")
            else:
               for i in range(len(indices)):
                  if int(indices[i]) in self.index_list:
                     print("Duplicate index detected, please provide another one.")
                     break
                  else:
                     self.index_list.append(int(indices[i]))
               if len(self.index_list) == len(self.nodes):
                  if not max(self.index_list) + 1 - BASE == len(self.index_list):
                     print("The given input is not a valid arithmetic sequence!")
                  else:
                     self.end_sorting()
         except:
            print("Please provide a sequence of valid integers.")
            
      # the user answers if they want to thin the image
      elif self.state == 4:
         if len(self.input) > 0:
            if self.input[0] == 'y' or self.input[0] == 'Y':
               self.contours = extract_contours(self.graph_gray, self.nodes, self.tW, self.tH, self.break_point, True)
               print("Number of contours detected: " + str(len(self.contours)))
               print("Please indicate a method to help extract the edges:")
               print("1. Simple")
               print("2. Normal")
               self.state = 4.5
            elif self.input[0] == 'n' or self.input[0] == 'N':
               self.contours = extract_contours(self.graph_gray, self.nodes, self.tW, self.tH, self.break_point, False)
               print(self.radius)
               print("Please indicate a method to help extract the edges:")
               print("1. Simple")
               print("2. Normal")
               self.state = 4.5
            else:
               print("Please answer y(es) or n(o)!")
      
      # the user answers which method to extract edges
      elif self.state == 4.5:
         if len(self.input) > 0:
            if self.input == '1' or self.input == '2':
               print("Retrieving edge data....")
               self.E, self.edges_center = get_edges(self.contours, self.nodes_center, self.radius, int(self.input))
               print(len(self.E))
               graph_display = self.graph.copy()
               draw_edges(graph_display, self.edges_center, False)
               self.loadImage(graph_display)
               print("Please indicate non-edge elements in the list in a sequence of indices or \"done\" to proceed to next step:")
               self.state = 5
            else:
               print("Invalid input detected!")
               
      elif self.state == 5:
         if self.input == DONE:
            self.state = 6
            graph_display = self.graph.copy()
            draw_vertices(graph_display, self.nodes, self.tW, self.tH, True, False)
            draw_edges(graph_display, self.edges_center, False)
            self.loadImage(graph_display)
            print("Number of edges detected: " + str(len(self.E)))
            print("Edges:")
            print(self.E)
         else:
            indices = self.input.split()
            valid = True
            for i in indices:
               if not is_valid_type(i, int, "Invalid input detected!"):
                  valid = False
               elif int(i) < BASE or int(i) >= BASE + len(self.eddes):
                  print("Error: index out of bound!\n")
                  valid = False
            if valid == True:
               self.edges = remove_indices(self.input, self.edges)
               graph_display = self.graph.copy()
               draw_vertices(graph_display, self.nodes, self.tW, self.tH, True, False)
               draw_edges(graph_display, self.edges, False)
               self.loadImage(graph_display)
               print("Edges:")
               print(self.E)
               print("Please indicate non-edge elements in the list in a sequence of indices or \"done\" to proceed to next step:")
               
   def __del__(self):
      try:
         sys.stdout = sys.__stdout__
         sys.stderr = sys.__stderr__
      except:
         pass
   
   def get_image(self, response, keyword):
      image = None
      image_gray = None
      if is_valid_type(response, int, "Please provide an integer!"):
         index = int(response)
         if index >= 0 + BASE and index < len(self.input_dir) + BASE:
            try:
               image = cv2.imread(self.dir_path + self.input_dir[index - BASE])
               image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               valid = True
               print("Selected " + keyword + " file: " + 
                  str(self.input_dir[index - BASE]))
            except:
               print("Error: the " + keyword + " file is invalid or \
                  cannot be processed.")
               response = ''
         else:
            print("Error: index out of bound!\n")
      return image, image_gray
   
   def end_sorting(self):
      if self.state > 3.5:
         result = [0] * len(self.nodes)
         for i in range(len(self.index_list)):
            result[self.index_list[i] - BASE] = self.nodes[i]
         self.nodes = result
         graph_display = self.graph.copy()
         draw_vertices(graph_display, self.nodes, self.tW, self.tH, True, False)
         self.loadImage(graph_display)
      self.nodes_center = get_center_pos(self.nodes, self.tW, self.tH)
      self.state = 4
      print("Do you want to thin the image?(y/n)")
   
   def extract_edges(self, thin):
      graph_display = self.graph.copy()
      draw_vertices(graph_display, self.nodes, self.tW, self.tH, True, False)
      self.loadImage(graph_display)

###############################################################################
#                              Executing Codes                                #

if __name__ == "__main__":
   
   state = 0
   app = QApplication(sys.argv)
   window = Window()
   
   #window.loadImage(graph) 
   sys.exit(app.exec_()) 
   
   