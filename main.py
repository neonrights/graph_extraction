'''
main.py
Sha Lai
9/17/2016

This program interacts with the user in console to use the graph_extraction
program to retrieve a mathematical graph from a digital image.


'''

from graph_extraction import *
from common import *

###############################################################################
#                              User Interaction                               #

def get_image(dir_path, keyword):
   """Interacts with the user to read an image from a provided directory.
   Parameters
   ----------
   dir_path : string
      The path of the directory which the program should be looking into.
   keyword : string
      Specifies the type of content(graph vs template). For printing only.
   Returns
   -------
   image : numpy matrix of integers
      The original image.
   image_gray : numpy matrix of integers
      The grayscale version of the image.
   """
   image = None
   image_gray = None
   response = ''
   valid = False
   while response == '' or valid == False:
      input_dir = listdir(dir_path)
      print("Files in the input directory:")
      print_list(input_dir)
      response = input("Please provide the file by index of the " + keyword + 
         ": ")
      if is_valid_type(response, int, "Please provide an integer!"):
         index = int(response)
         if index >= 0 + BASE and index < len(input_dir) + BASE:
            try:
               image = cv2.imread(dir_path + input_dir[index - BASE], -1)
               # change all the transparent pixels to background ones
               for i in range(image.shape[0]):
                  for j in range(image.shape[1]):
                     if image[i][j][3] == 0:
                        image[i][j][0] = 255
                        image[i][j][1] = 255
                        image[i][j][2] = 255
               image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               valid = True
               print("Selected " + keyword + " file: " + 
                  str(input_dir[index - BASE]))
            except:
               print("Error: the " + keyword + " file is invalid or \
                  cannot be processed.")
               response = ''
         else:
            print("Error: index out of bound!\n")
   return image, image_gray


def get_images(show_graph = False, show_template = False):
   """Interacts with the user to read the image of the graph as well as that of
   the template used to locate the vertices.
   Parameters
   ----------
   show_graph and show_template : boolean
      When set to True, the original graph image or the template will be shown
      to the user.
   Returns
   -------
   graph and graph_gray : numpy matrix of integers
      The original graph image and it's grayscale version.
   template_gray : numpy matrix of integers
      The grayscale version of the template.
   break_point : int
      The value that is used to distinguish the background and the content in
      the original graph image.
   """
   graph = None
   graph_gray = None
   template = None
   template_gray = None

   graph, graph_gray = get_image(GRAPH_PATH, "graph") 
   template, template_gray = get_image(TEMPLATE_PATH, "vertex example")
   
   if show_graph:
      cv2.startWindowThread()
      cv2.namedWindow(GRAPH)
      cv2.imshow(GRAPH, graph)
      cv2.waitKey(1)
   if show_template:
      cv2.startWindowThread()
      cv2.namedWindow(TEMPLATE)
      cv2.imshow(TEMPLATE, template)
      cv2.waitKey(1)
   break_point = get_threshold(graph_gray)
   return graph, graph_gray, template_gray, break_point


def get_valid_list(user_input, prompt_sentence, list_length):
   """Keep asking the user to provide a list of indices until a valid one(can
      be DONE) is entered.
   Parameters
   ----------
   user_input : string
      The input from the user.
   promt_sentence : string
      A string that is used to ask the user.
   list_length : int
      The length of the list that is being processed.
   """
   valid = False
   while valid == False:
      while user_input == '':
         user_input = input(prompt_sentence)
      indices = user_input.split()
      valid = True
      if user_input != DONE:
         for i in indices:
            if not is_valid_type(i, int, "Invalid input detected!"):
               valid = False
               user_input = ''
            elif int(i) < BASE or int(i) >= BASE + list_length:
               print("Error: index out of bound!\n")
               valid = False
               user_input = ''
   return user_input


def hide_vertices(image, nodes_center, radius, color = 0):
   """Puts a circle filled with the background color on each vertex.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that is being studied.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   Returns
   -------
   None
   """
   for n in nodes_center:
      cv2.circle(image, n, int(radius), color, cv2.FILLED)


def get_kernel_shape(shape_str):
   """Returns the kernel shape constant in OpenCV package.
   Parameters
   ----------
   shape_str : string
      A string starting with r/R(ectangle), e/E(llipse), or c/C(ross).
   Returns
   -------
   OpenCV shape indicator or None if the parameter is invalid.
   """
   if shape_str[0] == 'r' or shape_str[0] == 'R':
      return cv2.MORPH_RECT
   elif shape_str[0] == 'e' or shape_str[0] == 'E':
      return cv2.MORPH_ELLIPSE
   elif shape_str[0] == 'c' or shape_str[0] == 'C':
      return cv2.MORPH_CROSS
   else:
      print("Invalid shape string.")
      return None

def adjust_labels(image_display, window_name, ref_pos, rel_pos,
                  font_size = FONT_SIZE, font_thickness = FONT_THICKNESS):
   """Allows the user to adjust some font properties on the displaying image.
   Parameters
   ----------
   image_display : numpy matrix of integers
      The image that will be used to displayed. Note that this must be the
      original image possibly with only frames on.
   window_name : string
      The name of the displaying window.
   ref_pos : List[(int, int)]
      Stores the reference position for text.
   rel_pos : (int, int)
      Stores the coordinates relative to each point in ref_pos.
   font_size : float
      Stores the font size value.
   font_thickness : int
      Stores the font thickness value.
   Returns
   -------
   rel_pos : (int, int)
      Stores the coordinates relative to each point in ref_pos.
   font_size : float
      Stores the font size value.
   font_thickness : int
      Stores the font thickness value.
   """
   print("Please enter a valid command to adjust font properties, type " +
         "\"help\" for assistance, or enter " + DONE + " to proceed.")
   user_input = ''
   while user_input != DONE:
      while len(user_input) == 0:
         user_input = input()
      input_list = user_input.split()
      if len(input_list) > 1:
         if input_list[0] == 'font' and input_list[1] == 'size':
            if len(input_list) == 2:
               print("Font size = " + str(font_size))
            elif is_valid_type(input_list[2], float, "Please provide a " +
                                 "number for font size"):
               font_size = float(input_list[2])
               image_display_c = image_display.copy()
               label_vertices(image_display_c, ref_pos, rel_pos, font_size,
                              font_thickness)
               cv2.imshow(window_name, image_display_c)
               cv2.waitKey(1)
            user_input = ''
         elif input_list[0] == 'font' and input_list[1] == 'thickness':
            if len(input_list) == 2:
               print("Font thickness = " + str(font_thickness))
            elif is_valid_type(input_list[2], int, "Please provide an integer!"):
               font_thickness = int(input_list[2])
               image_display_c = image_display.copy()
               label_vertices(image_display_c, ref_pos, rel_pos, font_size,
                              font_thickness)
               cv2.imshow(window_name, image_display_c)
               cv2.waitKey(1)
            user_input = ''
         elif input_list[0] == 'text' and input_list[1] == 'position':
            if len(input_list) == 2:
               print("Currently the texts are at " + str(rel_pos) +
                        " relative to the top-right corner of the frames.")
            elif len(input_list) == 4:
               directions = {'up' : -1, 'left' : -1, 'down' : 1, 'right' : 1}
               if input_list[2] in directions and \
                  is_valid_type(input_list[3], int,\
                  "Please provide an integer to indicate the amount of units."):
                  dir = input_list[2]
                  val = int(input_list[3])
                  if dir == 'left' or dir == 'right':
                     rel_pos = (rel_pos[0] + directions[dir] * val, rel_pos[1])
                  else:
                     rel_pos = (rel_pos[0], rel_pos[1] + directions[dir] * val)
                  image_display_c = image_display.copy()
                  label_vertices(image_display_c, ref_pos, rel_pos, font_size,
                                 font_thickness)
                  cv2.imshow(window_name, image_display_c)
                  cv2.waitKey(1)
            user_input = ''
         else:
            print("Invalid command, please try again.")
            user_input = ''
      elif user_input[0] == 'h' or user_input[0] == 'H':
         user_input = ''
         print("There are three properties that can be adjusted here:")
         print("font size, font thickness, and text position.")
         print("Should you enter any of these key terms, the current value " +
               "would be displayed.")
         print("If you want to adjust the font size to some number x, please" +
               " enter \"font size x\".")
         print("If you want to adjust the font thickness to some number n, " +
               "please enter \"font thickness n\".")
         print("If you want to adjust the text position to some " + 
               "direction(up, down, left, or right) dir for \nsome distance " +
               "d, please enter \"text position dir d\".")
         print("Also note that the font thickness value and the distance " +
               "value should both be integers.")
      elif user_input != DONE:
         print("Invalid command, please try again.")
         user_input = ''
   return rel_pos, font_size, font_thickness
         

def find_vertices(image_display, image_work, template, tW, tH):
   """Repeated asks the user for the amount of undetected vertices in the
   graph until all are marked. The false ones can be removed at the end.
   Parameters
   ----------
   image_display : numpy matrix of integers
      The image that is intended to be displayed to the user.
   image_work : numpy matrix of integers
      The image that is intended to be hidden for intermediate process.
   template : numpy matrix of integers
      A piece of the original image containing an example of the vertices.
   tW and tH : int
      The dimension values of the template.
   Returns
   -------
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the detected vertices.
   rel_pos : (int, int)
      Stores the coordinates relative to each point in ref_pos.
   font_size : float
      Stores the font size value.
   font_thickness : int
      Stores the font thickness value.
   """
   image_display2 = image_display.copy()
   nodes = [] # stores the upper-left coordinates of the vertices
   nodes_center = [] # stores the center coordinates of the vertices
   rel_pos = (int(tW / 3), int(tH * 2 / 3))
   user_input = 1
   while user_input > 0:
      user_input = input("How many vertices are you looking for?(0 means " + 
         "done)")
      if len(user_input) > 0:
         if is_valid_type(user_input, int, "Cannot recognize the input, " +
                           "please provide a number."):
            user_input = int(user_input)
            if user_input >= 0:
               locate_vertices(user_input, image_work, template, tW, tH, nodes)
               highlight_vertices(image_display2, nodes, tW, tH)
               image_display3 = image_display2.copy()
               label_vertices(image_display3, nodes, rel_pos)
               cv2.startWindowThread()
               cv2.namedWindow("Vertices")
               cv2.imshow("Vertices", image_display3)
               cv2.waitKey(1)
               print("Current vertices:")
               print_list(nodes)
            elif user_input < 0:
               print("Please provide a non-negative integer.")
               user_input = PLACE_HOLDER_INT
      else:
         user_input = 1
      
   cv2.destroyWindow(GRAPH)
   cv2.destroyWindow("Vertices")
   

   cv2.startWindowThread()
   cv2.namedWindow("Vertices with Labels")
   cv2.imshow("Vertices with Labels", image_display3)
   cv2.waitKey(1)
   rel_pos, font_size, font_thickness = adjust_labels(image_display2.copy(),
      "Vertices with Labels", nodes, rel_pos)
   
   # attempts to remove all the false vertices
   user_input = ''
   while not user_input == DONE:
      image_display4 = image_display.copy()
      highlight_vertices(image_display4, nodes, tW, tH)
      label_vertices(image_display4, nodes, rel_pos, font_size, font_thickness)
      cv2.startWindowThread()
      cv2.namedWindow("Vertices with Labels")
      cv2.imshow("Vertices with Labels", image_display4)
      cv2.waitKey(1)
      user_input = ''
      user_input = get_valid_list(user_input, "Indicate non-vertex elements " +
                                    "in a sequence of indices, enter " +
                                    "\"done\" to proceed to the next step:\n",\
                                    len(nodes))
      if user_input != DONE:
         nodes = remove_nodes(user_input, nodes)
         print("Current vertices:")
         print_list(nodes)
         user_input = ''
   
   print("Current vertices:")
   print_list(nodes)
   return nodes, rel_pos, font_size, font_thickness


def sort_vertices(nodes, image_display, window_name, rel_pos, font_size, font_thickness):
   """Interacts with the user to sort the vertices.
   Parameters
   ----------
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the detected vertices.
   image_display : numpy matrix of integers
      The image that is intended to be displayed to the user.
   window_name : string
      The name of the window that is used to display the image.
   rel_pos : (int, int)
      Stores the coordinates relative to each point in ref_pos.
   font_size : float
      Stores the font size value.
   font_thickness : int
      Stores the font thickness value.
   Returns
   -------
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the detected vertices while the
      false ones(decided by the user) has been removed.
   """
   answer = ''
   while answer == '':
      answer = input("Do you want to sort the vertices? (y/n): ")
      if len(answer) > 0:
         if answer[0] == 'y' or answer[0] == 'Y':
            result = [(0, 0)] * len(nodes)
            
            sorting_option = 0
            while sorting_option == 0:
               print("Please indicate the method by index you want to help" + 
                  " sorting:")
               print("1. One-by-one,")
               print("2. Once-for-all.")
               response = input("Your choice? ")
               if is_valid_type(response, int, "The input is not an integer."):
                  sorting_option = int(response)
                  if sorting_option < 1 or sorting_option > 2:
                     print("Please provide a valid integer!")
                     sorting_option = 0
            
            if sorting_option == 1:
               index_list = []
               for i in range(len(nodes)):
                  valid = False
                  while valid == False:
                     index = input("What's the correct index value of the " + 
                        "vertex " + str(BASE + i) + ". " + str(nodes[i]) + "? ")
                     if is_valid_type(index, int, "Please provide a valid " +
                                       "integer."):
                        index = int(index)
                        if index < BASE or index >= BASE + len(nodes):
                           print("Error: index out of bound!\n")
                        elif index in index_list:
                           print("Duplicate index detected, please provide " + 
                              "another one.")
                        else:
                           valid = True
                           index_list.append(index)
            elif sorting_option == 2:
               valid = False
               while valid == False:
                  index_list = []
                  indices = input("Please provide a sequence of correct indices" + 
                     " for each vertex or \"done\" to proceed to next step:\n")
                  try:
                     indices = indices.split()
                     if not len(indices) == len(nodes):
                        print("Not enough integers or too many of them, please " + 
                           "try again.")
                     else:
                        for i in range(len(indices)):
                           if int(indices[i]) in index_list:
                              print("Duplicate index detected, please provide " + 
                                 "another one.")
                              break
                           else:
                              index_list.append(int(indices[i]))
                        if len(index_list) == len(nodes):
                           if not max(index_list) + 1 - BASE == len(index_list):
                              print("The given input is not a valid " +
                                    "arithmetic sequence!")
                           else:
                              valid = True
                  except:
                     print("Please provide a sequence of valid integers.")
            else:
               print("Cannot sort the vertices, check the method indicating " +
                     "value.")
               sys.exit(1)
            for i in range(len(index_list)):
               result[index_list[i] - BASE] = nodes[i]
            nodes = result
            highlight_vertices(image_display, nodes, tW, tH)
            label_vertices(image_display, nodes, rel_pos, font_size,
                           font_thickness)
            cv2.startWindowThread()
            cv2.namedWindow(window_name)
            cv2.imshow(window_name, image_display)
            cv2.waitKey(1)
            print("Updated list of vertices:")
            print_list(nodes)
         elif answer[0] == 'n' or answer[0] == 'N':
            break
         else:
            answer = ''
            print("Please answer with y/n.")
      else:
         answer = ''
         print("Please answer with y/n.")
   return nodes


def noise_reduction(image_gray, break_point, nodes_center, radius):
   """This function interacts with the user to help them perform mathematical 
   morphology operations on the grayscale image, attempting to reduce the noise
   of the original image.
   Parameters
   ----------
   image_gray : numpy matrix of integers
      The grayscale version of the original image.
   break_point : int
      A value indicating the threshold between the background and the content.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      A value indicating the size of the block that is used to hide the
      vertices.
   """
   image_bin_inv = get_binary_image_inv(image_gray, break_point)
   show_binary_image(image_bin_inv, "Noise Reduction")
   image_stack = []
   message_stack = []
   image_stack.append(image_bin_inv)
   message_stack.append('')
   kernel_shape_str = KERNEL_SHAPE
   kernel_shape = get_kernel_shape(kernel_shape_str)
   kernel_size = KERNEL_SIZE
   kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
   response = ''
   last_step = ''
   result = image_bin_inv
   print("Current kernel shape and size is: " + kernel_shape_str +
         str(kernel_size))
   print("Pelease indicate which operation to perform:")
   print("(a)djust kernel(this cannot be undone)")
   print("(c)over vertices")
   print("(d)ilation")
   print("(e)rosion")
   print("(p)roceed")
   print("(t)hin")
   print("(u)ndo" + message_stack[-1])
   while len(response) == 0:
      response = input("Your choice is: ")
      if len(response) > 0:
         if response[0] == 'a':
            new_shape_str = ''
            while new_shape_str == '':
               print("Please provide the desired shape of the kernel:")
               print("(C)ross")
               print("(E)llipse")
               print("(R)ectangle")
               new_shape_str = input("Your choice is: ")
               if len(new_shape_str) > 0:
                  if new_shape_str[0] in ['r', 'R', 'e', 'E', 'c', 'C']:
                     kernel_shape = get_kernel_shape(new_shape_str)
                     kernel_shape_str = KERNEL_STR_MAP[new_shape_str[0]]
                  else:
                     print("Please provide a valid string!")
                     new_shape_str = ''
            new_size = ''
            while new_size == '':
               new_size = input("Enter the new kernel size as a single integer: ")
               if is_valid_type(new_size, int):
                  new_size = int(new_size)
                  if new_size > min(image_gray.shape[0], image_gray.shape[1])\
                     or new_size <= 0:
                     print("Please provide a valid integer!")
                     new_size = ''
                  else:
                     kernel_size = (new_size, new_size)
                     print("Kernel size is now " + str(kernel_size))
               else:
                  print("Please provide a valid integer!")
                  new_size = ''
            response = ''
            kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
            print("Updated kernel:")
            print(kernel)
         elif response[0] in ['c', 'd', 'e', 't']:
            image_temp = image_stack[-1].copy()
            if response[0] == 'c':
               print("Covering vertices....")
               hide_vertices(image_temp, nodes_center, radius)
               last_step = " (last step was (c)over vertices)"
            elif response[0] == 'd':
               print("Performing dilation....")
               image_temp = denoise(image_temp, cv2.dilate, kernel, 1)
               last_step = " (last step was (d)ilation)"
            elif response[0] == 'e':
               print("Performing erosion....")
               image_temp = denoise(image_temp, cv2.erode, kernel, 1)
               last_step = " (last step was (e)rosion)"
            elif response[0] == 't':
               print("Performing image thinning....")
               image_temp = thin(image_temp)
               last_step = " (last step was (t)hin)"
            image_stack.append(image_temp)
            message_stack.append(last_step)
            show_binary_image(image_temp, "Noise Reduction")
            print("Operation complete.")
            response = ''
         elif response[0] == 'u':
            if len(image_stack) > 1:
               image_stack.pop()
               message_stack.pop()
               image_temp = image_stack[-1]
               show_binary_image(image_temp, "Noise Reduction")
            else:
               print("There is no last step.")
            response = ''
         elif response[0] == 'p':
            result = image_stack[-1]
         else:
            print("Invalid input, please try again!")
            response = ''
         print("Current kernel shape and size is: " + kernel_shape_str +
               str(kernel_size))
         print("Pelease indicate which operation to perform:")
         print("(a)djust kernel(this cannot be undone)")
         print("(c)over vertices")
         print("(d)ilation")
         print("(e)rosion")
         print("(p)roceed")
         print("(t)hin")
         print("(u)ndo" + message_stack[-1])
   return result
            

def extract_edges(image_work, nodes_center, radius):
   """This function attempts to extract all the edges from the image with
   vertices being hidden.
   Parameters
   ----------
   image_work : numpy matrix of integers
      The image that is intended to be hidden for intermediate process.
   nodes_center : List[[int, int]]
      Stores the estimated center coordinates of the vertices.
   radius : float
      A value indicating the size of the block that is used to hide the
      vertices.
   Returns
   -------
   E : List[(int, int)]
      Each tuple (a, b) represents an edge connecting vertex a and b.
   """
   E = []
   endpoints = get_endpoints(image_work, nodes_center, radius)
   deg_seq = []
   for i in range(len(endpoints)):
      deg_seq.append(len(endpoints[i]))
   trails = []

   for i in range(len(endpoints)):
      for j in range(len(endpoints[i])):
         edge, trail = get_edge(image_work, endpoints[i][j], [], [],\
                                 nodes_center, i, radius)
         if edge != PLACE_HOLDER_COOR and not edge in E and\
            not (edge[1], edge[0]) in E:
            E.append(edge)
            trails.append(trail)
   image_temp = np.zeros(image_work.shape, np.uint8)
   
   
   '''
   edge, trail = get_edge(image_work, endpoints[0][0], [], [], nodes_center,\
                           0, endpoints, radius)
   print(edge)
   print(len(trail))
   
   
   image_temp = np.zeros(image_work.shape, np.uint8)
   for p in trail:
      image_temp[p[1], p[0]] = 1
   show_binary_image(image_temp, "trails", True)
   '''
   
   return E, deg_seq

def display_edges(E):
   """Shows all the edges.
   Parameters
   ----------
   E : List[(int, int)]
      Each tuple (a, b) represents an edge connecting vertex a and b.
   Returns
   -------
   None
   """
   print("Displaying edges....")
   print_list(E)

#                               End of Section                                #
###############################################################################
###############################################################################
#                              Executing Codes                                #

if __name__ == "__main__":   
   # Obtain the files.
   graph, graph_gray, template, break_point = get_images(True)
   
   # Process the template.
   template, (tH, tW), radius = process_template(template)
   
   # Find all the vertices. In particular variable nodes stores a list of
   # nodes' upper-right corner.
   nodes, rel_pos, font_size, font_thickness =\
      find_vertices(graph.copy(), graph_gray.copy(), template, tW, tH)
   
   # If neccesary, sort the vertices such that the order matches the given one.
   nodes = sort_vertices(nodes, graph.copy(), "Vertices with Labels", rel_pos,
                           font_size, font_thickness)
   
   nodes_center = get_center_pos(nodes, tW, tH)
   
   complete = False
   while not complete:
      graph_work = noise_reduction(graph_gray, break_point, nodes_center,\
                                    radius)
      E, deg_seq = extract_edges(graph_work, nodes_center, radius)
      display_edges(E)
      user_input = ''
      while len(user_input) == 0:
         user_input = input("Attempt again?(y/n)")
         if len(user_input) > 0:
            if user_input[0] == 'n' or user_input[0] == 'N':
               complete = True
            elif user_input[0] != 'y' and user_input[0] != 'Y':
               print("Invalud input, please try again!")
               user_input = ''
   halt = input("HALT (press enter to end)")