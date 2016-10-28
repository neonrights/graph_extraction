'''
main.py
Sha Lai
9/16/2016

This file contains most of the core functions being used in the project.

The first step before using this program is to crop an example of the vertices,
and then pass the template to the function process_template so that we can get
a new version where only the edges remain together with some helpful values
which will be used in later procedures.

To begin the user needs to call the function locate_vertices with an approaximate
number of vertices, a copy of the gray-scale image, the template and its size
values obtained in the previous step, and a list of nodes(this can be empty).
The function implements the multi-scale template matching technique, that is,
the function rescales the image, for each scale, it finds a piece of the image
which matches the template the most by calling a function matchTemplate from
the OpenCV package and store the upper-left coordinate of that piece into the
provided list nodes. In order to prevent the function from finding the same
piece of the image, everytime a piece is found, a white block is placed right
on its location so that the piece will be covered up. Note that this is done on
a copy of the image so that the original one will not be damaged. This function
may need to be called multiple times in order to find all the vertices. If
there is still one or more vertices remaining on the image undetected after a
few times then the image cannot be handled by this program. Otherwise, when all
the vertices are found the user must examine the node list, indicate the
indices of false vertices, and use the function remove_indices to get rid of
them. Be careful about the impact of the BASE value. Lastly the user may also
want to call teh function get_center_pos to obtain the center coordinates of
the true vertices. Once this step is finalized, a list of vertices can be
obtained. Moreover for the rest of the program whenever we need to deal with
a full list of elements that are all related to the vertices(the center
coordinate of each vertex for example), the elements in the new list must be in
the same order as in the nodes and the nodes_center. In other words we have
assigned each vertex an ID number implicitly.

Next we want to find the edges. There are two main steps involved. Firstly we
want to get rid of the noise on the image as much as possible. A commonly used
approach is mathematical morphology. In this file the denoise function can help
to do the job. When the image is in a desired stage after the noise reduction,
the function thin needs to be called such that the skeleton of the content can
be extracted. And then the user may want to hide the vertices and make sure
that the result is also in a desired view(no extra overlaps between edges for
example). Note that after this step a binary version of the image should be
obtained and we will use it throughout the rest of the program. We can now
start attempting to extract the edges.

From now on assume that in the binary image the content is represented by 1s
and the background is marked by 0s. The first thint to do is to find all the
starting points of the edges. With the vertices hidden, it is easy to scan all
the content pixels and find the ones with the smallest distances to some
located vertex. The output of this step should be a list of sets of pixels
where the ith set in the list should contain the coordinates of the pixels that
are at the end of the edges connecting to the ith vertex. Next we need to
extract the edges. For each starting point, the algorithm to determine which
vertex is the corresponding edge linked to can be described as follows:

1. Check if the current position has already reached a vertex that is not the
   starting one. If so then we are done and return, otherwise go to step 2.
2. Obtain a neighborhood of the current potition, examine the neighborhood and
   extract a list of pixels that are marked 1 and are not in the known list.
   Add them into the known list. If the length of the list is 0 then go to step
   3. Otherwise go to step 4.
3. Abort since this is the end of a line and there is no vertex to link to.
4. If the length of the list trail is 0 then for each candidate pixel
   go back to step 1. Otherwise, if the length is 1 and if the current position
   is not in the conjunction of two edges then append the current position to
   the trail list, if not in the conjunction then take the only candidate back
   to the first step. If there are more than one candidate, then determine if
   the current position is a cross point. If so then check if it is also in a
   conjunction part. If not append the current position to the trail list. If
   this is not a cross then the current position also needs to be added to the
   trail list. Proceed to step 5.
5. Using the trail list, construct a list of vectors between each pair of
   adjacent points in the trail, then compute a weighted sum of these vectors.
   For each candidate compute a vector that points to them from the current
   position. Compare each candidate vector and the sum vector, choose the one
   with the smallest angle with the sum vector and take that corresponding
   candidate pixel to step 1. If the returned result is not a valid vertex then
   choose the 2nd closest vector and its corresponding destination position.
   Repeat until a valid result is obtained or the options are run out.

The way in which a cross is detected is simple: for each candidate pixel
compute a vector that points from the current position to it and for each pair
of these candidate vectors check if their angle is greater than or equal to 90
degree. The idea behind the approach to decide which direction to go next at a
cross point is to make the decision based on a weighted sum of the previous
experience and the previous experience does not include any conjunction part.
Currently what is problematic is the weighting function applied on the vectors.
For now I have not yet found a really "good" funtion to use as much as I do
believe that my approach is reasonable.
'''

from common import *
import numpy as np
###############################################################################
#                             Helper Functions                                #

def highlight_vertices(image, nodes, tW, tH):
   """Highlights each vertex with a frame.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that is will be drawn on.
   nodes : List[(int, int)]
      Stores the upper-right coordinates of the vertices.
   tW and tH : int
      Dimension values of the template.
   show_indices : boolean
      Indices will be displayed if True.
   using_console : boolean
      True if the program is run on the console. This argument is needed since
      OpenCV stores the RGB color in a different order.
   Return
   ------
   None
      Everything will be drawn on the given image.
   """
   for i in range(len(nodes)):
      cv2.rectangle(image, nodes[i], (nodes[i][0] + tW,  nodes[i][1] + tH),
                     RECT_COLOR, RECT_THICKNESS)

def label_vertices(image, ref_pos, rel_pos, font_size = FONT_SIZE,
                  font_thinkness = FONT_THICKNESS):
   """Labels all the vertices with indices.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that will be used to labeled.
   ref_pos : List[(int, int)]
      List of original coordinates for each vertex(the top-right corner).
   rel_pos : (int, int)
      Coordinates relaive to the original ones.
   font_size : float
      Indicating the font size.
   font_thickness : int
      Indicating how thick the text is.
   Returns
   -------
   None
   """
   for i in range(len(ref_pos)):
      x = abs(ref_pos[i][0] + rel_pos[0])
      y = abs(ref_pos[i][1] + rel_pos[1])
      if x >= image.shape[1]:
         x -= 2 * rel_pos[0]
      if y >= image.shape[0]:
         y -= 2 * rel_pos[1]
      position = (x, y)
      cv2.putText(image, str(i + BASE), position, cv2.FONT_HERSHEY_SIMPLEX,\
         font_size, FONT_COLOR, font_thinkness, cv2.LINE_AA, False)

def draw_edges(image, edges_center, using_console = True):
   """Puts a label near the center pixel of each edge.
   Parameters
   ----------
   image : numpy matrix of integers
      The image that is will be drawn on.
   edges_center : List[[int, int]]
      Stores the coordinates of center pixel of each edge.
   using_console : boolean
      True if the program is run on the console. This argument is needed since
      OpenCV stores the RGB color in a different order.
   Returns
   -------
   None
      Everything will be drawn on the given image.
   """
   if not using_console:
      font_color = FONT_COLOR_G
   else:
      font_color = FONT_COLOR
   for i in range(len(edges_center)):
      cv2.putText(image, str(i + BASE), (edges_center[i][0],\
                  edges_center[i][1]), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, \
                  font_color, FONT_THICKNESS, cv2.LINE_AA)
      

def get_threshold(image_gray, show_detail = False):
   """Determines the threshold of color values which will be used to distinguish
   the background and the actual content. If the METHOD is 'STATIC' then a set
   value will be used immediately while if the METHOD is 'DYNAMIC' then the
   mode of all the pixel values will be used.
   Parameters
   ----------
   image_gray : numpy matrix of integers
      The grayscale image that is being studied.
   show_detail : boolean
      More printing statements if True. This is for debugging purpose.
   Returns
   -------
   threshold : int
      The threshold of color values.
   """
   # use a set value
   if METHOD == 'STATIC':
      if show_detail:
         print("METHOD = " + METHOD)
         print(THRESHOLD + " is being used.")
      return THRESHOLD
      
   # evaluate the mode of all the pixels in the image
   elif METHOD == 'DYNAMIC':
      if show_detail:
         print("METHOD = " + METHOD)
         print('mode = ' + str(mode(mode(image_gray)[0][0])[0][0]))
      return mode(mode(image_gray)[0][0])[0][0] - 1
   else:
      sys.exit('Cannot recognize the METHOD, please check the common file.')


def get_distance(p1, p2):
   """Computes the Euclidean distance between two points in 2D.
   Parameters
   ----------
   p1 and p2 : [int, int]
      Represents coordinates.
   """
   return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def remove_nodes(user_input, nodes):
   """Removes false vertices based on the user input.
   Parameters
   ----------
   user_input: string
      A string of edge indices subject to be removed.
   nodes : List[(int, int)]
      Each coordinate marks the upper-right corner of a piece of the original
      image.
   Returns
   -------
   nodes : same as above
      The false ones have been removed.
   """
   index_remove = user_input.split()
   for i in index_remove:
      nodes[int(i) - BASE] = PLACE_HOLDER_COOR
   nodes = [element for element in nodes if element != PLACE_HOLDER_COOR]
   return nodes


def get_center_pos(nodes, tW, tH):
   """Calculates the center coordinates of each vertex, based on the template
   taken by the user in some early step.
   Parameters
   ----------
   nodes : List[(int, int)]
      Each coordinate marks the upper-right corner of a piece of the original
      image.
   tW and tH : int
      The dimension values of the template that was used to find the vertices.
   Returns
   -------
   nodes_center :
   """
   nodes_center = []
   for node in nodes:
      nodes_center.append((int(node[0] + tW / 2), int(node[1] + tH / 2)))
   return nodes_center


def remove_edges(user_input, E, edges_center):
   """Removes false edges based on the user input.
   Parameters
   ----------
   user_input: string
      A string of edge indices subject to be removed.
   E : List[(int, int)]
      Each tuple contains two indices of the vertices in the vertex list.
   edges_center : List[(int, int)]
      Each element marks the estimated position of the center pixel of an edge.
   Returns
   -------
   E and edges_center : same as above
      The false ones have been removed.
   """
   index_remove = user_input.split()
   removing = []
   for i in index_remove:
      try:
         removing.append(int(i) - BASE)
      except:
         print("Invalid input, please try again!")
   E = [E[i] for i in range(len(E)) if not i in removing]
   edges_center = [edges_center[i] for i in range(len(edges_center))\
                     if not i in removing]
   return E, edges_center

###########################  IMAGE THINNING  ##################################
"""The following functions implements zhang-seun's image thinning algorithm
without too many explanations. The user can search for the algorithm for more
details. In order to perform image thinning, only the function thin needs to be
called by the client program.
"""
# Testing if the current pixel satisfies the required statement group #1.
def test1(p, A, B):
   return p[0] == 1 and \
         B >= 2 and B <= 6 and \
         A == 1 and \
         (p[1] * p[3] * p[5] == 0) and \
         (p[3] * p[5] * p[7] == 0)

# Testing if the current pixel satisfies the required statement group #2.
def test2(p, A, B):
    return p[0] == 1 and \
           B >= 2 and B <= 6 and \
           A == 1 and \
           (p[1] * p[3] * p[7] == 0) and \
           (p[1] * p[5] * p[7] == 0)

# Examines all the pixels in a given images, except for the outer ones, using
# a provided test to collect all the satisfied pixels and returns the list.
def examine(mat, test):
    output = []
    for i in range(1, mat.shape[0] - 1):
        for j in range(1, mat.shape[1] - 1):
            p = (int(mat[i, j]), int(mat[i - 1, j]), int(mat[i - 1, j + 1]), \
                  int(mat[i, j + 1]), int(mat[i + 1, j + 1]), \
                  int(mat[i + 1, j]), int(mat[i + 1, j - 1]), \
                  int(mat[i, j - 1]), int(mat[i - 1, j - 1]))
            A = 0
            B = p[1]
            for k in range(2, len(p)):
                if p[k] == 1 and p[k - 1] == 0:
                    A += 1
                B += p[k]
            if p[1] == 1 and p[-1] == 0:
                A += 1
            if test(p, A, B):
                output.append((i, j))
    return output

def thin(image_bin):
   """Thins a given binary image with zhang-seun's algorithm, assuming that in
   the image 1 represents the actual content while 0 represents the background.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      The binary image that is being studied.
   Returns
   -------
   mat : numpy matrix of integers
      A thinned version of the original binary image.
   """
   mat = image_bin.copy()
   if mat.shape[0] < 3 or mat.shape[1] < 3:
      sys.exit("Invalid image input: " + str(mat.shape))
   keep_itr = True
   while keep_itr:
      output1 = examine(mat, test1)
      for coor in output1:
         mat[coor[0], coor[1]] = 0
      output2 = examine(mat, test2)
      for coor in output2:
         mat[coor[0], coor[1]] = 0
      if len(output1) + len(output2) == 0:
         keep_itr = False
   return mat

############################  END OF SUBSECTION  ##############################




def denoise(image_bin, method, kernel, itr = 0):
   """Performs mathematical morphology operation on the provided binary image
   in order to reduce the noise.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      A binary image that is being studied.
   method : function
      This function must either be cv2.dilate or cv2.erode in order to make
      sure the program works correctly.
   kernel : numpy matrix of integers
      The dimension values are suggested to be odd numbers.
   """
   result = method(image_bin.copy(), kernel, iterations = itr)
   return result


def get_neighborhood(image, location):
   """Extracts the coordinates of the points in the neighborhood.
   Parameters
   ----------
   image : numpy matrix of intergers
      The image that is being studied.
   location : [int, int]
      The coordinate in the form [x, y] that marks the desired position.
   Returns
   -------
   p : List[[int, int]]
      Each element is a coordinate in the form [x, y]. The indices are
      associated with the relative positions in the neighborhood in this form:
      8 1 2
      7 0 3
      6 5 4
      Additionally if the location happens to be on an edge or a corner of the
      image then [PLACE_HOLDER_INT, PLACE_HOLDER_INT] will be used for
      out-of-reach points.
   """
   temp = [PLACE_HOLDER_INT, PLACE_HOLDER_INT]
   p = [temp[:]] * 9
   p[0] = location[:]
   if location[1] - 1 >= 0:
      p[1] = [location[0], location[1] - 1]
      if location[0] + 1 < image.shape[1]:
         p[2] = [location[0] + 1, location[1] - 1]
   if location[0] + 1 < image.shape[1]:
      p[3] = [location[0] + 1, location[1]]
      if location[1] + 1 < image.shape[0]:
         p[4] = [location[0] + 1, location[1] + 1]
   if location[1] + 1 < image.shape[0]:
      p[5] = [location[0], location[1] + 1]
      if location[0] - 1 >= 0:
         p[6] = [location[0] - 1, location[1] + 1]
   if location[0] - 1 >= 0:
      p[7] = [location[0] - 1, location[1]]
      if location[1] - 1 >= 0:
         p[8] = [location[0] - 1, location[1] - 1]
   return p


def get_neighborhood_value(image, location):
   """Extracts the pixel values in the neighborhood.
   Parameters
   ----------
   image : numpy matrix of intergers
      The image that is being studied.
   location : [int, int]
      The coordinate in the form [x, y] that marks the desired position.
   Returns
   -------
   p : List[int]
      A list of pixel values in the neighborhood. The indices are associated
      with the relative positions in the neighborhood in this form:
      8 1 2
      7 0 3
      6 5 4
      Additionally is the current position happens to be on an edge or a corner
      of the image then PLACE_HOLDER_INT will be used to represent the pixel
      value.
   """
   p = [PLACE_HOLDER_INT] * 9
   p[0] = image[location[1], location[0]]
   if location[1] - 1 >= 0:
      p[1] = image[location[1] - 1, location[0]]
      if location[0] + 1 < image.shape[1]:
         p[2] = image[location[1] - 1, location[0] + 1]
   if location[0] + 1 < image.shape[1]:
      p[3] = image[location[1], location[0] + 1]
      if location[1] + 1 < image.shape[0]:
         p[4] = image[location[1] + 1, location[0] + 1]
   if location[1] + 1 < image.shape[0]:
      p[5] = image[location[1] + 1, location[0]]
      if location[0] - 1 >= 0:
         p[6] = image[location[1] + 1, location[0] - 1]
   if location[0] - 1 >= 0:
      p[7] = image[location[1], location[0] - 1]
      if location[1] - 1 >= 0:
         p[8] = image[location[1] - 1, location[0] - 1]
   return p


def print_neighborhood_values(nv):
   """Prints the pixel values in the neighborhood.
   Parameters
   ----------
   nv : List[int]
      Represents the neighborhood pixel values. A neighborhood has the form
      8 1 2
      7 0 3
      6 5 4
      where each number is associated with the index in the list.
   """
   print(str(nv[8]) + " " + str(nv[1]) + " " + str(nv[2]))
   print(str(nv[7]) + " " + str(nv[0]) + " " + str(nv[3]))
   print(str(nv[6]) + " " + str(nv[5]) + " " + str(nv[4]))


def get_vector(p_from, p_to):
   """Computes the vector from one point to another.
   Parameters
   ----------
   p_from and p_to : [int, int]
      Coordintates in the form [x, y].
   Returns
   -------
   vector : [int, int]
   """
   return [p_to[0] - p_from[0], p_to[1] - p_from[1]]


def get_weight(x):
   """Computes the weighting coefficient for a given variable.
   Parameters
   ----------
   x : int
      This serves as the independent variable in the function f(x).
   Returns
   -------
   f(x) : float
      A weighting coefficient.
   """
   # using a normal distribution
   mu = 1
   sigma = 2
   f = (exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)) / (sqrt(2 * pi) * sigma)))
   return f


def get_vector_sum(list):
   """Computes the weighted sum of a list of vectors. Currently the weight
   function is set to be the normal distribution functino.
   Parameters
   ----------
   list : List[[int, int]]
      Each element is a pair of coordinates in the form [x, y].
   mu : float
      The mean of the weight function.
   sigma : float
      The deviation of the weight funtion.
   Returns
   -------
   result : [float, float]
      The weighted sum of the given list of vectors.
   """
   result = [0, 0]
   for i in range(len(list)):
      result[0] += list[-1 - i][0] * get_weight(i)
      result[1] += list[-1 - i][1] * get_weight(i)
   n = np.linalg.norm(result, 2)
   result[0] /= n
   result[1] /= n
   return result


def is_cross(current_pos, next):
   """Determines if the current position is a cross between two curves.
   Parameters
   ----------
   current_pos : List[int]
      Stores the current coordinates in the form [x, y].
   next : List[[int, int]]
      Stores the candidate coordinates each of which is in the form [x, y].
   Returns
   -------
   True if the current position is a cross point, False otherwise.
   """
   vectors = []
   for i in range(len(next)):
      vectors.append(get_vector(current_pos, next[i]))
   for i in range(len(vectors) - 1):
      for j in range(i + 1, len(vectors)):
         cos_theta = np.inner(vectors[i], vectors[j]) /\
            (np.linalg.norm(vectors[i], 2) * np.linalg.norm(vectors[j], 2))
         if cos_theta <= 0:
            return True
   return False
#                               End of Section                                #
###############################################################################
###############################################################################
#                               Main Functions                                #

def process_template(template):
   """Extracts the edges in the template.
   Parameters
   ----------
   template : a numpy matrix of integers
   Returns
   -------
   template : a numpy matrix of integers
      Contains the edges(gap between pixel values).
   (tH, tW) : (int, int)
      Dimension values of the template.
   radius : float
      Half of the length of the diagonal of the template.
   """
   template = cv2.Canny(template, 50, 200)
   (tH, tW) = template.shape[:2]
   
   # this will serve as a threshold of the distance from some end point of
   # an edge to the center of a vertex
   radius = sqrt(pow((1 + SCALE_PERCENTAGE) * tH, 2) + pow((1 + 
      SCALE_PERCENTAGE) * tW, 2)) / 2
   return template, (tH, tW), radius


def locate_vertices(amount, image, template, tW, tH, nodes):
   """This function performs multi-scale template matching in order to locate
      all the vertices, contributed by Adrian Rosebrock from PyImageResearch.
      Parameters
      ----------
      amount : int
         The estimated amount of vertices that remains undetected.
      image : numpy matrix of integers
         The image that are being studied.
      template : numpy matrix of integers
         The cropped piece of the original image containing an example of the
         vertices.
      tW and tH : int
         The dimension values of the template.
      nodes : List[(int, int)]
         Stores the upper-right coordinate of the vertices  that have been
         detected so far.
      Returns
      -------
      nodes : List[(int, int)]
         This list is returned by reference.
   """
   for i in range(amount):
      found = None
      # rescale the image, for each scale, find the piece that has the best
      # matching score
      for scale in np.linspace(0.2, 1.0, 20)[::-1]:
         resized = imutils.resize(image, width = \
            int(image.shape[1] * scale))
         r = image.shape[1] / float(resized.shape[1])
        
         if resized.shape[0] < tH or resized.shape[1] < tW:
            break
            
         edged = cv2.Canny(resized, 50, 200)
         result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
         (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
         if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
      (_, maxLoc, r) = found
      (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
      (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

      # image is not intended to be displayed
      cv2.rectangle(image, (startX, startY), (endX, endY), \
         (255, 255, 225), cv2.FILLED)
      nodes.append((int(maxLoc[0] * r), int(maxLoc[1] * r)))

   
def get_endpoints(image_bin, nodes_center, radius):
   """Finds the endpoints of the edge pixels and associate them with the vertices
   by indices.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      Stores the binary image that are being studied, with contents marked by
      1s and background marked by 0s.
   nodes_center : List[(int, int)]
      Each tuple is the coordinate of the center position of a vertex.
   radius : float
      The radius of the circle block covering the vertices.
   Returns
   -------
   endpoints : List[[int, int]]
      The ith list in endpoints contains all the pixel coordinates that are the
      starting points of the edges from the ith vertex.
   """
   endpoints = []
   for i in range(len(nodes_center)):
      endpoints.append([])
   for x in range(image_bin.shape[1]):
      for y in range(image_bin.shape[0]):
         candidates = []
         if image_bin[y, x] == 1:
            for i in range(len(nodes_center)):
               d = get_distance((x, y), nodes_center[i])
               if d < radius + 1:
                  new_end = True
                  endpoints_icopy = endpoints[i][:]
                  for ep in endpoints_icopy:
                     n = get_neighborhood(image_bin, ep)
                     if [x, y] in n:
                        new_end = False
                        if d < get_distance(ep, nodes_center[i]):
                           endpoints[i].remove(ep)
                           endpoints[i].append([x, y])
                  if new_end:      
                     endpoints[i].append([x, y])
   return endpoints


def get_edge(image_bin, current_pos, trail, known, nodes_center,
               starting_index, radius, is_joint = False, image_debug = None):
   """Returns an edge and it's trail. An edge is in the form (a, b) where a
   marks the index of the starting vertex and b the index of the ending
   one. A trail is a list of pixels that connect the two vertices, not
   including the ones in conjunctions between two or more edges.
   Parameters
   ----------
   image_bin : numpy matrix of integers
      This is a binary image where 1 represents the content while 0 marks
      the background.
   current_pos : List[int]
      A two-element list representing the (x, y) coordinate of hte current
      position.
   trail : List[[int, int]]
      Each element list in trail represents a coordinate of the pixel
      connecting the current pixel and the starting one.
   known : set of lists of integers
      It stores all the coordinates of the pixels that have been examinied
      so far.
   nodes_center : List[(int, int)]
      Each tuple is the coordinate of the center position of a vertex.
   starting_index : int
      The index of the starting vertex in the vertex list.
   radius : float
      The radius of the circle block covering the vertices.
   is_joint : boolean
      True if the current_pos is in a conjunction between two or more
      vertices.
   image_debug : numpy matrix of integers
      This is a binary image where 1 represents the content while 0 marks
      the background, for debugging purpose.
   Returns
   -------
   edge : a tuple of integers
      If the function manages to find an edge, then a tuple of two integers
      each of which marks the index of a vertex in the vertex list will be
      returned. Otherwise a default tuple implying that the attempt fails
      will be returned.
   trail : List[[int, int]]
      Each element list in trail represents a coordinate of the pixel
      connecting the current pixel and the starting one.
   """
   for i in range(len(nodes_center)): # check if it has reached a vertex
      if get_distance(nodes_center[i], current_pos) < radius + 1 and\
         i != starting_index:
         #print("Vertex reached.")
         return (starting_index + BASE, i + BASE), trail
   n = get_neighborhood(image_bin, current_pos)
   nv = get_neighborhood_value(image_bin, current_pos)
   
   candidate_indices = []
   for i in range(1, len(n)):
      if nv[i] == 1 and not n[i] in known:
         candidate_indices.append(i)
         known.append(n[i])
   
   if DEBUG_MODE: # just more printing statements
      print("\ncurrent_pos = " + str(current_pos))
      print("len(trail) = " + str(len(trail)))
      print_neighborhood_values(nv)
      print("len(candidate_indices) = " + str(len(candidate_indices)))
   if image_debug is None:
      image_debug = image_bin.copy()
      
   if len(candidate_indices) == 0: # if this is the end of the line/curve
      #print("End of the line.")
      #print_neighborhood_values(nv)
      #cv2.circle(image_bin, current_pos, 5, 1, cv2.FILLED)
      return PLACE_HOLDER_COOR, trail
   else: # if there is at least one possibly unknown pixel ahead
      if len(trail) == 0: # this is a starting point
         #print("This is just a start.")
         known.append(current_pos)
         for i in candidate_indices:
            edge, new_trail = get_edge(image_bin, n[i], [current_pos], known,\
                                       nodes_center, starting_index, radius,\
                                       is_joint, image_debug)
            if edge != PLACE_HOLDER_COOR: # returns the found edge immediately
               return edge, new_trail
         return PLACE_HOLDER_COOR, trail
      else: # we are in the middle of some edge
         if len(candidate_indices) == 1: # no choice to make, go for the next
            if is_joint == False:
               trail.append(current_pos)
            return get_edge(image_bin, n[candidate_indices[0]], trail, known,\
                              nodes_center, starting_index, radius,\
                              is_joint, image_debug)
         else: # need to make a choice
            next = [] # stores the coordinate of the candidate pixels
            for i in candidate_indices:
               next.append(n[i])
            if is_cross(current_pos, next): # if this is a cross point
               #cv2.circle(image_debug, (current_pos[0], current_pos[1]), 4, 1, 1)
               #image_show = get_binary_image(image_debug, 0, 255)
               #cv2.imshow("image_debug", image_show)
               #cv2.waitKey()
               #print("cross:")
               #print_neighborhood_values(nv)
               #print(current_pos)
               
               '''
               If is_joint is True, then this is a point where we are leaving
               the conjunction part, so is_joint should be turned to False;
               otherwise we are entering one.
               '''
               if is_joint:
                  is_joint = False
               else:
                  is_joint = True
                  trail.append(current_pos)
            else: # not a cross
               trail.append(current_pos)
               

            # fit a curve using the previous m points
            m = 5
            if len(trail) < m:
               # fit a second order polynomial
               alpha = np.polyfit(trail[-m:][0], trail[-m:][1], 2)
               # create the third order solution to find min distance
               poly = lambda x: [2*alpha[0]**2, 3*alpha[0]*alpha[1],
                                 1 + 2*alpha[0]*alpha[2] - 2*alpha[0]*x[1] + alpha[1]**2,
                                 -x[0] + alpha[1]*alpha[2] - alpha[1]*x[1]]
               euclid = lambda x: lambda z: (z - x[0])**2 + (np.poly1d(a)(z) - x[1])**2
            else:
               # fit a first order polynomial
               alpha = np.polyfit(trail[:][0], trail[:][1], 1)
               # create the first order solution to find min distance
               poly = lambda x: [alpha[0]^2 + 1, alpha[0]*alpha[1] - alpha[0]*x[1] - x[0]]
               euclid = lambda x: lambda z: (z - x[0])**2 + (np.poly1d(a)(z) - x[1])**2

            # calculate min distance to curve for each point
            best_index = candidate_indices[0]
            best_dist = float('inf')
            for i in candidate_indices:
               # solve roots of polynomial, remove complex values
               roots = np.roots(poly(n[i]))
               roots = [root for root in roots if root.imag == 0]

               # see if candidate point matches the curve the best
               dist = min(map(euclid(n[i]), roots))
               if dist < best_dist:
                  best_index = i
                  best_dist = dist

            trail.append(candidate_indices[best_index])
            return get_edge(image_bin, candidate_indices[best_index], trail,\
                            known, nodes_center, starting_index, radius,\
                            is_joint, image_debug)

            """
            # The following codes intend to determine which direction to go.
            vector_list = []
            for i in range(len(trail) - 1): # sums the vectors along the trail
               vector_list.append(get_vector(trail[i], trail[i + 1]))
            vector_list.append(get_vector(trail[-1], current_pos))
            vector_sum = get_vector_sum(vector_list)
            
            '''
            Compute the cosine value of the angle between the sum vector and
            each candidate one, starting from the direction with the highest
            cosine value, that is, with the smallest angle, select the one that
            yields a valid edge.
            '''
            cos2next = {}
            for i in candidate_indices:
               v_temp = get_vector(current_pos, n[i])
               cos_theta = np.inner(vector_sum, v_temp) /\
                           (np.linalg.norm(vector_sum, 2) *\
                            np.linalg.norm(v_temp, 2))
               cos2next[cos_theta] = n[i]
            keys_list = list(cos2next.keys())
            keys_list.sort()
            result = PLACE_HOLDER_COOR
            new_trail = trail[:]
            temp = 0
            while result == PLACE_HOLDER_COOR and temp < len(keys_list):
               result, new_trail = get_edge(image_bin,\
                                             cos2next[keys_list[-1 - temp]],\
                                             trail, known, nodes_center,\
                                             starting_index, radius, is_joint,\
                                             image_debug)
               temp += 1

            return result, new_trail
            """         
         
#                               End of Section                                #
###############################################################################