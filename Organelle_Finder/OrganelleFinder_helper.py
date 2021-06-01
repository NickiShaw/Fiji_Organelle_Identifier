# Copyright (C) 2021 Nicolette Shaw - All Rights Reserved
from ij	import IJ, ImagePlus
from ij.gui import Line, Plot
import math
import gc
from ij.plugin.frame import RoiManager
from ij.process import ImageProcessor
from ijopencv.ij import ImagePlusMatConverter as imp2mat
from org.bytedeco.javacpp.opencv_core  import Point2f, vconcat, hconcat, cvmGet, subtract, Size, Point2f, Rect, Mat, MatVector, GpuMatVector, KeyPointVector, Scalar, CvMat, vconcat,ACCESS_READ, Point
from org.bytedeco.javacpp.opencv_imgproc  import pointPolygonTest, line, minAreaRect, floodFill, arcLength, CHAIN_APPROX_NONE,morphologyDefaultBorderValue, morphologyEx, watershed, connectedComponents, threshold, distanceTransform, getStructuringElement, dilate, findContours, CvMoments, cvMoments, contourArea, drawContours, putText, boundingRect, cvtColor, RETR_FLOODFILL, RETR_EXTERNAL, RETR_LIST, CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, COLOR_GRAY2RGB,COLOR_RGB2GRAY
from ij.ImageStack import create, addSlice, deleteSlice, size, getSliceLabel
from ij.gui import GenericDialog, Roi, Line, Plot, PointRoi, NewImage, MessageDialog, WaitForUserDialog, PolygonRoi, NonBlockingGenericDialog
from ijopencv.opencv import MatImagePlusConverter as mat2ip
from ij.plugin import MontageMaker
from ij.plugin.filter import GaussianBlur, EDM
import threading
import sys
import time


def filterbyItem(listItem, value, comparison):
	if not listItem[0]: # pass if the criterion should be evaluated.
		if comparison == "smaller than":
			if value < listItem[1]:
				return True
			else:
				return False
		if comparison == "larger than":
			if value > listItem[1]:
				return True
			else:
				return False
	else:
		return True

def filteringOutput(filtering_dict, keyname):
	if filtering_dict[keyname][0] == False:
		return filtering_dict[keyname][1]
	else:
		return "N/A"
	
def printTime(time_start, time_end, message):
	print(message + ": " + str(time_end - time_start))


class ColourUtils:
	@staticmethod
	def newcol(i):
		if i in range(0,1000,6):
			b = 255
			a=c=0
		if i in range(1,1000,6):
			c = 255
			a=b=0
		if i in range(2,1000,6):
			a = 255
			b=c=0
		if i in range(3,1000,6):
			b = a =  255
			c=0
		if i in range(4,1000,6):
			b = c = 255
			a=0
		if i in range(5,1000,6):
			a = c = 255
			b=0
		return a,b,c

class DevUtils:
	@staticmethod
	def get_methods(object, spacing=20):
	  methodList = []
	  for method_name in dir(object):
	    try:
	        if callable(getattr(object, method_name)):
	            methodList.append(str(method_name))
	    except:
	        methodList.append(str(method_name))
	  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
	  for method in methodList:
	    try:
	        print(str(method.ljust(spacing)) + ' ' +
	              processFunc(str(getattr(object, method).__doc__)[0:90]))
	    except:
	        print(method.ljust(spacing) + ' ' + ' getattr() failed')



class ContourPoint:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def add(self, p):
		self.x += p.x
		self.y += p.y

	def substract(self, p):
		self.x -= p.x
		self.y -= p.y

	def divide(self, p):
		self.x /= p.x
		self.y /= p.y

	def getDistanceFromPoint(self, p):
		x = self.x - p.x
		y = self.y - p.y
		return math.sqrt(x*x + y*y)

	def __str__(self):
		return "(" + str(self.x) + "," + str(self.y) + ")"
		
	def __repr__(self):
		return str(self)

class ContourMathUtils:
	@staticmethod
	def getAngleWithCenter(center, p):
		x = p.x - center.x
		y = p.y - center.y
		angle = math.atan2(y,x)
		if angle <= 0.0:
			angle = 2 * math.pi + angle
		return angle

	@staticmethod
	def comparePoints(a, b):
		angleA = ContourMathUtils.getAngleWithCenter(ContourPoint(0,0), a)
		angleB = ContourMathUtils.getAngleWithCenter(ContourPoint(0,0), b)
		if angleA < angleB:
			return -1
		dA = ContourPoint(0,0).getDistanceFromPoint(a)
		dB = ContourPoint(0,0).getDistanceFromPoint(b)
		if (angleA == angleB) and (dA < dB):
			return -1
		return 1

	@staticmethod
	def getContoursBoundingRects(contours_matvec, margin=0):
		bounding_rects = {}
		for i in range(contours_matvec.size()):
			rect = boundingRect(contours_matvec.get(i))
			x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
			x = x - margin
			y = y - margin
			w = w + margin*2
			h = h + margin*2
			bounding_rects[i] = (x,y,w,h)
		return bounding_rects

	# Computes the list of intersecting contours according to their bounding rectangles,
	# where an additional margin can be provided.
	@staticmethod 
	def computeIntersectingContours(contours_matvec, margin=0):
		def intersects(rect1, rect2):
			return not (rect2[0] >= rect1[0] + rect1[2]  \
    					or rect2[0] + rect2[2] <= rect1[0] \
    					or rect2[1] >= rect1[1] + rect1[3]  \
    					or rect2[1] + rect2[3] <= rect1[1])

		bounding_rects = ContourMathUtils.getContoursBoundingRects(contours_matvec, margin)
		
		num_contours = contours_matvec.size()
		intersections = {}
		for i in range(num_contours):
			for j in range(i + 1, num_contours):
				rect1 = bounding_rects[i]
				rect2 = bounding_rects[j]
				intersections[(i,j)] = intersects(rect1, rect2)

		return intersections

class CustomContour:
	def __init__(self, points):
		self.points = points

	def sortPointsClockwise(self):
		center = ContourPoint(0,0)
		for point in self.points:
			center.add(point)
		center.divide(ContourPoint(len(self.points),len(self.points)))
		for point in self.points:
			point.substract(center)
		self.points = sorted(self.points, cmp=lambda a,b : ContourMathUtils.comparePoints(a,b))
		for point in self.points:
			point.add(center)

	def toMat(self):
		contourmat = Mat(len(self.points), 1, 12)
		for i in range(len(self.points)):
			contourmat.getIntBuffer().put(i * 2, int(self.points[i].x))
			contourmat.getIntBuffer().put(i * 2 + 1,int(self.points[i].y))
		return contourmat

class MatrixUtils:
	@staticmethod
	def getMatCoordinate(mat, x, y):
		width = mat.cols()
		mat_location = y * width + x
		return mat_location
		
	@staticmethod
	def getPixelVal(mat, mat_cord):
		return mat.getByteBuffer().get(mat_cord)
		
	@staticmethod
	def setPixelVal(mat, mat_cord, val):
		mat.getByteBuffer().put(mat_cord, val)

	@staticmethod
	def getLabelVal(mat, mat_cord):
		return mat.getIntBuffer().get(mat_cord)
		
	@staticmethod
	def setLabelVal(mat, mat_cord, val):
		mat.getIntBuffer().put(mat_cord, val)


class PixelUtils:

	@staticmethod
	def getMaxIntensity(img):
		img_width = img.getWidth()
		img_height = img.getHeight()
		max_intensity = 0
		ip = img #.getProcessor()
		for x in range(img_width):
			for y in range(img_height):
				max_intensity = max(max_intensity, ip.getValue(x,y))
		return max_intensity
	
	@staticmethod
	def getAvgIntensity(img):
		img_width = img.getWidth()
		img_height = img.getHeight()
		all_intensities = []
		ip = img.getProcessor()
		for x in range(img_width):
			for y in range(img_height):
				all_intensities.append(ip.getValue(x,y))
		if sum(all_intensities) == 0:
			return 0
		if len(all_intensities) == 0:
			print("An image/ROI passed to getAvgIntensity contains 0 pixels, exiting...")
			exit(0)
		else:
			return sum(all_intensities)/len(all_intensities)

class ImgUtils:
	@staticmethod
	def Img2Mat(img):
		# Convert ImagePlus to Mat object.
		img_matrix = imp2mat.toMat(img.getProcessor())
		return img_matrix

	@staticmethod
	def IP2Mat(imp):
		# Convert ImagePlus to Mat object.
		img_matrix = imp2mat.toMat(imp)
		return img_matrix

	@staticmethod
	def ColourConvert(img_matrix, colmat, conversion):
		if conversion == "GRAY2RGB":
			cvtColor(img_matrix, colmat, COLOR_GRAY2RGB)
		if conversion == "GRAY2BGR":
			cvtColor(img_matrix, colmat, COLOR_GRAY2BGR)
		if conversion == "BGR2GRAY":
			cvtColor(img_matrix, colmat, COLOR_BGR2GRAY)
		if conversion == "RGB2GRAY":
			cvtColor(img_matrix, outmat, COLOR_RGB2GRAY)
		final_image  = mat2ip.toImageProcessor(colmat)
		new_img = ImagePlus("final image", final_image)
		return new_img, colmat

	@staticmethod
	def convertImgColour(img, conversion):
		greymat = ImgUtils.Img2Mat(img)
		new_img, colmat = ImgUtils.ColourConvert(greymat, Mat(), conversion)
		return new_img, greymat, colmat

	@staticmethod
	def convertImpColour2Img(imp, conversion):
		greymat = ImgUtils.IP2Mat(imp)
		new_img, colmat = ImgUtils.ColourConvert(greymat, Mat(), conversion)
		return new_img, greymat, colmat

class ContourSorter:

	# Merges bucket 2 into bucket 1 and updates the bucket mapping.
	@staticmethod
	def merge_buckets(buckets, bucket_index1, bucket_index2, bucket_map):
		for bucket_elem in buckets[bucket_index2]:
			buckets[bucket_index1].append(bucket_elem)
			bucket_map[bucket_elem] = bucket_index1
		buckets[bucket_index2] = []
	
	# Gets the bucket index for an element. 
	# Returns -1 if the element does not belong to any bucket.
	@staticmethod
	def get_bucket_index(bucket_map, elem):
		if elem in bucket_map:
			return bucket_map[elem]
		return -1
	
	# Adds an element to a bucket if it doesn't belong to it already.
	# Updates the bucket mapping.
	@staticmethod
	def add_elem_to_bucket(buckets, bucket_index, elem, elem_bucket_index, contour_bucket_map):
		if elem_bucket_index != bucket_index:
			buckets[bucket_index].append(elem)
			contour_bucket_map[elem] = bucket_index
			
	# Sorts sublists where there is any overlap in their elements.
	@staticmethod
	def getBuckets(filtered_list):
		contour_bucket_map = {}
		buckets = []
		for contour_pair in filtered_list:
			elem1 = contour_pair[0]
			elem2 = contour_pair[1]
			bucket_index1 = ContourSorter.get_bucket_index(contour_bucket_map, elem1)
			bucket_index2 = ContourSorter.get_bucket_index(contour_bucket_map, elem2)
		
			# No element in the pair belongs to a bucket.
			if bucket_index1 == -1 and bucket_index2 == -1:
				bucket_index = len(buckets)
				buckets.append([])	
				ContourSorter.add_elem_to_bucket(buckets, bucket_index, elem1, bucket_index1, contour_bucket_map)
				ContourSorter.add_elem_to_bucket(buckets, bucket_index, elem2, bucket_index2, contour_bucket_map)
				continue
		
			# One element in the pair belongs to a bucket.
			if bucket_index1 == -1 or bucket_index2 == -1:
				bucket_index = max(bucket_index1, bucket_index2)
				ContourSorter.add_elem_to_bucket(buckets, bucket_index, elem1, bucket_index1, contour_bucket_map)
				ContourSorter.add_elem_to_bucket(buckets, bucket_index, elem2, bucket_index2, contour_bucket_map)
				continue
		
			# Both elements in the pair belong to a bucket.
		
			# Both elements already belong to the same bucket.
			if bucket_index1 == bucket_index2:
				continue
		
			# The elements belong to two different buckets. Merge the buckets.
			buckets[bucket_index1].append(elem1)
			buckets[bucket_index2].append(elem2)
			ContourSorter.merge_buckets(buckets, bucket_index1, bucket_index2, contour_bucket_map)
			
		return [x for x in buckets if x != []] # Removes empty lists.

class ROIUtils:
	@staticmethod
	def ROItoImage(ROIMatVec, num_across, num_down):
		out_mat = Mat()
		full_mat = MatVector()
		for v in range(0, num_across * num_down, num_down):
			concat_row = Mat()
			row = MatVector()
			for x in range(v, v + num_down):
				row.push_back(ROIMatVec.get(x))
			vconcat(row,concat_row)
			full_mat.push_back(concat_row)
		hconcat(full_mat,out_mat)
		return out_mat

class ROIProcessor:
	def __init__(self, img, roi_width, roi_height):
		self.img = img
		self.img_width = img.getWidth()
		self.img_height = img.getHeight()
		self.roi_width = roi_width
		self.roi_height = roi_height

	def getVariableROIRange(self):
		origins = []
		x_segments = RangeUtils.getVariableRange(0, self.roi_width, self.img_width)
		y_segments = RangeUtils.getVariableRange(0, self.roi_height, self.img_height)
		for x in x_segments:
			for y in y_segments:
				origins.append((x,y))
		return origins, (len(x_segments), len(y_segments))

	def getMultipleVariableROI(self):
		ROI_range, stack_dimensions = self.getVariableROIRange()
		ROI_matlist = []
		for (x,y) in ROI_range:
			# Adjust the width of the ROI when needed.
			if (x + self.roi_width) > self.img_width:
				true_width = self.img_width - x
			else:
				true_width = self.roi_width
			# Adjust the height of the ROI when needed.
			if (y + self.roi_height) > self.img_height:
				true_height = self.img_height - y
			else:
				true_height = self.roi_height
			rect = Roi(x, y, true_width, true_height)
			new = self.img.duplicate()
			new.setRoi(rect)
			ROI = new.crop()
			roi_mat = imp2mat.toMat(ROI.getProcessor())
			ROI_matlist.append(roi_mat)
		return ROI_range, ROI_matlist, stack_dimensions

class RangeUtils:

	@staticmethod
	def splitPointArrays(array):
		x_points = []
		y_points = []
		for pt in array:
			x_points.append(pt[0])
			y_points.append(pt[1])
		return x_points, y_points

	# Get tuples for start and end locations for the intervals given by width (in range of start to maxim).
	#Shortens the width of the final interval where the length is not divisable by the width given.
	@staticmethod
	def getVariableRange(start, width, maxim):
		collect = []
		i = start
		while i < maxim:
			collect.append(i)
			i = i + width
		return collect
		
# General function to get slope from 2 points.
def getslope(x1, x2, y1, y2):
    rise = y2 - y1
    run = x2 - x1
    if run == 0 :
    	return 1
    if rise == 0 :
    	return 0
    return rise / run

    
class coordinateSolver:
	def __init__(self, r, ox, oy):
		self.r = r
		self.ox = ox
		self.oy = oy

	# To find the distance of the circle edge from center given a pixel height y.
	def solvePythagorean(self, y):
		# r^2 = x^2 + y^2 ==> R = X + Y
		R = self.r ** 2
		Y = y ** 2
		X = R - Y
		x = math.sqrt(X)
		return x

class listManipulator:
	# Expects a sorted list.
	def __init__(self, listA):
		self.listA = listA
		self.listSet = dict()
		for elem in listA:
			self.listSet[elem] = 1
			
	# Find overlapping items in two lists (sort lists before running this).
	def intersection(self, listB): 
		overlap = []
		for elem in listB:
			if elem in self.listSet:
				overlap.append(elem)
		return overlap

	# If there are overlapping items in 2 lists return True, if no overlap return False.
	def overlap(self, listB): 
		for elem in listB:
			if elem in self.listSet:
				return True
		return False

	# If item is found in list, return True, if no match return False.
	def match(self, elem):
		if elem in self.listSet:
			return True
		return False

	# Find points 2 pixels away from the original in another list.
	def watershedIntersection(self, listB): 
		overlap = []
		for elem in listB:
			x = elem[0]
			y = elem[1]
			pixel2_buffer = [(x-2,y-2),(x-2,y-1),(x-2,y),(x-2,y+1),(x-2,y+1),(x+2,y-2),(x+2,y-1),(x+2,y),(x+2,y+1),(x+2,y+2),(x-1,y-2),(x-1,y+2),(x,y-2),(x,y+2),(x+1,y-2),(x+1,y+2)]
			for point in pixel2_buffer:
				if point in self.listSet:
					overlap.append(elem)
					break
		return overlap
	    
	# Returns indexes for the elements in a list that match any item in a list of values.
	def indexList(self, values):
		index_list = []
		for i in range(len(values)):
			ind = self.listA.index(values[i])
			index_list.append(ind)
		return index_list

	# Gives true if the value is found in the elements subset from a list.
	def checkList(self, value, indexes):
	    if self.listA == []:
	        return False
	    # For when you only want to check certain elements in the list.
	    if isinstance(indexes, list):
	        for ind in indexes:
	            if str(value) == str(self.listA[ind]):
	                return True
	        return False
	    # For when you want to use all the items in list.
	    if indexes == -1:
	        for i in self.listA:
	            if str(i) == value:
	                return True
	        return False

	# Find the indexes for an overlap of two lists (indexes correspond to list A).
	def intersectIndex(self, listB):
		return self.indexList(listManipulator(listB).intersection(self.listA))


class pixelInensities:
	def __init__(self, line_pixels):
		self.line_pixels = line_pixels
		
	# Collect a list of indexes along the line where peaks/troughs are found.
	def getPeaksandTroughs(self):
		indexes = []
		for i in range(len(self.line_pixels)-3):
			slope1 = self.line_pixels[i+1] - self.line_pixels[i]
			t = i + 1
			# If the values plateau, only retrieve the slope2 where the plateau ends.
			while self.line_pixels[t] == self.line_pixels[t+1]:
				t += 1
				# Prevent looking beyond the line length.
				if t >= len(self.line_pixels)-3:
					t = len(self.line_pixels)-3
					break
			slope2 = self.line_pixels[t+1] - self.line_pixels[t]
			# Get central point p, if there is a plateau.
			p = int((i + t+1)/2)
			# If the slope changes from positive to negative or vice versa, save the index where this occurs.
			if slope1 < 0 and slope2 > 0 or slope1 > 0 and slope2 < 0:
				indexes.append(p)
		return indexes

class pixelBounds:
	def __init__(self, line_pixels, minimum_slope, minimum_slope_width, min_slope_height, max_size, min_intensity):
		self.line_pixels = line_pixels
		self.minimum_slope = minimum_slope
		self.minimum_slope_width = minimum_slope_width
		self.min_slope_height = min_slope_height
		self.max_size = max_size
		self.min_intensity = min_intensity
		
	# Retrieve the bounds of two slopes given an estimated peak/though location.
	def getSlopeBounds(self, ind):
		# Scan for slope limit behind point.
		back_index = backorigin = ind - 5
		# Do not scan if too close to 0.
		if back_index < 0:
			back_index = 0
		else:
			back_slope = getslope(back_index, back_index - 1, self.line_pixels[back_index], self.line_pixels[back_index - 1])
			while abs(back_slope) >= self.minimum_slope and back_index > 1 and (backorigin - back_index) < self.minimum_slope_width:
			    back_index -= 1
			    back_slope = getslope(back_index, back_index - 1, self.line_pixels[back_index], self.line_pixels[back_index - 1])
		# Scan for slope limit in front of point.
		front_index = frontorigin = ind + 5
		# Do not scan if too close to max_size.
		if front_index >= self.max_size - 1:
			front_index = self.max_size - 2
		else:
			front_slope = getslope(front_index, front_index + 1, self.line_pixels[front_index], self.line_pixels[front_index + 1])
			while abs(front_slope) >= self.minimum_slope and front_index < self.max_size - 2 and (front_index - frontorigin) < self.minimum_slope_width:
			    front_index += 1
			    front_slope = getslope(front_index, front_index + 1, self.line_pixels[front_index], self.line_pixels[front_index + 1])
		return([(back_index, ind), (ind, front_index)])

	# Obtain the approximate bounds of a steep slope.
	def getSlopePairs(self, indexes):
		peak_trough_bounds = []
		for index in indexes:
			bound_tup = self.getSlopeBounds(index)
			# Filter any tuples less than min_slope_width and min_slope_height apart.
			for tup in bound_tup:
				point_xdist = tup[1] - tup[0]
				point_ydist = abs(self.line_pixels[tup[1]] - self.line_pixels[tup[0]])
				# Filter any point pairs that do not reach min_intensity.
				if self.line_pixels[tup[1]] >= self.min_intensity or self.line_pixels[tup[0]] >= self.min_intensity:
					if point_xdist >= self.minimum_slope_width and point_ydist >= self.min_slope_height:
						peak_trough_bounds.append(tup)
		return peak_trough_bounds


class ContourUtils:

	@staticmethod		
	def bestContourIndex(contours_matvec):
		contour_ratings = {}
		for i in range(contours_matvec.size()):
			area, _, circularity, _, _ = ContourFindUtils.contourStats(contours_matvec.get(i))
			contour_ratings[i] = area * circularity
		max_rating = max(contour_ratings.values())
		best_contour_index = contour_ratings.keys()[contour_ratings.values().index(max_rating)]
		return best_contour_index
		
	@staticmethod		
	def moveContours(contours, offsetx, offsety, appending_metvec):
		for contour_int in range(contours.size()):
			contour_length = contours.get(contour_int).getIntBuffer().limit()
			for i in range(contour_length)[::2]:
				x = contours.get(contour_int).getIntBuffer().get(i)
				contours.get(contour_int).getIntBuffer().put(i, x + offsetx)
				y = contours.get(contour_int).getIntBuffer().get(i+1)
				contours.get(contour_int).getIntBuffer().put(i + 1, y + offsety)
			appending_metvec.push_back(contours.get(contour_int))
		return appending_metvec

	# Convert the matrix object from findContours into a list of (list of points) for each contour.
	@staticmethod		
	def getContoursPoints(contours):
		all_contour_points = []
		for contour_int in range(contours.size()):
			contour_length = contours.get(contour_int).getIntBuffer().limit()
			contour_points = []
			for i in range(contour_length)[::2]:
				x = contours.get(contour_int).getIntBuffer().get(i)
				y = contours.get(contour_int).getIntBuffer().get(i+1)
				contour_points.append((x,y))
			all_contour_points.append(contour_points)
		return all_contour_points
	
	# For a MatVector of contour Mats, returns a list of lists containing the coordinates of all points within the contour.
	@staticmethod		
	def getInteriorContoursPoints(contours, img_mat):
		all_contour_points = []
		bounding_rects = ContourMathUtils.getContoursBoundingRects(contours, 0)
		for i in range(contours.size()):
			contour_points = []
			for x in range(bounding_rects[i][0], bounding_rects[i][0] + bounding_rects[i][2]):	#range(img_mat.cols()):
				for y in range(bounding_rects[i][1], bounding_rects[i][1] + bounding_rects[i][3]):	#range(img_mat.rows()):
					point = Point2f(float(x), float(y))
					result = pointPolygonTest(contours.get(i), point, False)
					if result >= 0:
						contour_points.append((x,y))
			all_contour_points.append(contour_points)
		return all_contour_points
			
	# Get intensity from corresponding a list of points in the contour (made by getContoursPoints).
	@staticmethod		
	def getContourIntensities(contour_points, grey_ROI):
		all_contour_inten = []
		for (x,y) in contour_points:
			intensity = grey_ROI.getPixel(x,y)[0]
			all_contour_inten.append(intensity)
		return all_contour_inten

	# Get the number of pixels in a contour - contour pair where pixels are neighbouring (assuming watershed algorithmn is used, searches for neighbour points 2 pixels away).
	@staticmethod
	def getNeighbourIndex(all_contour_points, intersecting_contours):
		neighbour_cross = {}
		for i in range(len(all_contour_points)):
			contour_list = listManipulator(all_contour_points[i])
			for j in range(i+1,len(all_contour_points)):
					if intersecting_contours[(i,j)] == False:
						continue
					matches = contour_list.watershedIntersection(all_contour_points[j])
					if (len(matches)) > 0:
						neighbour_cross[(i,j)] = len(matches)
		return neighbour_cross

	# Given a min_contact_scale value (e.g. 0.15), sets a minimum percentage (e.g. 15%) that the perimeters must be in contact to merit a pairing.
	# Only one of the contours must have 15% of its contour to merit a pairing (i.e. the pair depends on the smaller contour).
	@staticmethod
	def filterNeighbourIndex(cnts, neighbour_dict, min_contact_scale):
		filtered_list = []
		for key, value in neighbour_dict.items():
			contour_a = key[0]
			a_perim = arcLength(cnts.get(contour_a),True)
			contour_b = key[1]
			b_perim = arcLength(cnts.get(contour_b),True)
			if a_perim > 0 and b_perim > 0:
				if value > (min_contact_scale * a_perim) or value > (min_contact_scale * b_perim):
					filtered_list.append([contour_a, contour_b])
		return filtered_list

	@staticmethod
	def flood(ROI, cnts, integer):
		drawContours(ROI, cnts, integer, Scalar(255,255,255,255))
		# Floodfill the contours.
		mask = Mat()
		rect = Rect()
		M = CvMoments()
		cvMoments(CvMat(cnts.get(integer)), M)
		if M.m00() != 0:
			cx = int(M.m10()/M.m00())
			cy = int(M.m01()/M.m00())
			floodFill(ROI, mask, Point(cx,cy), Scalar(255,255,255,255), rect, Scalar(50,50,50,50), Scalar(50,50,50,50),4)
		return ROI

	@staticmethod
	def floodfillContours(cnts, roi_width, roi_height, cnts_to_draw):
		# Draw sliced contours on blank ROIs and re-create final image.
		final_ROI = Mat.zeros(roi_height, roi_width, 0).asMat()
		TL = MatrixUtils.getMatCoordinate(final_ROI, 0, 0)
		TR = MatrixUtils.getMatCoordinate(final_ROI, roi_width-1, 0)
		BL = MatrixUtils.getMatCoordinate(final_ROI, 0, roi_height-1)
		BR = MatrixUtils.getMatCoordinate(final_ROI, roi_width-1, roi_height-1)
		for t in cnts_to_draw:
			blank_ROI = Mat.zeros(roi_height, roi_width, 0).asMat()
			# Floodfill a single contour (t).
			test_fill = ContourUtils.flood(blank_ROI, cnts, t)
			# If hasn't filled the whole image perform same action on final_ROI (check if all four corners are white to exclude contour as unclosed).
			if not (test_fill.getByteBuffer().get(TL) == -1 and test_fill.getByteBuffer().get(TR) == -1 and test_fill.getByteBuffer().get(BL) == -1 and test_fill.getByteBuffer().get(BR) == -1):
				final_ROI = ContourUtils.flood(final_ROI, cnts, t)
		return final_ROI

	@staticmethod
	def contours2buckets(cnts, contact_index):
		# Preprocess contours to get a contour-contour intersection list.
		intersecting_contours = ContourMathUtils.computeIntersectingContours(cnts, 2)
		# For image get points from the contours.
		cnt_points = ContourUtils.getContoursPoints(cnts)
		# Get a dictionary of matches where each pixel in a contour is searched for neighbouring contour points.
		neighbour_dict = ContourUtils.getNeighbourIndex(cnt_points, intersecting_contours)
		# Filter and sort contours into groups based on amount of contact with neighbouring contours.
		filtered_list = ContourUtils.filterNeighbourIndex(cnts, neighbour_dict, contact_index)
		# Coerce the list of valid neighbours into buckets, which clusters matched contours into groups.
		buckets = ContourSorter.getBuckets(filtered_list)
		return cnt_points, buckets

	@staticmethod
	def filterBuckets(buckets, cnt_points, cnts, imgmat, min_prtcle_size, max_prtcle_size, separate_corners):
		roiwidth = imgmat.cols()
		roiheight = imgmat.rows()
		filtered_buckets = []
		cornered_bucket_cnts = []
		# Filter and draw bucket contours (colour-coded) on the original slice.
		for i in range(len(buckets)):
			a,b,c = ColourUtils.newcol(i)
			bucket_x = []
			bucket_y = []
			for j in buckets[i]:
				# Find the largest and smallest x and y values to find the height and width of the merged contours in the bucket.
				buck_cnt_points = cnt_points[j]
				for (x,y) in buck_cnt_points:
					bucket_x.append(x)
					bucket_y.append(y)
			# Save contours in buckets cut off by the edge of the ROI.
			if separate_corners:
				epsilon = 1
				if min(bucket_y) <= epsilon or min(bucket_x) <= epsilon or max(bucket_y) >= (roiheight-epsilon) or max(bucket_x) >= (roiwidth-epsilon):
					for c in buckets[i]:
						cornered_bucket_cnts.append(c)
				else:
					# Filter out buckets too large or too small.
					bucket_height = max(bucket_y) - min(bucket_y)
					bucket_width = max(bucket_x) - min(bucket_x)
					if bucket_height > min_prtcle_size and bucket_height < max_prtcle_size and bucket_width > min_prtcle_size and bucket_width < max_prtcle_size:
						filtered_buckets.append(buckets[i])
			else:
				# Filter out buckets too large or too small.
				bucket_height = max(bucket_y) - min(bucket_y)
				bucket_width = max(bucket_x) - min(bucket_x)
				if bucket_height > min_prtcle_size and bucket_height < max_prtcle_size and bucket_width > min_prtcle_size and bucket_width < max_prtcle_size:
					filtered_buckets.append(buckets[i])
		return cornered_bucket_cnts, filtered_buckets, imgmat

	@staticmethod
	def addContourLabel(contour, img_mat, start_int):
		M = CvMoments()
		cvMoments(CvMat(contour), M)
		if M.m00() != 0:
			cx = int(M.m10()/M.m00())
			cy = int(M.m01()/M.m00())
			putText(img_mat, str(start_int), Point(cx,cy),0,1,Scalar(0,0,0,255),2,0,False)
			start_int = start_int + 1
	
	@staticmethod
	def cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight, min_prtcle_size, max_prtcle_size):
		return_contours = MatVector()
		# Make clean slate for contour-joining.
		for i in range(len(filtered_buckets)):
			blank_mat = Mat.zeros(roiwidth, roiheight, 0).asMat()
			blank_mat = ContourUtils.floodfillContours(cnts, roiwidth, roiheight, filtered_buckets[i])
			# Merge watershed contours in the same buckets.
			closed_mat = Mat()
			morphologyEx(blank_mat, closed_mat, 3, getStructuringElement(2, Size(10,10)))			
			# Find full mitochondria contour.
			mito_contours = MatVector()
			findContours(closed_mat, mito_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE)
			if mito_contours.size() > 0:
				drawContours(final_mat, mito_contours, -1, Scalar(255,0,0,0))
				for c in range(mito_contours.size()):
					rect = minAreaRect(mito_contours.get(c))
					poi = Point2f(8)
					rect.points(poi)
					width = rect.size().width()
					height = rect.size().height()
					if width > min_prtcle_size and height > min_prtcle_size and width < max_prtcle_size and height < max_prtcle_size:
						return_contours.push_back(mito_contours.get(c))
		return final_mat, return_contours

	@staticmethod
	def getNumberThreads():
		return 4

	@staticmethod
	def isLastThread(threadIndex):
		return threadIndex == ContourUtils.getNumberThreads() - 1

	@staticmethod
	def SingleThread_processROIs(contour_list, ROI_matlist, ROI_range, min_prtcle_size, max_prtcle_size):
		sliced_cnt_vect = MatVector()
		firstfound_cnt_vect = MatVector()
		final_contours_mat = MatVector()
		for ind in range(len(contour_list)):
			cnts = contour_list[ind]
			# Get original image slice.
			ROImat = ROI_matlist[ind]
			roip = mat2ip.toImageProcessor(ROImat)
			roiwidth = ROImat.cols()
			roiheight = ROImat.rows()
		
			_ , ROIcolmat, _ = ImgUtils.convertImpColour2Img(roip, "GRAY2RGB")
			final_mat = ROIcolmat.clone()
		
			cnt_points, buckets = ContourUtils.contours2buckets(cnts, 0.15)
			cornered_bucket_cnts, filtered_buckets, _ = ContourUtils.filterBuckets(buckets, cnt_points, cnts, ROIcolmat, min_prtcle_size, max_prtcle_size, True)
			found_cnt_mat, ROI_contours = ContourUtils.cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight, min_prtcle_size, max_prtcle_size)
			firstfound_cnt_vect.push_back(found_cnt_mat)
		
			# Move ROI contours to position of image and append to final_contours_mat.
			offsetx = ROI_range[ind][0]
			offsety = ROI_range[ind][1]
			final_contours_mat = ContourUtils.moveContours(ROI_contours, offsetx, offsety, final_contours_mat)
			
			# Draw sliced contours on blank ROIs and re-create final image.
			sliced_mat = ContourUtils.floodfillContours(cnts, roiwidth, roiheight, cornered_bucket_cnts)
			sliced_cnt_vect.push_back(sliced_mat)
			
		return sliced_cnt_vect, firstfound_cnt_vect, final_contours_mat

	@staticmethod
	def processSingleROI(cnts, ROImat, ROI_dims, min_prtcle_size, max_prtcle_size):

		final_contours_mat = MatVector()

		# Get original image slice.
		roip = mat2ip.toImageProcessor(ROImat)
		roiwidth = ROImat.cols()
		roiheight = ROImat.rows()
	
		_ , ROIcolmat, _ = ImgUtils.convertImpColour2Img(roip, "GRAY2RGB")
		final_mat = ROIcolmat.clone()
		
		cnt_points, buckets = ContourUtils.contours2buckets(cnts, 0.15)
		_, filtered_buckets, _ = ContourUtils.filterBuckets(buckets, cnt_points, cnts, ROIcolmat, min_prtcle_size, max_prtcle_size, False)		
		_, ROI_contours = ContourUtils.cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight, min_prtcle_size, max_prtcle_size)
		
		# Move ROI contours to position of image and append to final_contours_mat.
		offsetx = ROI_dims[0]
		offsety = ROI_dims[1]
		final_contours_mat = ContourUtils.moveContours(ROI_contours, offsetx, offsety, final_contours_mat)

		return final_contours_mat
	
	@staticmethod
	def processROIsThread(contour_list, ROI_matlist, ROI_range, min_prtcle_size, max_prtcle_size, threadIndex, out_sliced_cnt_vect, out_firstfound_cnt_vect, out_final_contours_mat):
		num_per_thread = len(contour_list)//ContourUtils.getNumberThreads()
		start = threadIndex * num_per_thread
		end = start + num_per_thread
		if ContourUtils.isLastThread(threadIndex):
			end = len(contour_list)

		for ind in range(start, end):
			cnts = contour_list[ind]
			# Get original image slice.
			ROImat = ROI_matlist[ind]
			roip = mat2ip.toImageProcessor(ROImat)
			roiwidth = ROImat.cols()
			roiheight = ROImat.rows()
		
			_ , ROIcolmat, _ = ImgUtils.convertImpColour2Img(roip, "GRAY2RGB")
			final_mat = ROIcolmat.clone()
			
			cnt_points, buckets = ContourUtils.contours2buckets(cnts, 0.15)
			cornered_bucket_cnts, filtered_buckets, imgmat = ContourUtils.filterBuckets(buckets, cnt_points, cnts, ROIcolmat, min_prtcle_size, max_prtcle_size, True)
			found_cnt_mat, ROI_contours = ContourUtils.cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight, min_prtcle_size, max_prtcle_size)
			out_firstfound_cnt_vect.push_back(found_cnt_mat)

			del final_mat
			gc.collect()
			
			# Move ROI contours to position of image and append to final_contours_mat.
			offsetx = ROI_range[ind][0]
			offsety = ROI_range[ind][1]
			out_final_contours_mat = ContourUtils.moveContours(ROI_contours, offsetx, offsety, out_final_contours_mat)
			
			# Draw sliced contours on blank ROIs and re-create final image.
			sliced_mat = ContourUtils.floodfillContours(cnts, roiwidth, roiheight, cornered_bucket_cnts)
			out_sliced_cnt_vect.push_back(sliced_mat)

			
	# Fort each ROI the following processing occurs: filtering by contour size (remove single points), clustering (into buckets), filtering buckets (by size)
	@staticmethod
	def processROIs(contour_list, ROI_matlist, ROI_range, min_prtcle_size, max_prtcle_size):
		threads = []
		sliced_cnt_vect_arr = []
		firstfound_cnt_vect_arr = []
		final_contours_mat_arr = []
		for threadIndex in range(ContourUtils.getNumberThreads()):
			sliced_cnt_vect = MatVector()
			sliced_cnt_vect_arr.append(sliced_cnt_vect)
			firstfound_cnt_vect = MatVector()
			firstfound_cnt_vect_arr.append(firstfound_cnt_vect)
			final_contours_mat = MatVector()
			final_contours_mat_arr.append(final_contours_mat)
			thread = threading.Thread(target=ContourUtils.processROIsThread, args=(contour_list, ROI_matlist, ROI_range, min_prtcle_size, max_prtcle_size, threadIndex,sliced_cnt_vect,firstfound_cnt_vect,final_contours_mat))
			thread.start()
			threads.append(thread)
		for thread in threads:
			thread.join()

		sliced_cnt_vect = MatVector()
		for vec in sliced_cnt_vect_arr:
			for i in range(vec.size()):
				sliced_cnt_vect.push_back(vec.get(i))

		firstfound_cnt_vect = MatVector()
		for vec in firstfound_cnt_vect_arr:
			for i in range(vec.size()):
				firstfound_cnt_vect.push_back(vec.get(i))

		final_contours_mat = MatVector()
		for vec in final_contours_mat_arr:
			for i in range(vec.size()):
				final_contours_mat.push_back(vec.get(i))

		return sliced_cnt_vect, firstfound_cnt_vect, final_contours_mat
		
	
	@staticmethod
	def getManualROIs(manual_ROI_list, img):
		ROI_matlist = []
		for [x,y,w,h] in manual_ROI_list:
			rect = Roi(x, y, w, h)
			new = img.duplicate()
			new.setRoi(rect)
			ROI = new.crop()
			roi_mat = imp2mat.toMat(ROI.getProcessor())
			ROI_matlist.append(roi_mat)
		return ROI_matlist

	@staticmethod
	def getManualSingleROI(add_rois_elem, img):
		x,y,w,h = add_rois_elem
		rect = Roi(x, y, w, h)
		new = img.duplicate()
		new.setRoi(rect)
		ROI = new.crop()
		roi_mat = imp2mat.toMat(ROI.getProcessor())
		return roi_mat

	# Find contours in image and apply watershed, no filtering performed - using matvectors instead of stacks
	@staticmethod
	def getVariableContours(ROI_matlist, threshold_value):
		output_matlist = []
		contour_list = []
		# Isolate each slice.
		for i in range(len(ROI_matlist)):
			roi_mat = ROI_matlist[i]
			slce_ip = mat2ip.toImageProcessor(roi_mat)
			slce = ImagePlus("", slce_ip)
			slce_show = slce.duplicate()

			# Set threshold of image (and automatically converts to binary).
			slce.setProcessor("thresholded", slce_ip)
			if threshold_value == "auto":
				avg_intensity = PixelUtils.getAvgIntensity(slce)
				slce_ip.setThreshold(avg_intensity, 255, ImageProcessor.NO_LUT_UPDATE)
			else:
				slce_ip.setThreshold(threshold_value, 255, ImageProcessor.NO_LUT_UPDATE)
			IJ.run(slce, "Convert to Mask", "")
			# Despeckle.
			IJ.run(slce, "Despeckle", "")
			# Invert.
			slce.getProcessor().invert()

			# Watershed contours - default function.
			edm = EDM()
			edm.setup("watershed", None)
			edm.run(slce.getProcessor())
			
			# Locate contours on the image.
			colored_image, img_matrix, _ = ImgUtils.convertImgColour(slce, "GRAY2RGB")
			contours = MatVector()
			findContours(img_matrix, contours, RETR_LIST, CHAIN_APPROX_NONE)

			# Output contour list
			contour_list.append(contours)

			# Output image
			output_matlist.append(colored_image)

		return contour_list, output_matlist

	@staticmethod
	def getSingleVariableContours(roi_mat, threshold_value):
		slce_ip = mat2ip.toImageProcessor(roi_mat)
		slce = ImagePlus("", slce_ip)
		slce_show = slce.duplicate()

		# Set threshold of image (and automatically converts to binary).
		slce.setProcessor("thresholded", slce_ip)
		if threshold_value == "auto":
			avg_intensity = PixelUtils.getAvgIntensity(slce)
			slce_ip.setThreshold(avg_intensity, 255, ImageProcessor.NO_LUT_UPDATE)
		else:
			avg_intensity = None
			slce_ip.setThreshold(threshold_value, 255, ImageProcessor.NO_LUT_UPDATE)
		IJ.run(slce, "Convert to Mask", "")
		# Despeckle.
		IJ.run(slce, "Despeckle", "")
		# Invert.
		slce.getProcessor().invert()

		# Watershed contours - default function.
		edm = EDM()
		edm.setup("watershed", None)
		edm.run(slce.getProcessor())
		
		# Locate contours on the image.
		output_ROI, img_matrix, _ = ImgUtils.convertImgColour(slce, "GRAY2RGB")
		contours = MatVector()
		findContours(img_matrix, contours, RETR_LIST, CHAIN_APPROX_NONE)

		return contours, avg_intensity

class ContourFindUtils:
	@staticmethod
	def findContoursFromImage(image, threshold):
		# Set threshold of image (and automatically converts to binary).
		ip = image.getProcessor()
		ip.setThreshold(threshold, 255, ImageProcessor.NO_LUT_UPDATE)
		IJ.run(image, "Convert to Mask", "")
		# Despeckle.
		IJ.run(image, "Despeckle", "")
		# Invert.
		image.getProcessor().invert()
		# Find contours.
		colored_image, img_matrix, _ = ImgUtils.convertImgColour(image, "GRAY2RGB")
		contours = MatVector()
		findContours(img_matrix, contours, RETR_LIST, CHAIN_APPROX_NONE)
		return contours

	@staticmethod
	def filterContoursBySize(contours, max_len, min_len):
		retained_contours = MatVector()
		cnt_points = ContourUtils.getContoursPoints(contours)
		for i in range(len(cnt_points)):
			cntpoints = cnt_points[i]
			xpoints = []
			ypoints = []
			for (x,y) in cntpoints:
				xpoints.append(x)
				ypoints.append(y)
			height = max(ypoints) - min(ypoints)
			width = max(xpoints) - min(xpoints)
			if height < max_len and height > min_len and width < max_len and width > min_len:
				retained_contours.push_back(contours.get(i))
		return retained_contours

	@staticmethod
	def getRotatedRectangle(contour):
		rect = minAreaRect(contour)
		poi = Point2f(8)
		rect.points(poi)
		return rect

	@staticmethod
	def contourStats(contour):
		rect = ContourFindUtils.getRotatedRectangle(contour)
		# Get areas of mitos.
		area = contourArea(contour)
		# Get perimeter of mitos.
		perim = arcLength(contour, True)
		# Calculate circularities.
		circularity = (4 * math.pi * area)/(perim * perim)
		width = min(rect.size().height(), rect.size().width())
		length = max(rect.size().height(), rect.size().width())
		return area, perim, circularity, width, length

	# Filter by determined parameters.
	@staticmethod
	def filterContoursbyManualOptions(mito_contours, imgd, filtering_dict):
		_, img_mat, _ = ImgUtils.convertImgColour(imgd, "GRAY2RGB")
		filtered_contours = MatVector()
		if not all([filtering_dict['maxinten_filter'][0], filtering_dict['mininten_filter'][0]]):
			all_contour_points = ContourUtils.getInteriorContoursPoints(mito_contours, imp2mat.toMat(imgd.getProcessor()))
				
		for i in range(mito_contours.size()):
			if not all([filtering_dict['circularity_filter'][0], filtering_dict['maxlen_filter'][0], filtering_dict['minlen_filter'][0], filtering_dict['ratio_filter'][0], filtering_dict['minarea_filter'][0], filtering_dict['maxarea_filter'][0]]):
				area, perim, circularity, width, length = ContourFindUtils.contourStats(mito_contours.get(i))

				if length > 0:
					ratio = float(width)/float(length)
				else:
					continue # skip contours with no length.
			else:
				ratio = length = circularity = None
					
			if not all([filtering_dict['maxinten_filter'][0], filtering_dict['mininten_filter'][0]]):
				intensities = ContourUtils.getContourIntensities(all_contour_points[i], imgd)
				contour_intensity = sum(intensities)/len(intensities)
			else:
				contour_intensity = None
			if filterbyItem(filtering_dict['circularity_filter'], circularity, "larger than") \
				and filterbyItem(filtering_dict['minarea_filter'], circularity, "larger than") \
				and filterbyItem(filtering_dict['maxarea_filter'], circularity, "smaller than") \
				and filterbyItem(filtering_dict['maxlen_filter'], length, "smaller than") \
				and filterbyItem(filtering_dict['minlen_filter'], length, "larger than") \
				and filterbyItem(filtering_dict['ratio_filter'], ratio, "larger than") \
				and filterbyItem(filtering_dict['maxinten_filter'], contour_intensity, "smaller than") \
				and filterbyItem(filtering_dict['mininten_filter'], contour_intensity, "larger than"):
				filtered_contours.push_back(mito_contours.get(i))
		return filtered_contours

	@staticmethod
	def showRotRect(contours, img_mat):
		for i in range(contours.size()):
			rect = ContourFindUtils.getRotatedRectangle(contours.get(i))
			BL = Point(int(poi.get(0)),int(poi.get(1)))
			TL = Point(int(poi.get(2)),int(poi.get(3)))
			TR = Point(int(poi.get(4)),int(poi.get(5)))
			BR = Point(int(poi.get(6)),int(poi.get(7)))
			line(img_mat,TL,TR,Scalar(255,255,0,0),3,0,0)
			line(img_mat,TL,BL,Scalar(255,255,0,0),3,0,0)
			line(img_mat,BR,TR,Scalar(255,255,0,0),3,0,0)
			line(img_mat,BR,BL,Scalar(255,255,0,0),3,0,0)
		ImagePlus("Mitochondria blob", mat2ip.toImageProcessor(img_mat)).show()


class guiManager:

	@staticmethod
	def getLikely(inputtype):
		if inputtype == "tiff" or inputtype == "tif":
			out = "tiff"
		elif inputtype == "jpeg" or inputtype == "jpg":
			out = "jpeg"
		elif inputtype == "png":
			out = "png"
		else:
			out = "tiff"
		return out
		
	@staticmethod
	def keepFilterChanges():
		gui = NonBlockingGenericDialog("Keep these changes")
		gui.setCancelLabel("No - return to filtering")
		gui.showDialog()
		if gui.wasOKed():
			return "Keep"
		if gui.wasCanceled():
			return "Return"

	@staticmethod
	def autoManual():
		gui = NonBlockingGenericDialog("Expected size of mitochondria")
		gui.addRadioButtonGroup("Would you like to run automatic processing?", ["Yes","Skip to manual"], 2, 1, "Yes")
		gui.showDialog()
		if gui.wasCanceled():
			sys.exit("Starting menu was cancelled, ending program")
		if gui.wasOKed():
			choice = gui.getNextRadioButton()
		if choice == "Yes":
			return True
		if choice == "Skip to manual":
			return False

	@staticmethod
	def ManualOptionsMenu(question_text):
		gui = NonBlockingGenericDialog("Continue")
		gui.addChoice(question_text, ["Yes - Deletion/Detection/Drawing mode", "Yes - Fine-tuning mode", "Find lipid droplets", "No, finish program"], "Yes - Deletion/Semi-automatic Mode")
		gui.showDialog()
		if gui.wasOKed():
			choice = gui.getNextChoice()
			return choice
		else:
			return "Cancelled"

	@staticmethod
	def ManualCheckExit(return_command):
		check_decision = NonBlockingGenericDialog("Are you sure")
		check_decision.addRadioButtonGroup("You selected cancel, are you sure you don't want to finish the program?", ["Finish program","End without finishing",return_command], 3, 1, "Finish program")
		check_decision.showDialog()
		if check_decision.wasOKed():
			final_decision = check_decision.getNextRadioButton()
			if final_decision == "Finish program":
				return None
			if final_decision == return_command:
				return "re-run menu"
		if check_decision.wasCanceled():
			sys.exit("Menus were cancelled twice in a row, automatically ending program")
		else:
			sys.exit("Menu was cancelled, ending program")
		
	@staticmethod		
	def manualOptions(question_text):
		gui_out = guiManager.ManualOptionsMenu(question_text)
		# Safety loop in case menu was cancelled by accident.
		if gui_out == "Cancelled":
			check_out = guiManager.ManualCheckExit("Return to manual menu")
			if check_out == "re-run menu":
				gui_check = guiManager.ManualOptionsMenu(question_text)
				if gui_check == "Cancelled":
					sys.exit("Menus were cancelled twice in a row, automatically ending program")
				else:
					return gui_check
			if check_out == None:
				return "Finish program"
		else:
			return gui_out

	@staticmethod
	def FilteringOptionsMenu(question_text):
		gui = NonBlockingGenericDialog("Filtering")
		gui.addNumericField("Ratio filtering (x:1) =", 0.5)
		gui.addCheckbox("No ratio filtering ", False)
		gui.addNumericField("Circularity filtering (x/1) =", 0.5)
		gui.addCheckbox("No circularity filtering ", False)
		gui.addNumericField("Maximum area  =", 100)
		gui.addCheckbox("No maximum area filtering ", True)
		gui.addNumericField("Minimum area =", 100)
		gui.addCheckbox("No minimum area filtering ", True)
		gui.addNumericField("Maximum length (px) =", 100)
		gui.addCheckbox("No maximum length filtering ", True)
		gui.addNumericField("Minimum length (px) =", 100)
		gui.addCheckbox("No minimum length filtering ", True)
		gui.addNumericField("Maximum intensity (x/255) =", 100)
		gui.addCheckbox("No maximum intensity filtering ", True)
		gui.addNumericField("Minimum intensity (x/255) =", 100)
		gui.addCheckbox("No minimum intensity filtering ", True)
		gui.showDialog()
		if gui.wasOKed():
			filtering_dict = {}
			filtering_dict["ratio_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			filtering_dict["circularity_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			filtering_dict["maxarea_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			filtering_dict["minarea_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			filtering_dict["maxlen_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			filtering_dict["minlen_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			filtering_dict["maxinten_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			filtering_dict["mininten_filter"] = [gui.getNextBoolean(), gui.getNextNumber()]
			return filtering_dict
		else:
			return "Cancelled"

	@staticmethod		
	def filteringOptions(question_text):
		gui_out = guiManager.FilteringOptionsMenu(question_text)
		# Safety loop in case menu was cancelled by accident.
		if gui_out == "Cancelled":
			check_out = guiManager.ManualCheckExit("Return to filtering menu")
			if check_out == "re-run menu":
				gui_check = guiManager.FilteringOptionsMenu(question_text)
				if gui_check == "Cancelled":
					sys.exit("Menus were cancelled twice in a row, automatically ending program")
				else:
					return gui_check
			if check_out == None:
				return "Finish program"
		else:
			return gui_out

	@staticmethod
	def particleOptionMenu(default_max, default_min, default_threshold_factor, error):
		gui = NonBlockingGenericDialog("Expected size of mitochondria")
		if error != "":
			gui.addMessage(str(error))
		gui.addNumericField("Maximum (pixels): ", default_max)
		gui.addNumericField("Minimum (pixels): ", default_min)
		gui.addCheckbox("Set manual threshold", False)
		gui.showDialog()
		if gui.wasOKed():
			max_prtcle_size = gui.getNextNumber()
			min_prtcle_size = gui.getNextNumber()
			if gui.getNextBoolean():
				t_gui = NonBlockingGenericDialog("Set Manual Threshold Value")
				t_gui.addNumericField("Threshold (0-255): ", 150)
				t_gui.showDialog()
				if t_gui.wasOKed():
					threshold_factor = t_gui.getNextNumber()
				else:
					threshold_factor = "auto"
			else:
				threshold_factor = "auto"
			return max_prtcle_size, min_prtcle_size, threshold_factor
		else:
			return "Cancelled"

	@staticmethod
	def particleCheckExit():
		check_decision = NonBlockingGenericDialog("Are you sure")
		check_decision.addRadioButtonGroup("You selected cancel, are you sure you don't want to continue?", ["Cancel","Skip to manual editing","Return to automatic menu"], 3, 1, "Cancel")
		check_decision.showDialog()
		if check_decision.wasOKed():
			final_decision = check_decision.getNextRadioButton()
			if final_decision == "Skip to manual editing":
				return None
			if final_decision == "Return to automatic menu":
				return "re-run menu"
		if check_decision.wasCanceled():
			sys.exit("Menus were cancelled twice in a row, automatically ending program")
		else:
			sys.exit("Menu was cancelled, ending program")
			

	@staticmethod
	def particleOptions(default_max, default_min, default_threshold_factor, error):
		gui_out = guiManager.particleOptionMenu(default_max, default_min, default_threshold_factor, error)
		if gui_out == "Cancelled":
			check_out = guiManager.particleCheckExit()
			if check_out == "re-run menu":
				gui_out = guiManager.particleOptionMenu(default_max, default_min, default_threshold_factor, error)
				if gui_out == "Cancelled":
					sys.exit("Menus were cancelled twice in a row, automatically ending program")
				else:
					max_prtcle_size, min_prtcle_size, threshold_factor = gui_out
					return max_prtcle_size, min_prtcle_size, threshold_factor
			else:
				return check_out # Should only be None
		else:
			max_prtcle_size, min_prtcle_size, threshold_factor = gui_out
			return max_prtcle_size, min_prtcle_size, threshold_factor


	@staticmethod	
	def lipidOptions(default_max, default_min, default_threshold, error):
		gui = NonBlockingGenericDialog("Expected size of lipid droplets")
		if error != "":
			gui.addMessage(str(error))
		gui.addNumericField("Maximum (pixels): ", default_max)
		gui.addNumericField("Minimum (pixels): ", default_min)
		gui.addNumericField("Lipid minimum intensity (max. 255): ", default_threshold)
		gui.showDialog()
		if gui.wasOKed():
			max_len = gui.getNextNumber()
			min_len = gui.getNextNumber()
			lipid_threshold = gui.getNextNumber()
			return max_len, min_len, lipid_threshold
		else:
			cont = guiManager.runEditCheck("Any more edits to make?")

	@staticmethod
	def parsePointROI(roi_elem):
		x = roi_elem.getXBase()
		y = roi_elem.getYBase()
		return (int(x),int(y))
		
	@staticmethod
	def parseRectangleROI(roi_elem):
		x = roi_elem.getXBase()
		y = roi_elem.getYBase()
		w = roi_elem.getFloatWidth()
		h = roi_elem.getFloatHeight()
		return (int(x),int(y), int(w), int(h))
		
	@staticmethod
	def parseCompositeROI(roi_elem):
		manual_contour = []
		poly = roi_elem.getPolygon()
		for j in range(poly.npoints):
			x = poly.xpoints[j]
			y = poly.ypoints[j]
			manual_contour.append(ContourPoint(int(x),int(y)))
		return manual_contour
		
	
	@staticmethod		
	def getManualInput():
		RM = RoiManager()
		rm = RM.getRoiManager() 
		rm.runCommand("reset")
		rm.runCommand("Show All without labels")
		waiting = WaitForUserDialog("Manual Editing Activated", "Use the 'Point' tool to remove unwanted contours and the 'Rectangle' tool to select new ROI's. \nWhen finished, click ok to process ROI's and points.")
		waiting.show()
		roi_array = rm.getRoisAsArray()
		num_rois = rm.getCount()
		
		rm_points = []
		add_rois = []
		manual_contours = []
		for roi_elem in roi_array:
			if roi_elem.getTypeAsString() == "Point":
				rm_points.append(guiManager.parsePointROI(roi_elem))
			if roi_elem.getTypeAsString() == "Rectangle":
				add_rois.append(guiManager.parseRectangleROI(roi_elem))
			if roi_elem.getTypeAsString() == "Composite" or roi_elem.getTypeAsString() == "Freehand":
				manual_contours.append(guiManager.parseCompositeROI(roi_elem))

		return rm_points, add_rois, manual_contours

	@staticmethod		
	def getAlteredContour(cntpoints, imgp_draw):

		RM = RoiManager()
		rm = RM.getRoiManager() 
		rm.runCommand("reset")
		rm.runCommand("Show All without labels")
		
		for a in cntpoints:
			x_points, y_points = RangeUtils.splitPointArrays(a)
			contour_poly = PolygonRoi(x_points, y_points, 3) # FREEROI = 3
			rm.addRoi(contour_poly)
		
#		ImagePlus("custom edits", imgp_draw).show()
		rm.runCommand("show all")

		waiting = WaitForUserDialog("Fine-tuning Mode Activated", "Select the contour you with to edit from the ROI manager menu, it will turn blue. \nNext use the 'Selection Brush Tool' (under the oval menu) to alter the contour you have selected (note: there will be a line remaining where the original contour presided). \nWhen finished, click 'update' in the ROI manager menu, the old line will disappear. Now repeat for other chosen contours. \nPress ok to finish.")
		waiting.show()

		rm.runCommand("update")
		roi_array = rm.getRoisAsArray()
		altered_contours = []
		for roi_elem in roi_array:
			altered_contours.append(guiManager.parseCompositeROI(roi_elem))
		
		return altered_contours

	