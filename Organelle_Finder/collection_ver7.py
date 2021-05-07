from ij	import IJ, ImagePlus
from ij.gui import Line, Plot
import math
from ij.plugin.frame import RoiManager
from ij.process import ImageProcessor
from ijopencv.ij import ImagePlusMatConverter as imp2mat
from org.bytedeco.javacpp.opencv_core  import Point2f, vconcat, hconcat, cvmGet, subtract, Size, Point2f, Rect, Mat, MatVector, GpuMatVector, KeyPointVector, Scalar, CvMat, vconcat,ACCESS_READ, Point
from org.bytedeco.javacpp.opencv_imgproc  import pointPolygonTest, line, minAreaRect, floodFill, arcLength, CHAIN_APPROX_NONE,morphologyDefaultBorderValue, morphologyEx, watershed, connectedComponents, threshold, distanceTransform, getStructuringElement, dilate, findContours, CvMoments, cvMoments, contourArea, drawContours, putText, boundingRect, cvtColor, RETR_FLOODFILL, RETR_EXTERNAL, RETR_LIST, CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, COLOR_GRAY2RGB,COLOR_RGB2GRAY
from ij.ImageStack import create, addSlice, deleteSlice, size, getSliceLabel
from ij.gui import GenericDialog, Roi, Line, Plot, PointRoi, NewImage, MessageDialog, WaitForUserDialog, PolygonRoi
from ijopencv.opencv import MatImagePlusConverter as mat2ip
from ij.plugin import MontageMaker
from ij.plugin.filter import GaussianBlur, EDM
import threading

def median(lis):
	lis.sort()
	length = len(lis)
	if length % 2 == 0:
	    return ((lis[length/2 - 1] + lis[length/2])/2) # Even 
	else:
	    return lis[length/2] # Odd

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


def getAngleWithCenter(center, p):
	x = p.x - center.x
	y = p.y - center.y
	angle = math.atan2(y,x)
	if angle <= 0.0:
		angle = 2 * math.pi + angle
	return angle
	
def comparePoints(a, b):
	angleA = getAngleWithCenter(ContourPoint(0,0), a)
	angleB = getAngleWithCenter(ContourPoint(0,0), b)
	if angleA < angleB:
		return -1
	dA = ContourPoint(0,0).getDistanceFromPoint(a)
	dB = ContourPoint(0,0).getDistanceFromPoint(b)
	if (angleA == angleB) and (dA < dB):
		return -1
	return 1

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
		self.points = sorted(self.points, cmp=lambda a,b : comparePoints(a,b))
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


class AreaUtils:
	@staticmethod
	def getBoxBounds(x0,y0,r):
		x = x0 - r
		y = y0 - r
		w = h = 2 * r
		return [x,y,w,h]

	@staticmethod
	def getDonutPoints(x0, y0, r_min, r_max, tolerance = 0.0001):
		box = AreaUtils.getBoxBounds(x0,y0, r_max)
		squared_radius = r_max * r_max
		squared_hole_radius = r_min * r_min
		points = []
		for x in range(box[0], box[0] + box[2] + 1):
			x_transformed = x - x0
			for y in range(box[1], box[1] + box[3] + 1):
				y_transformed = y - y0
				dist = x_transformed*x_transformed + y_transformed*y_transformed
				if (dist - squared_radius) < tolerance and (dist - squared_hole_radius) > -tolerance:
					points.append((x,y))
		return points

	@staticmethod
	def getRingPoints(x0, y0, r, tolerance = 0.0001):
		return AreaUtils.getDonutPoints(x0, y0, r, r, tolerance)

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

class Colour:
	@staticmethod
	def makeMatColour(img):
		# Convert ImagePlus to Mat object.
		img_matrix = imp2mat.toMat(img.getProcessor())
		colored_image = Mat()
		cvtColor(img_matrix, colored_image, COLOR_GRAY2RGB)
		return colored_image, img_matrix

	@staticmethod
	def convertImgColour(img, conversion):
		if conversion == "GRAY2RGB":
			img_matrix = imp2mat.toMat(img.getProcessor())
			out = Mat()
			cvtColor(img_matrix, out, COLOR_GRAY2RGB)
			final_image  = mat2ip.toImageProcessor(out)
			new_img = ImagePlus("final image", final_image)
			return new_img
		if conversion == "GRAY2BGR":
			img_matrix = imp2mat.toMat(img.getProcessor())
			out = Mat()
			cvtColor(img_matrix, out, COLOR_GRAY2BGR)
			final_image  = mat2ip.toImageProcessor(out)
			new_img = ImagePlus("final image", final_image)
			return new_img
		if conversion == "BGR2GRAY":
			img_matrix = imp2mat.toMat(img.getProcessor())
			out = Mat()
			cvtColor(img_matrix, out, COLOR_BGR2GRAY)
			final_image  = mat2ip.toImageProcessor(out)
			new_img = ImagePlus("final image", final_image)
			return new_img
		if conversion == "RGB2GRAY":
			img_matrix = imp2mat.toMat(img.getProcessor())
			out = Mat()
			cvtColor(img_matrix, out, COLOR_RGB2GRAY)
			final_image  = mat2ip.toImageProcessor(out)
			new_img = ImagePlus("final image", final_image)
			return new_img

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

	# Get list of points that delineate ROIs in an image (some overlap may occur at boundary ROIs).
	def getROIRange(self):
		origins = []
		x_segments = RangeUtils.getRange(0, self.roi_width, self.img_width)
		y_segments = RangeUtils.getRange(0, self.roi_height, self.img_height)
		for x in x_segments:
			for y in y_segments:
				origins.append((x,y))
		return origins, (len(x_segments), len(y_segments))

	def getVariableROIRange(self):
		origins = []
		x_segments = RangeUtils.getVariableRange(0, self.roi_width, self.img_width)
		y_segments = RangeUtils.getVariableRange(0, self.roi_height, self.img_height)
		for x in x_segments:
			for y in y_segments:
				origins.append((x,y))
		return origins, (len(x_segments), len(y_segments))

	# Splits image into ROIs using the origin values given in ROI_range.
	def getMultipleROI(self):
		ROI_range, stack_dimensions = self.getROIRange()
#		ROI_range = [(728,1108), (1125,450), (1125,225), (1125,0), (900,0), (675,1128), (675,900)] 
		ROI_stack = create(self.roi_width, self.roi_height, 0, 8)
		for (x,y) in ROI_range:
			rect = Roi(x, y, self.roi_width, self.roi_height)
			new = self.img.duplicate()
			new.setRoi(rect)
			ROI = new.crop()
			slice_label = "[" + str(x) + "," + str(y) + "]"
			ROI_stack.addSlice(slice_label, ROI.getProcessor())
		ROI_stack.deleteSlice(1)
		imp = ImagePlus('ROI_stack', ROI_stack)
		imp.setStack(ROI_stack)
		return ROI_stack, stack_dimensions
	
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


	# Returns ROI given an origin (x,y) point, as a stack that can be used for getContours on a single ROI.
	def getROI(self, point):
		x, y = point
		ROI_stack = create(self.roi_width, self.roi_height, 0, 8)
		rect = Roi(x, y, self.roi_width, self.roi_height)
		new = self.img.duplicate()
		new.setRoi(rect)
		ROI = new.crop()
		slice_label = "[" + str(x) + "," + str(y) + "]"
		ROI_stack.addSlice(slice_label, ROI.getProcessor())
		ROI_stack.deleteSlice(1)
		imp = ImagePlus('ROI_stack', ROI_stack)
		imp.setStack(ROI_stack)
		return ROI_stack

	# Find contours in image and apply watershed, no filtering performed.
	def getContours(self, min_prtcle_area, ROI_stack, format):
		sample_stack = create(self.roi_width, self.roi_height, 0, 24)
		output_stack = create(self.roi_width, self.roi_height, 0, 24)
		contour_list = []
		roi_avg_intensities = []
		# Isolate each slice.
		for i in range(1, size(ROI_stack) + 1):
			slce_ip = ROI_stack.getProcessor(i)
			slce_label = ROI_stack.getSliceLabel(i)
			slce = ImagePlus(str(slce_label), slce_ip)
			slce_show = slce.duplicate()


			# Set threshold of image (and automatically converts to binary).
			slce.setProcessor("thresholded", slce_ip)
			avg_intensity = AreaUtils.getAvgIntensity(slce)
			roi_avg_intensities.append(avg_intensity)
			slce_ip.setThreshold(avg_intensity-5, 255, ImageProcessor.NO_LUT_UPDATE)
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
			colored_image, img_matrix = Colour.makeMatColour(slce)
			contours = MatVector()
			findContours(img_matrix, contours, RETR_LIST, CHAIN_APPROX_NONE)
#			drawContours(colored_image, contours, -1, Scalar(0,255,0,0))

			contour_list.append(contours)

			# Output image
			contours_ip  = mat2ip.toImageProcessor(colored_image)
			output_stack.addSlice(slce_label, contours_ip)
	
			if format == "collage":
				# Add sucessful slices to stack.
				sample_stack.addSlice(slce_label, slce_show.flatten().getProcessor())
				sample_stack.addSlice(slce_label, contours_ip)
			if format == "simple":
				contour_stack = create(self.roi_width, self.roi_height, 0, 24)
				contour_stack.addSlice(slce_label, slce_show.flatten().getProcessor())
				contour_stack.addSlice(slce_label, contours_ip)
				contour_stack.deleteSlice(1)
				couple = ImagePlus("Contour comparison", contour_stack)
				couple.setStack(contour_stack)
				newm = MontageMaker()
				newm.makeMontage(couple, couple.getStackSize(), 1, 1.0, 1, couple.getStackSize(), 1, 1, True)

		if format == "collage":
			# Show montage of sucessful images.
			sample_stack.deleteSlice(1)
			couple = ImagePlus("Contour comparison", sample_stack)
			couple.setStack(sample_stack)
			newm = MontageMaker()
			newm.makeMontage(couple, 4, couple.getStackSize()/4, 1.0, 1, couple.getStackSize(), 1, 1, True)
		
		output_stack.deleteSlice(1)
		return contour_list, output_stack #, roi_avg_intensities



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
	# Get tuples for start and end locations for the intervals given by width (in range of start to maxim).
	# However, also maintain the width of the final interval instead of shortening.
	@staticmethod
	def getRange(start, width, maxim):
		collect = []
		i = start
		collect.append(i)
		while i <= maxim:
			i = i + width
			if i > maxim - width:
				i = maxim - width
				collect.append(i)
				break
			collect.append(i)
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

	# Finds the theta corresponding to an angle needed to find an adjacent pixel using length x.
	def getTheta(self):
		y = 1 # pixels have a size of 1 pixel.
		x = self.solvePythagorean(y)
		theta = math.atan(y/x)
		return theta

	# Convert polar coordinates to cartesian coordinates given radius (r) and ange (theta).
	def polartoCartesian(self, theta):
		x = self.r * math.cos(theta) + self.ox
		y = self.r * math.sin(theta) + self.oy
		return((round(x, 0),round(y, 0)))

	# Using polar coordinates ranges, obtain the points of a ring in cartesian coordinates.
	def getCircleEdgePoints(self):
		theta = self.getTheta()
		iterations = int(2*math.pi / theta) + 2
		cirlcepoints = []
		for i in range(1, iterations, 1):
			coord = self.polartoCartesian(theta * i)
			cirlcepoints.append(coord)
		return cirlcepoints

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

class Profile:
	def __init__(self, x1, y1, x2, y2, img, min_slope, min_slope_width, min_slope_height, max_size, min_intensity, showplot):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.img = img
		self.linewidth = x2 - x1
		self.min_slope = min_slope
		self.min_slope_width = min_slope_width
		self.min_slope_height = min_slope_height
		self.max_size = max_size
		self.min_intensity = min_intensity
		self.showplot = showplot

	def getLineProfile(self):
		# Get pixel values (as integers) across a line.
		line = Line(self.x1, self.y1, self.x2, self.y2, self.img).getPixels()

		# Collect a list of indexes along the line where peaks/troughs are found.
		line_pixelclass = pixelInensities(line)
		indexes = line_pixelclass.getPeaksandTroughs()
		
		# Get bounds of steep inflections and slope direction.
		bounds = pixelBounds(line, self.min_slope, self.min_slope_width, self.min_slope_height, self.max_size, self.min_intensity)
		peak_trough_bounds = bounds.getSlopePairs(indexes)
		
		# Get points to plot the line profile and slope boundaries.
		y_bound_points = []
		flat_bound_array = []
		midpoints = []
		riseorfall = []
		for a in peak_trough_bounds:
			flat_bound_array.append(a[0])
			flat_bound_array.append(a[1])
			y_bound_points.append(line[a[0]])
			y_bound_points.append(line[a[1]])
			if line[a[1]] > line[a[0]]:
				riseorfall.append('rise')
			else:
				riseorfall.append('fall')
			midpoints.append(int((a[0] + a[1]) / 2))
		
		# Plot the line profile and slope boundaries.
		name = str(self.x1) +", "+ str(self.y1) +", "+ str(self.x2) +", "+ str(self.y2) + " line intensity plot"
		plt = Plot(name, "Line axis", "Intensity")
		plt.setLimits(0.0, self.linewidth, 0.0, 255.0)
		for i in range(len(line)-1):
			plt.drawLine(i, line[i], i+1, line[i+1])
		plt.addPoints(flat_bound_array, y_bound_points, 8)
		if self.showplot:
			plt.show()
		
		return midpoints, riseorfall

class mitoIdentifier:
	def __init__(self, points, riseorfall, min_prtcle_size, max_prtcle_size):
		self.points = points
		self.riseorfall = riseorfall
		self.min_prtcle_size = min_prtcle_size
		self.max_prtcle_size = max_prtcle_size

	def identifyCandidates(self):
		point_class = listManipulator(self.points)
		risefall_class = listManipulator(self.riseorfall)
		filtered_points = []
		for i in range(len(self.points)):
			# Filter so only searching points within a valid distance to correspond to a particle.
			low_range = range(self.points[i] - self.max_prtcle_size, self.points[i] - self.min_prtcle_size, 1)
			high_range = range(self.points[i] + self.min_prtcle_size, self.points[i] + self.max_prtcle_size, 1)
			
			# High and low candidates are the integers corresponding to the points and riseorfall values that meet the above criteria.
			low_candidates = point_class.intersectIndex(low_range)
			high_candidates = point_class.intersectIndex(high_range)
	
			# Only consider points that match the fall and rise pattern of the mitochondira.
			if self.riseorfall[i] == 'fall' and high_candidates != []:
				if risefall_class.checkList('rise', high_candidates): # high candidates should be rise
					filtered_points.append(i)
			if self.riseorfall[i] == 'rise' and low_candidates != []:
				if risefall_class.checkList('fall', low_candidates): # low candidates should be fall
					filtered_points.append(i)
		return filtered_points


class ContourUtils:

	@staticmethod
	def getLipids(image, threshold, max_len, min_len):
		# Set threshold of image (and automatically converts to binary).
		ip = image.getProcessor()
		ip.setThreshold(threshold, 255, ImageProcessor.NO_LUT_UPDATE)
		IJ.run(image, "Convert to Mask", "")
		# Despeckle.
		IJ.run(image, "Despeckle", "")
		# Invert.
		image.getProcessor().invert()
		# Find contours.
		colored_image, img_matrix = Colour.makeMatColour(image)
		contours = MatVector()
		findContours(img_matrix, contours, RETR_LIST, CHAIN_APPROX_NONE)

		# Filter for size.
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
	def largestContourIndex(contours_matvec):
		contour_sizes = {}
		for contour_idx in range(contours_matvec.size()):
			area = contourArea(contours_matvec.get(contour_idx))
			contour_sizes[contour_idx] = area
		max_area = max(contour_sizes.values())
		max_contour_index = contour_sizes.keys()[contour_sizes.values().index(max_area)]
		return max_contour_index

		
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
		for contour_int in range(contours.size()):
			contour_points = []
			for x in range(img_mat.cols()):
				for y in range(img_mat.rows()):
					point = Point2f(float(x), float(y))
					result = pointPolygonTest(contours.get(contour_int), point, False)
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
	def getNeighbourIndex(all_contour_points):
		neighbour_cross = {}
		for i in range(len(all_contour_points)):
			contour_list = listManipulator(all_contour_points[i])
			for j in range(i+1,len(all_contour_points)):
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
		# For image get points from the contours.
		cnt_points = ContourUtils.getContoursPoints(cnts)
		# Get a dictionary of matches where each pixel in a contour is searched for neighbouring contour points.
		neighbour_dict = ContourUtils.getNeighbourIndex(cnt_points)
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
			a,b,c = newcol(i)
			bucket_x = []
			bucket_y = []
			for j in buckets[i]:
				# Find the largest and smallest x and y values to find the height and width of the merged contours in the bucket.
				buck_cnt_points = cnt_points[j]
				for (x,y) in buck_cnt_points:
					bucket_x.append(x)
					bucket_y.append(y)
		
				drawContours(imgmat, cnts, j, Scalar(a,b,c,0))
				M = CvMoments()
				cvMoments(CvMat(cnts.get(j)), M)
#				if M.m00() != 0:
#					cx = int(M.m10()/M.m00())
#					cy = int(M.m01()/M.m00())
#					putText(imgmat, str(j), Point(cx-10,cy),0,0.3,Scalar(0,0,0,255),1,0,False)
			
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
#		ImagePlus("manually filtered final", mat2ip.toImageProcessor(imgmat)).show()

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
	def cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight):
		return_contours = MatVector()
#		final_mat = Mat.zeros(roiwidth, roiheight, 0).asMat() # option
		# Make clean slate for contour-joining.
		for i in range(len(filtered_buckets)):
			blank_mat = Mat.zeros(roiwidth, roiheight, 0).asMat()
			blank_mat = ContourUtils.floodfillContours(cnts, roiwidth, roiheight, filtered_buckets[i])
#			ImagePlus("Mitochondria, watershed", mat2ip.toImageProcessor(blank_mat)).show()

			# Merge watershed contours in the same buckets.
			closed_mat = Mat()
			morphologyEx(blank_mat, closed_mat, 3, getStructuringElement(2, Size(10,10)))
#			ImagePlus("Mitochondria blob", mat2ip.toImageProcessor(closed_mat)).show()
			
			# Find full mitochondria contour.
			mito_contours = MatVector()
			findContours(closed_mat, mito_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE)
			if mito_contours.size() > 0:
				drawContours(final_mat, mito_contours, -1, Scalar(255,0,0,0))
				for c in range(mito_contours.size()):
					return_contours.push_back(mito_contours.get(c))
#				ImagePlus("Final mitochondria contour", mat2ip.toImageProcessor(final_mat)).show()
		return final_mat, return_contours

	@staticmethod
	# Get minimum rectangle and filter for width.
	def filterRotRectandCirc(mito_contours, img_mat, ratio_threshold, circ, draw):
		filtered_contours = MatVector()
		for v in range(mito_contours.size()):
			rect = minAreaRect(mito_contours.get(v))
			poi = Point2f(8)
			rect.points(poi)

			# Get areas of mitos.
			area = contourArea(mito_contours.get(v))
			# Get perimeter of mitos.
			perim = arcLength(mito_contours.get(v), True)
			# Calculate circularities.
			circularity = (4 * math.pi * area)/(perim * perim)
			
			M = CvMoments()
			cvMoments(CvMat(mito_contours.get(v)), M)
					
			width = rect.size().width()
			height = rect.size().height()

			
			if height > 0 and circularity >= circ:
				ratio = float(height)/float(width)
				if ratio > ratio_threshold:
					if draw:
						BL = Point(int(poi.get(0)),int(poi.get(1)))
						TL = Point(int(poi.get(2)),int(poi.get(3)))
						TR = Point(int(poi.get(4)),int(poi.get(5)))
						BR = Point(int(poi.get(6)),int(poi.get(7)))
						line(img_mat,TL,TR,Scalar(255,255,0,0),3,0,0)
						line(img_mat,TL,BL,Scalar(255,255,0,0),3,0,0)
						line(img_mat,BR,TR,Scalar(255,255,0,0),3,0,0)
						line(img_mat,BR,BL,Scalar(255,255,0,0),3,0,0)
#						if M.m00() != 0:
#							cx = int(M.m10()/M.m00())
#							cy = int(M.m01()/M.m00())
#							putText(img_mat, str(round(ratio, 2)), Point(cx,cy),0,0.6,Scalar(0,0,0,255),2,0,False)
					filtered_contours.push_back(mito_contours.get(v))
		
		return filtered_contours
		
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
			roi = ImagePlus("ROI", roip)
		
			ROIcolmat, _ = Colour.makeMatColour(roi)
			final_mat = ROIcolmat.clone()
			mito_mat = ROIcolmat.clone()
		
			cnt_points, buckets = ContourUtils.contours2buckets(cnts, 0.15)
			
			
			cornered_bucket_cnts, filtered_buckets, _ = ContourUtils.filterBuckets(buckets, cnt_points, cnts, ROIcolmat, min_prtcle_size, max_prtcle_size, True)
#			ImagePlus("Mitochondira, sliced", mat2ip.toImageProcessor(ROIcolmat)).show() # ---
			
			found_cnt_mat, ROI_contours = ContourUtils.cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight)
			firstfound_cnt_vect.push_back(found_cnt_mat)
#			ImagePlus("Mitochondira, sliced", mat2ip.toImageProcessor(found_cnt_mat)).show()
		
			# Move ROI contours to position of image and append to final_contours_mat.
			offsetx = ROI_range[ind][0]
			offsety = ROI_range[ind][1]
			final_contours_mat = ContourUtils.moveContours(ROI_contours, offsetx, offsety, final_contours_mat)
			
			# Draw sliced contours on blank ROIs and re-create final image.
			sliced_mat = ContourUtils.floodfillContours(cnts, roiwidth, roiheight, cornered_bucket_cnts)
			sliced_cnt_vect.push_back(sliced_mat)
		
		#	ImagePlus("Mitochondira, sliced", mat2ip.toImageProcessor(sliced_mat)).show()
		return sliced_cnt_vect, firstfound_cnt_vect, final_contours_mat

	@staticmethod
	def processSingleROI(cnts, ROImat, ROI_dims, min_prtcle_size, max_prtcle_size):

		final_contours_mat = MatVector()

		# Get original image slice.
		roip = mat2ip.toImageProcessor(ROImat)
		roiwidth = ROImat.cols()
		roiheight = ROImat.rows()
		roi = ImagePlus("ROI", roip)
	
		ROIcolmat, _ = Colour.makeMatColour(roi)
		final_mat = ROIcolmat.clone()
		mito_mat = ROIcolmat.clone()
		
		cnt_points, buckets = ContourUtils.contours2buckets(cnts, 0.15)
		_, filtered_buckets, _ = ContourUtils.filterBuckets(buckets, cnt_points, cnts, ROIcolmat, min_prtcle_size, max_prtcle_size, False)

#		ImagePlus("Mitochondira, sliced", mat2ip.toImageProcessor(ROIcolmat)).show() # ---
		
		_, ROI_contours = ContourUtils.cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight)

		# Move ROI contours to position of image and append to final_contours_mat.
		offsetx = ROI_dims[0]
		offsety = ROI_dims[1]
		final_contours_mat = ContourUtils.moveContours(ROI_contours, offsetx, offsety, final_contours_mat)

		return final_contours_mat
	
	@staticmethod
	def processROIsThread(contour_list, ROI_matlist, ROI_range, min_prtcle_size, max_prtcle_size, threadIndex, out_sliced_cnt_vect, out_firstfound_cnt_vect, out_final_contours_mat):
		start = threadIndex * len(contour_list)/4
		end = threadIndex * len(contour_list)/4 +len(contour_list)/4
		if ContourUtils.isLastThread(threadIndex):
			end = len(contour_list)
	
		for ind in range(start, end):
			cnts = contour_list[ind]
			# Get original image slice.
			ROImat = ROI_matlist[ind]
			roip = mat2ip.toImageProcessor(ROImat)
			roiwidth = ROImat.cols()
			roiheight = ROImat.rows()
			roi = ImagePlus("ROI", roip)
		
			ROIcolmat, _ = Colour.makeMatColour(roi)
			final_mat = ROIcolmat.clone()
			mito_mat = ROIcolmat.clone()
		
			cnt_points, buckets = ContourUtils.contours2buckets(cnts, 0.15)
		
			cornered_bucket_cnts, filtered_buckets, imgmat = ContourUtils.filterBuckets(buckets, cnt_points, cnts, ROIcolmat, min_prtcle_size, max_prtcle_size, True)
#			ImagePlus("Mitochondira, sliced", mat2ip.toImageProcessor(ROIcolmat)).show() # ---
			
			found_cnt_mat, ROI_contours = ContourUtils.cleanDrawContours(cnts, final_mat, filtered_buckets, roiwidth, roiheight)
			out_firstfound_cnt_vect.push_back(found_cnt_mat)
		#	ImagePlus("Mitochondira, sliced", mat2ip.toImageProcessor(found_cnt_mat)).show()
		
			# Move ROI contours to position of image and append to final_contours_mat.
			offsetx = ROI_range[ind][0]
			offsety = ROI_range[ind][1]
			out_final_contours_mat = ContourUtils.moveContours(ROI_contours, offsetx, offsety, out_final_contours_mat)
			
			# Draw sliced contours on blank ROIs and re-create final image.
			sliced_mat = ContourUtils.floodfillContours(cnts, roiwidth, roiheight, cornered_bucket_cnts)
			out_sliced_cnt_vect.push_back(sliced_mat)

			if ind == end - 1:
				print(out_sliced_cnt_vect.size())
		
		#	ImagePlus("Mitochondira, sliced", mat2ip.toImageProcessor(sliced_mat)).show()
			
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
				avg_intensity = AreaUtils.getAvgIntensity(slce)
				slce_ip.setThreshold(avg_intensity-5, 255, ImageProcessor.NO_LUT_UPDATE)
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
			colored_image, img_matrix = Colour.makeMatColour(slce)
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
			avg_intensity = AreaUtils.getAvgIntensity(slce)
			slce_ip.setThreshold(avg_intensity-5, 255, ImageProcessor.NO_LUT_UPDATE)
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
		output_ROI, img_matrix = Colour.makeMatColour(slce)
		contours = MatVector()
		findContours(img_matrix, contours, RETR_LIST, CHAIN_APPROX_NONE)


		return contours, output_ROI, avg_intensity


class guiManager:

	@staticmethod		
	def runCheck(question_text):
		# Make edits decision.
		gui = GenericDialog("Continue")
		gui.addChoice(question_text, ["Yes - Deletion/Detection/Drawing mode", "Yes - Fine-tuning mode", "Find lipid droplets", "No, finish program"], "Yes - Deletion/Semi-automatic Mode")
		gui.showDialog()
		if gui.wasOKed():
			choice = gui.getNextChoice()
		else:
			choice = "No, finish program"
		return choice

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
		
		ImagePlus("custom edits", imgp_draw).show()
		rm.runCommand("show all")

		waiting = WaitForUserDialog("Fine-tuning Mode Activated", "Select the contour you with to edit from the ROI manager menu, it will turn blue. \nNext use the 'Selection Brush Tool' (under the oval menu) to alter the contour you have selected (note: there will be a line remaining where the original contour presided). \nWhen finished, click 'update' in the ROI manager menu, the old line will disappear. Now repeat for other chosen contours. \nPress ok to finish.")
		waiting.show()

		rm.runCommand("update")
		roi_array = rm.getRoisAsArray()
		altered_contours = []
		for roi_elem in roi_array:
			altered_contours.append(guiManager.parseCompositeROI(roi_elem))
		
		return altered_contours

	
