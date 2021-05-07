from ij	import IJ, WindowManager, ImagePlus, Prefs
from ij.plugin.frame import RoiManager
from ijopencv.ij import ImagePlusMatConverter as imp2mat
from ijopencv.opencv import MatImagePlusConverter as mat2ip
from ij import ImagePlus
from org.bytedeco.javacpp import Pointer, BytePointer
from org.bytedeco.javacpp.opencv_core  import Point2f, vconcat, hconcat, Rect, Mat, MatVector, GpuMatVector, KeyPointVector, Scalar, CvMat, vconcat,ACCESS_READ, Point
from org.bytedeco.javacpp.opencv_imgproc  import line, minAreaRect, getStructuringElement, floodFill, CvFont, drawMarker, putText, circle, MARKER_DIAMOND, arcLength, findContours, contourArea, CvMoments, drawContours, putText, boundingRect, cvtColor, RETR_FLOODFILL, RETR_LIST,CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS, CHAIN_APPROX_SIMPLE, COLOR_GRAY2RGB,COLOR_RGB2GRAY #cvThreshold
from ij.process import ImageProcessor
from net.imglib2.img.imageplus import ImagePlusImgs
from ij.gui import GenericDialog, Roi, Line, Plot, PointRoi, NewImage, MessageDialog, WaitForUserDialog, PolygonRoi
from ij.plugin import FFT, MontageMaker
from ij.plugin.filter import GaussianBlur, EDM
from ij.ImageStack import create, addSlice, deleteSlice, size, getSliceLabel
from java.awt import Color, Graphics
import math
import csv
import time
import sys
import os


sys.path.append(str(os.getcwd()) + "\\plugins\\Organelle_Finder\\")
from collection_ver7 import * 

# For testing open clean image automatically.
#path = ""
#img = IJ.openImage(path)

img = IJ.getImage() # use open image.
imgd = img.duplicate()

img_width = img.getWidth()
img_height = img.getHeight()

gui = GenericDialog("Expected size of mitochondria")
gui.addNumericField("Maximum (pixels): ", 200)
gui.addNumericField("Minimum (pixels): ", 80)
gui.showDialog()
if gui.wasOKed():
	max_prtcle_size = gui.getNextNumber()
	min_prtcle_size = gui.getNextNumber()

if max_prtcle_size < min_prtcle_size:
	sys.exit("The maximum particle size was set smaller than the minumum particle size")

if max_prtcle_size > img_width or max_prtcle_size > img_height:
	sys.exit("The maximum particle size is larger than the dimension of the image")


min_prtcle_area = math.pi * (min_prtcle_size/2 * min_prtcle_size/2)
max_prtcle_area = math.pi * (max_prtcle_size/2 * max_prtcle_size/2)
min_prtcle_perim = math.pi * min_prtcle_size/2
max_prtcle_perim = math.pi * max_prtcle_size/2 * 1.2

roi_height = int(max_prtcle_size * 1.5)
roi_width =  int(max_prtcle_size * 1.5)

ROI_obj = ROIProcessor(img, roi_width, roi_height)

# Find contours for each image in a list.
ROI_range, ROI_matlist, stack_dimensions = ROI_obj.getMultipleVariableROI()
contour_list, _ = ContourUtils.getVariableContours(ROI_matlist, "auto")

# Fort each ROI the following processing occurs: filtering by contour size (remove single points), clustering (into buckets), filtering buckets (by size)
sliced_cnt_vect, firstfound_cnt_vect, final_contours_mat = ContourUtils.processROIs(contour_list, ROI_matlist, ROI_range, min_prtcle_size, max_prtcle_size)

num_across = stack_dimensions[0]
num_down = stack_dimensions[1]

print(sliced_cnt_vect)
# Draw full image with edge mitochondria.
out_mat = ROIUtils.ROItoImage(sliced_cnt_vect, num_across, num_down) ##
edge_img = ImagePlus("Edges - tbd", mat2ip.toImageProcessor(out_mat))

# Draw full image with first found mitochondria.
firstout_mat = ROIUtils.ROItoImage(firstfound_cnt_vect, num_across, num_down)
allfinalcnts_mat = firstout_mat.clone()
firstout_img = ImagePlus("First round", mat2ip.toImageProcessor(firstout_mat))

# For edge image find contours and continue to assign buckets and filter.
edge_colored_image, edge_img_matrix = Colour.makeMatColour(edge_img)
edge_contours = MatVector()
findContours(edge_img_matrix, edge_contours, RETR_LIST, CHAIN_APPROX_NONE)
edge_cnt_points, edge_buckets = ContourUtils.contours2buckets(edge_contours, 0.15)
img_matrix, _ = Colour.makeMatColour(img)
imgp_draw = mat2ip.toImageProcessor(img_matrix.clone()) # for fine-tuning mode
_, edge_filtered_buckets, bucket_mat = ContourUtils.filterBuckets(edge_buckets, edge_cnt_points, edge_contours, img_matrix, min_prtcle_size, max_prtcle_size, True)

# Get clean contours for this final assignment of edge contours.
second_final_mat, _ = Colour.makeMatColour(imgd)
second_cnt_mat, return_contours = ContourUtils.cleanDrawContours(edge_contours, second_final_mat, edge_filtered_buckets, img_width, img_height)

# Show edge contours on original image.
second_roundip  = mat2ip.toImageProcessor(second_cnt_mat)
second_round = ImagePlus("second round", second_roundip)

# Draw edge contours on final image.
drawContours(allfinalcnts_mat, return_contours, -1, Scalar(255,0,0,0))
for i in range(return_contours.size()):
	# Append to final_contours_mat.
	final_contours_mat.push_back(return_contours.get(i))

# Show final image with all sucessful contours.
allfinalcnts_ip  = mat2ip.toImageProcessor(allfinalcnts_mat)
allfinalcnts = ImagePlus("final image", allfinalcnts_ip)

# Get minimum rectangle and filter for width.
filtered_img, _ = Colour.makeMatColour(imgd)
filtered_contours = ContourUtils.filterRotRectandCirc(final_contours_mat, filtered_img, 0.5, 0.5, False)
drawContours(filtered_img, filtered_contours, -1, Scalar(255,0,0,0))

ImagePlus("filtered final", mat2ip.toImageProcessor(filtered_img)).show()

imgmat, _ = Colour.makeMatColour(imgd)
imgp_draw = mat2ip.toImageProcessor(imgmat.clone())

cont = guiManager.runCheck("Make manual edits?")

while cont != "No, finish program":

	# Procedure for general edit mode (deletion/semi automatic).
	if cont == "Yes - Deletion/Detection/Drawing mode":
		rm_points, add_rois, manual_mitos = guiManager.getManualInput()
		all_contours_points = ContourUtils.getInteriorContoursPoints(filtered_contours, imp2mat.toMat(imgd.getProcessor()))
		cnts_to_remove = []

		# If more than one point is set, remove multiple contours.
		if len(rm_points) > 1:
			rm_list = listManipulator(sorted(rm_points))
			for a in range(len(all_contours_points)):
				if rm_list.overlap(sorted(all_contours_points[a])):
					cnts_to_remove.append(a)

		# If one point is set remove single contour.
		if len(rm_points) == 1:
			for a in range(len(all_contours_points)):
				p_list = listManipulator(sorted(all_contours_points[a]))
				if p_list.match(rm_points[0]):
					cnts_to_remove.append(a)
		
		# Remove selected contours.
		retained_contours = MatVector()
		for i in range(filtered_contours.size()):
			if i not in cnts_to_remove:
				retained_contours.push_back(filtered_contours.get(i))
		
		range_size = 10
		# Add in new ROIs.
		for elem in add_rois:
			_,_,w,h = elem
			manual_ROI = ContourUtils.getManualSingleROI(elem, imgd)
			manual_contour, _, intensity_value  = ContourUtils.getSingleVariableContours(manual_ROI, "auto")
			manual_contours = ContourUtils.processSingleROI(manual_contour, manual_ROI, elem, min_prtcle_size*0.3, max(img_width, img_height))
			larger_candidate_cnt = False
			min_contour_length = max(w, h) * 0.5
			for i in range(manual_contours.size()):
				height = minAreaRect(manual_contours.get(i)).size().height()
				width = minAreaRect(manual_contours.get(i)).size().width()
				if max(height, width) > min_contour_length:
					larger_candidate_cnt = True
					
			# If the inital search did not yield a long enough contour try over range and return largest.
			if larger_candidate_cnt:
				contours_in_range = MatVector()
				for factor in range(-int(range_size/2), int(range_size/2)):
					manual_contour, _, _  = ContourUtils.getSingleVariableContours(manual_ROI, intensity_value - 5 + factor)
					manual_contours = ContourUtils.processSingleROI(manual_contour, manual_ROI, elem, min_prtcle_size*0.3, max(img_width, img_height))
					# Get largest contour from manual_contours.
					max_area = ContourUtils.largestContourIndex(manual_contours)
					contours_in_range.push_back(manual_contours.get(max_area))
				max_area_in_range = ContourUtils.largestContourIndex(contours_in_range)
				retained_contours.push_back(contours_in_range.get(max_area_in_range))
			# If initial search was sucessful push the largest contour only.
			else:
				passing_cnt = ContourUtils.largestContourIndex(manual_contours)
				retained_contours.push_back(manual_contours.get(passing_cnt))

		# Add any manually drawn mitochondria.
		for elem in manual_mitos:
			custom = CustomContour(elem)
			#custom.sortPointsClockwise()
			contourmat = custom.toMat()
			retained_contours.push_back(contourmat)

		# Re-draw with new contours.
		man_filtered_img, _ = Colour.makeMatColour(imgd)
		drawContours(man_filtered_img, retained_contours, -1, Scalar(255,0,0,0))
		for j in range(retained_contours.size()):
			M = CvMoments()
			cvMoments(CvMat(retained_contours.get(j)), M)
			if M.m00() != 0:
				cx = int(M.m10()/M.m00())
				cy = int(M.m01()/M.m00())
				putText(imgmat, str(j), Point(cx,cy),0,0.6,Scalar(0,0,0,255),2,0,False)
		ImagePlus("manually filtered final", mat2ip.toImageProcessor(man_filtered_img)).show()
		filtered_contours = retained_contours
		
		cont = guiManager.runCheck("Any more edits to make?")

	# Procedure for manual selection mode.
	if cont == "Yes - Fine-tuning mode":
		retained_contours = MatVector()
		
		cntpoints = ContourUtils.getContoursPoints(filtered_contours)
		new_composite = guiManager.getAlteredContour(cntpoints, imgp_draw)

		for elem in new_composite:
			custom = CustomContour(elem)
			custom.sortPointsClockwise()
			contourmat = custom.toMat()
			retained_contours.push_back(contourmat)

		filtered_contours = retained_contours

		cont = guiManager.runCheck("Any more edits to make?")

	# Add lipid droplets with automatic search.
	if cont == "Find lipid droplets":

		# Get estimate size.
		gui = GenericDialog("Expected size of lipid droplets")
		gui.addNumericField("Maximum (pixels): ", 80)
		gui.addNumericField("Minimum (pixels): ", 30)
		gui.addNumericField("Lipid minimum intensity (x/255): ", 150)
		gui.showDialog()
		if gui.wasOKed():
			max_len = gui.getNextNumber()
			min_len = gui.getNextNumber()
			lipid_threshold = gui.getNextNumber()

		retained_contours = ContourUtils.getLipids(imgd.duplicate(), 175, max_len, min_len)

		for i in range(retained_contours.size()):
			filtered_contours.push_back(retained_contours.get(i))

		# Re-draw with new contours.
		lipid_filtered_img, _ = Colour.makeMatColour(imgd)
		drawContours(lipid_filtered_img, filtered_contours, -1, Scalar(255,0,0,0))
		for j in range(filtered_contours.size()):
			M = CvMoments()
			cvMoments(CvMat(filtered_contours.get(j)), M)
			if M.m00() != 0:
				cx = int(M.m10()/M.m00())
				cy = int(M.m01()/M.m00())
				putText(imgmat, str(j), Point(cx,cy),0,0.6,Scalar(0,0,0,255),2,0,False)
		ImagePlus("lipid filtered final", mat2ip.toImageProcessor(lipid_filtered_img)).show()

		cont = guiManager.runCheck("Any more edits to make?")

# When "No, finish program" is selected, process contours remaining in filtered_contours for output.
contour_center_locations = []
contour_lengths = []
contour_widths = []
contour_areas = []
contour_perims = []
contour_circularities = []
contour_intensities = []
label = []

final_labelled_img, _ = Colour.makeMatColour(imgd)
all_contour_points = ContourUtils.getInteriorContoursPoints(filtered_contours, imp2mat.toMat(imgd.getProcessor()))

drawContours(final_labelled_img, filtered_contours, -1, Scalar(255,0,0,0))
for j in range(filtered_contours.size()):
	
	# Get dimensions of mitos.
	rect = minAreaRect(filtered_contours.get(j))
	poi = Point2f(8)
	rect.points(poi)			
	width = rect.size().width()
	height = rect.size().height()
	contour_lengths.append(max(width, height))
	contour_widths.append(min(width, height))

	# Get areas of mitos.
	area = contourArea(filtered_contours.get(j))
	contour_areas.append(area)

	# Get perimeter of mitos.
	perim = arcLength(filtered_contours.get(j), True)
	contour_perims.append(perim)

	# Calculate circularities.
	circularity = (4 * math.pi * area)/(perim * perim)
	contour_circularities.append(circularity)

	# Get intensities.
	intensities = ContourUtils.getContourIntensities(all_contour_points[j], imgd)
	avg_inten = sum(intensities)/len(intensities)
	contour_intensities.append(avg_inten)

	if avg_inten > lipid_threshold:
		label.append("lipid")
		text_col = Scalar(0,255,0,0)
	else:
		label.append("mito")
		text_col = Scalar(0,0,0,255)
	
	M = CvMoments()
	cvMoments(CvMat(filtered_contours.get(j)), M)
	if M.m00() != 0:
		cx = int(M.m10()/M.m00())
		cy = int(M.m01()/M.m00())
		contour_center_locations.append((cx,cy))
		putText(final_labelled_img, str(j+1), Point(cx,cy),0,0.6,text_col,2,0,False)

ImagePlus("Output image", mat2ip.toImageProcessor(final_labelled_img)).show()

# Get output file path.
gui = GenericDialog("Save Spreadsheet")
gui.addFileField("Choose file path", None)
gui.showDialog()
if gui.wasOKed():
	output_spreadheet_path = str(gui.getNextString()) + ".csv"

# Write data to spreadsheet.
with open(output_spreadheet_path, mode='wb') as csv_file:
    fieldnames = ['Count', 'Label', 'Average Intensity', 'Cicularity', 'Length (px)', 'Width (px)', 'Area (px^2)', 'Perimeter (px)', 'Location']
    Writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    Writer.writeheader()
    for i in range(filtered_contours.size()):
        Writer.writerow({'Count': i+1, 'Label': label[i], 'Average Intensity': contour_intensities[i],
                         'Cicularity': contour_circularities[i], 'Length (px)': contour_lengths[i],
                         'Width (px)': contour_widths[i], 'Area (px^2)': contour_areas[i], 'Perimeter (px)': contour_perims[i],
                         'Location': contour_center_locations[i]})

		