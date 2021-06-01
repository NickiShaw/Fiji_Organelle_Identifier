# Copyright (C) 2021 Nicolette Shaw - All Rights Reserved
import gc
import sys
import os
## Marker for path initialization ##
from ij	import IJ, WindowManager, ImagePlus, Prefs
from ij.plugin.frame import RoiManager
from ijopencv.ij import ImagePlusMatConverter as imp2mat
from ijopencv.opencv import MatImagePlusConverter as mat2ip
from ij import ImagePlus
from fiji.util.gui import GenericDialogPlus
from org.bytedeco.javacpp import Pointer, BytePointer
from org.bytedeco.javacpp.opencv_core  import Point2f, vconcat, hconcat, Rect, Mat, MatVector, GpuMatVector, KeyPointVector, Scalar, CvMat, vconcat,ACCESS_READ, Point
from org.bytedeco.javacpp.opencv_imgproc  import line, minAreaRect, getStructuringElement, floodFill, CvFont, drawMarker, putText, circle, MARKER_DIAMOND, arcLength, findContours, contourArea, CvMoments, drawContours, putText, boundingRect, cvtColor, RETR_FLOODFILL, RETR_LIST,CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS, CHAIN_APPROX_SIMPLE, COLOR_GRAY2RGB,COLOR_RGB2GRAY #cvThreshold
from ij.process import ImageProcessor
from net.imglib2.img.imageplus import ImagePlusImgs
from ij.gui import Roi, Line, Plot, PointRoi, NewImage, MessageDialog, WaitForUserDialog, PolygonRoi
from ij.plugin import FFT, MontageMaker
from ij.plugin.filter import GaussianBlur, EDM
from ij.ImageStack import create, addSlice, deleteSlice, size, getSliceLabel
from java.awt import Color, Graphics
import math
import csv
import time

from OrganelleFinder_helper import *

# List for settings record.
output_lines = []

# For testing open clean image automatically.
#path = ""
#img = IJ.openImage(path)

img = IJ.getImage() # use open image.

# Check that image is 8-bit greyscale.
if img.getType() != 0: # 0 = GRAY8, 1 = GRAY16, 2 = GRAY32, 3= COLOR_256, 4 = COLOR_RGB
	img_error = GenericDialogPlus("Analysis Complete")
	img_error.addMessage("Please provide an 8-bit greyscale image")
	img_error.addMessage("Ending program")
	img_error.showDialog()
	sys.exit("Image provided was not 8-bit greyscale image")

# Get file type to match saved file at end (won't work for incomplete file names).
try:
	input_filetype = str(img.getTitle()).rsplit('.')[1]
except:
	input_filetype = "tif"

# Get dimensions for resizing or continuing.
img_width = img.getWidth()
img_height = img.getHeight()

# Downsize large images (1500x1500 maximum)
if max(img_width, img_width) > 1500:
	if (img_width > img_height):
		new_width = 1500
		new_height = img_height / (img_width/new_width)
	elif (img_width < img_height):
		new_height = 1500
		new_width = img_width / (img_height/new_height)
	else:
		print("equal")
		new_width = 1500
		new_height = 1500
	img = img.resize(new_width, new_height, 'bilinear')
	img_width = new_width
	img_height = new_height
	ImagePlus("Resized Image", img.getProcessor()).show()

imgd = img.duplicate()
RGBImg, _, img_mat = ImgUtils.convertImgColour(imgd, "GRAY2RGB")

for _ in [0]:
	decision_package = guiManager.particleOptions(250, 60, 5, "")
	if decision_package == None:
		break
	# Dimensions in wrong order.
	max_prtcle_size, min_prtcle_size, lipid_threshold = decision_package
	if min_prtcle_size > max_prtcle_size:
		decision_package = guiManager.particleOptions(min(250, img_width, img_height), 60, decision_package[2], "ERROR: Maximum expected particle size was set lower than minimum expected particle size")
		if decision_package == None:
			break
	# Maximum is too large for image.
	max_prtcle_size, min_prtcle_size, lipid_threshold = decision_package
	if max_prtcle_size > img_width or max_prtcle_size > img_height:
		max_prtcle_size, min_prtcle_size, lipid_threshold = guiManager.particleOptions(min(250, img_width, img_height), decision_package[1], decision_package[2], "ERROR: The maximum particle size is larger than the dimension of the image")
		if decision_package == None:
			break
	max_prtcle_size, min_prtcle_size, lipid_threshold = decision_package
	output_lines.append("Maximum lipid size = " + str(max_prtcle_size) + " (px)")
	output_lines.append("Minimum lipid size = " + str(min_prtcle_size) + " (px)")
	output_lines.append("Lipid threshold = " + str(lipid_threshold))

# Ask if auto processing should be performed or skip to manual only.
autoprocessing = guiManager.autoManual()

if autoprocessing:
	
	roi_height = int(max_prtcle_size * 1.5)
	roi_width =  int(max_prtcle_size * 1.5)
	
	time1 = time.time()
	
	ROI_obj = ROIProcessor(img, roi_width, roi_height)
	
	# Find contours for each image in a list.
	ROI_range, ROI_matlist, stack_dimensions = ROI_obj.getMultipleVariableROI()
	contour_list, _ = ContourUtils.getVariableContours(ROI_matlist, lipid_threshold)
	
	# Fort each ROI the following processing occurs: filtering by contour size (remove single points), clustering (into buckets), filtering buckets (by size)
	sliced_cnt_vect, firstfound_cnt_vect, final_contours_mat = ContourUtils.processROIs(contour_list, ROI_matlist, ROI_range, min_prtcle_size, max_prtcle_size)
	
	num_across = stack_dimensions[0]
	num_down = stack_dimensions[1]

	# Draw full image with first found mitochondria.
	firstout_mat = ROIUtils.ROItoImage(firstfound_cnt_vect, num_across, num_down)
	allfinalcnts_mat = firstout_mat.clone()
	
	# Draw full image with edge mitochondria.
	out_mat = ROIUtils.ROItoImage(sliced_cnt_vect, num_across, num_down) ##
	edge_img = ImagePlus("Edges - tbd", mat2ip.toImageProcessor(out_mat))
	del out_mat
	del sliced_cnt_vect
	
	# For edge image find contours and continue to assign buckets and filter.
	edge_colored_image, edge_img_matrix, _ =  ImgUtils.convertImgColour(edge_img, "GRAY2RGB")
	edge_contours = MatVector()
	findContours(edge_img_matrix, edge_contours, RETR_LIST, CHAIN_APPROX_NONE)
	edge_cnt_points, edge_buckets = ContourUtils.contours2buckets(edge_contours, 0.15)
	img_matrix = img_mat.clone()
	_, edge_filtered_buckets, bucket_mat = ContourUtils.filterBuckets(edge_buckets, edge_cnt_points, edge_contours, img_matrix, min_prtcle_size, max_prtcle_size, True)
	del img_matrix
	
	# Get clean contours for this final assignment of edge contours.
	second_final_mat =  img_mat.clone()
	second_cnt_mat, return_contours = ContourUtils.cleanDrawContours(edge_contours, second_final_mat, edge_filtered_buckets, img_width, img_height, min_prtcle_size, max_prtcle_size)
	del second_final_mat
	
	# Draw edge contours on final image.
	drawContours(allfinalcnts_mat, return_contours, -1, Scalar(255,0,0,0), 2, -1, Mat(), 2, Point(0,0))
	for i in range(return_contours.size()):
		# Append to final_contours_mat.
		final_contours_mat.push_back(return_contours.get(i))
	
	# Show final image with all sucessful contours.
	allfinalcnts_ip  = mat2ip.toImageProcessor(allfinalcnts_mat)
	allfinalcnts = ImagePlus("final image", allfinalcnts_ip)
	del allfinalcnts_mat
	gc.collect()

	# Show combined unfiltered.
	both_img = img_mat.clone()
	drawContours(both_img, final_contours_mat, -1, Scalar(255,0,0,0), 2, -1, Mat(), 2, Point(0,0))
	ImagePlus("Combined unfiltered", mat2ip.toImageProcessor(both_img)).show()
	
	# Filter all contours given manual inputs.
	continueorReFilter = "Return"
	while continueorReFilter == "Return":
		filtering_dict = guiManager.filteringOptions("Filtering options")
		if filtering_dict != "Finish program":
			filtered_img = img_mat.clone()
			filtered_contours = ContourFindUtils.filterContoursbyManualOptions(final_contours_mat, imgd, filtering_dict)
			drawContours(filtered_img, filtered_contours, -1, Scalar(255,0,0,0), 2, -1, Mat(), 2, Point(0,0))
			ImagePlus("filtered final", mat2ip.toImageProcessor(filtered_img)).show()
			continueorReFilter = guiManager.keepFilterChanges()
			if continueorReFilter == "Return":
				ImagePlus("Combined unfiltered", mat2ip.toImageProcessor(both_img)).show()
			else:
				output_lines.append("Filtering: Circularity > " + str(filteringOutput(filtering_dict, 'circularity_filter')))
				output_lines.append("Filtering: Ratio > " + str(filteringOutput(filtering_dict, 'ratio_filter')))
				output_lines.append("Filtering: Maximum length < " + str(filteringOutput(filtering_dict, 'maxlen_filter')))
				output_lines.append("Filtering: Minimum length > " + str(filteringOutput(filtering_dict, 'minlen_filter')))
				output_lines.append("Filtering: Maximum intensity < " + str(filteringOutput(filtering_dict, 'maxinten_filter')))
				output_lines.append("Filtering: Minimum intensity > " + str(filteringOutput(filtering_dict, 'mininten_filter')))
		else:
			continueorReFilter = "Keep"
			ImagePlus("Combined unfiltered", mat2ip.toImageProcessor(both_img)).show()
			filtered_contours = final_contours_mat
			continueorReFilter == "Keep"
	
	imgmat = img_mat.clone()
	imgp_draw = mat2ip.toImageProcessor(imgmat.clone())

cont = guiManager.manualOptions("Make manual edits?")

# Make blank filtered_contours object if the automatic processing was skipped.
if not autoprocessing:
	filtered_contours = MatVector()

lipid_threshold = None
removed_contours = 0
added_contours = 0
drawn_contours = 0

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
				removed_contours += 1
				if rm_list.overlap(sorted(all_contours_points[a])):
					cnts_to_remove.append(a)

		# If one point is set remove single contour.
		if len(rm_points) == 1:
			removed_contours += 1
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
		failed_ROIs = 0
		# For each drawn ROI, locate the largest contour.
		for elem in add_rois:
			larger_candidate_cnt = True
			_,_,w,h = elem
			manual_ROI = ContourUtils.getManualSingleROI(elem, imgd)
			manual_contour, intensity_value  = ContourUtils.getSingleVariableContours(manual_ROI, "auto")
			# Try the original threshold with the specific ROI.
			manual_contours = ContourUtils.processSingleROI(manual_contour, manual_ROI, elem, 0, max(img_width, img_height))
			min_contour_length = max(w, h) * 0.5
			# Decide whether to try over range, if any contour is a good enough size and circularity, it passes.
			# If no contours were found, automatically continues to range loop.
			for i in range(manual_contours.size()):
				_, _, circularity, _, length = ContourFindUtils.contourStats(manual_contours.get(i))
				if length > min_contour_length and circularity > 0.6:
					retained_contours.push_back(manual_contours.get(i))
					added_contours += 1
					# Skip range loop if a sufficient contour was found.		
					larger_candidate_cnt = False
			# If the inital search did not yield a good enough contour try over range and return largest/most circular.
			if larger_candidate_cnt:
				contours_in_range = MatVector()
				for factor in range(-int(range_size/2), int(range_size/2), 2):
					manual_contour, _  = ContourUtils.getSingleVariableContours(manual_ROI, intensity_value - 5 + factor)
					manual_contours = ContourUtils.processSingleROI(manual_contour, manual_ROI, elem, 0, max(img_width, img_height))
					if manual_contours.size() != 0:
						# Add detected contours to master MatVector for whole range.
						for i in range(manual_contours.size()):
							contours_in_range.push_back(manual_contours.get(i))
				if contours_in_range.size() > 0:
					best_contour_idx = ContourUtils.bestContourIndex(contours_in_range)
					retained_contours.push_back(contours_in_range.get(best_contour_idx))
					added_contours += 1
				else:
					failed_ROIs += 1

		
		# Add any manually drawn mitochondria.
		for elem in manual_mitos:
			drawn_contours += 1
			custom = CustomContour(elem)
			#custom.sortPointsClockwise()
			contourmat = custom.toMat()
			retained_contours.push_back(contourmat)

		# Re-draw with new contours.
		man_filtered_img = img_mat.clone()
		drawContours(man_filtered_img, retained_contours, -1, Scalar(255,0,0,0), 2, -1, Mat(), 2, Point(0,0))
		ImagePlus("manually filtered final", mat2ip.toImageProcessor(man_filtered_img)).show()
		filtered_contours = retained_contours
		# Print message if a ROI didn't return any contours.
		if failed_ROIs > 0:
			contour_error = GenericDialogPlus("Manual contour error")
			contour_error.addMessage(str(failed_ROIs) + " contours couldn't be detected")
			contour_error.addMessage("Try manual drawing")
			contour_error.showDialog()
			
		cont = guiManager.manualOptions("Any more edits to make?")

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

		# Re-draw with new contours.
		tuned_img = img_mat.clone()
		drawContours(tuned_img, filtered_contours, -1, Scalar(255,0,0,0), 2, -1, Mat(), 2, Point(0,0))
		ImagePlus("tuned image", mat2ip.toImageProcessor(tuned_img)).show()

		cont = guiManager.manualOptions("Any more edits to make?")

	# Add lipid droplets with automatic search.
	if cont == "Find lipid droplets":

		# Get estimate size.
		max_len, min_len, lipid_threshold = guiManager.lipidOptions(80, 30, 150, "")
		# If lipid threshold excedes 255, or dimensions are flipped.
		if lipid_threshold > 255:
			max_len, min_len, lipid_threshold = guiManager.lipidOptions(max_len, min_len, 150, "ERROR: Lipid intensity exceded the maximum value of 255, please re-try")
		if min_len > max_len:
			max_len, min_len, lipid_threshold = guiManager.lipidOptions(max_len, min_len, lipid_threshold, "ERROR: Maximum expected lipid size was set lower than minimum expected lipid size")

		output_lines.append("Lipid droplets threshold = " + str(lipid_threshold))
		output_lines.append("Lipid droplets maximum size = " + str(max_len) + " (px)")
		output_lines.append("Lipid droplets minimum size = " + str(min_len) + " (px)")
		
		lipid_contours = ContourFindUtils.findContoursFromImage(imgd.duplicate(), lipid_threshold)
		retained_contours = ContourFindUtils.filterContoursBySize(lipid_contours, max_len, min_len)

		for i in range(retained_contours.size()):
			filtered_contours.push_back(retained_contours.get(i))

		# Re-draw with new contours.
		lipid_filtered_img = img_mat.clone()
		drawContours(lipid_filtered_img, filtered_contours, -1, Scalar(255,0,0,0), 2, -1, Mat(), 2, Point(0,0))
		ImagePlus("lipid filtered final", mat2ip.toImageProcessor(lipid_filtered_img)).show()

		cont = guiManager.manualOptions("Any more edits to make?")


# Draw final image with numbers on contours.
final_numbered_img = img_mat.clone()
drawContours(final_numbered_img, filtered_contours, -1, Scalar(255,0,0,0), 2, -1, Mat(), 2, Point(0,0))
for j in range(filtered_contours.size()):
	M = CvMoments()
	cvMoments(CvMat(filtered_contours.get(j)), M)
	if M.m00() != 0:
		cx = int(M.m10()/M.m00())
		cy = int(M.m01()/M.m00())
		putText(final_numbered_img, str(j+1), Point(cx,cy),0,0.6,Scalar(0,0,0,255),2,0,False)
final_imgplus = ImagePlus("Final", mat2ip.toImageProcessor(final_numbered_img))
final_imgplus.show()

# Add manual stats to settings output.
output_lines.append("Manually removed contours = " + str(removed_contours))
output_lines.append("Manually added contours = " + str(added_contours))
output_lines.append("Manually drawn contours = " + str(drawn_contours))

# When "No, finish program" is selected, process contours remaining in filtered_contours for output.
contour_center_locations = []
contour_lengths = []
contour_widths = []
contour_areas = []
contour_perims = []
contour_circularities = []
contour_intensities = []
label = []


all_contour_points = ContourUtils.getInteriorContoursPoints(filtered_contours, imp2mat.toMat(imgd.getProcessor()))

# If lipid_threshold is not set, default to 150.
if lipid_threshold == None:
	lipid_threshold = 150

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

	# Get centers.
	M = CvMoments()
	cvMoments(CvMat(filtered_contours.get(j)), M)
	if M.m00() != 0:
		cx = int(M.m10()/M.m00())
		cy = int(M.m01()/M.m00())
		contour_center_locations.append((cx,cy))
	else:
		contour_center_locations.append((0,0))
		
# Get output file path.
gui = GenericDialogPlus("Save Spreadsheet")
gui.addFileField("Choose file path", None)
gui.hideCancelButton()
gui.showDialog()
if gui.wasOKed():
	base_filename = str(gui.getNextString())
	output_spreadheet_path = base_filename + ".csv"

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

	# Output settings chosen for run.
	output_settings_path = base_filename + ".txt"
	settings_file = open(output_settings_path, "w")
	for i in range(len(output_lines)):
		settings_file.write(output_lines[i])
		settings_file.write("\n")
	settings_file.close()

gui = GenericDialogPlus("Analysis Complete")
gui.addMessage("Would you like to save the final image?")
gui.addRadioButtonGroup("Image format", ["tiff","jpeg","png"], 3, 1, guiManager.getLikely(input_filetype))
gui.setOKLabel("Yes")
gui.setCancelLabel("No")
gui.showDialog()
if gui.wasOKed():
	output_filetype = gui.getNextRadioButton()
	IJ.saveAs(final_imgplus, output_filetype, base_filename)

	