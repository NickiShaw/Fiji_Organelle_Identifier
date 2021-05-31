# Copyright (C) 2021 Nicolette Shaw - All Rights Reserved
from fiji.util.gui import GenericDialogPlus
import sys
import os

main_python_file_name = "Run_OrganelleFinder.py"
collection_file_name = "OrganelleFinder_helper.py"

# Try path for Windows.
expected_folder_path = str(os.getcwd()) + "\\plugins\\Organelle_Finder\\"

def errorPopup(errormessage):
	gui = GenericDialogPlus("Locate Installation folder")
	gui.hideCancelButton()
	gui.addMessage(errormessage)
	gui.showDialog()
	sys.exit(errormessage)

def present(element, listSet):
	if element in listSet:
		return True
	else:
		return False
def findMarker(marker, listSet):
	for i in range(len(listSet)):
		if marker in listSet[i]:
			return i
	errorPopup("Initialization Failed: Could not find marker in file " + main_python_file_name)


# Check if general file path exists.
try:
	expected = os.listdir(expected_folder_path)
	true_folder_path = expected_folder_path
# If failed, allow user to specify path.
except:
	gui = GenericDialogPlus("Locate Installation folder")
	gui.addDirectoryField("Choose Installation folder", None)
	gui.showDialog()
	if gui.wasOKed():
		true_folder_path = str(gui.getNextString())
	else:
		errorPopup("Initialization Failed: Locate the program folder on your computer and run initialization again")

# Check if all three files are present in the folder.
files_in_folder = os.listdir(true_folder_path)
found_file_count = 0
for x in files_in_folder:
	if x == main_python_file_name or x == collection_file_name:
		found_file_count += 1

# Escape the \ symbols to allow code to run.
true_folder_path_escaped = true_folder_path.replace('\\', '\\\\')

if found_file_count < 2:
	errorPopup("Initialization Failed: Please ensure both the " + main_python_file_name + " and " + collection_file_name + " files are present in you folder path and run initialization again")
else:
	print("Both important files were located in your path! Running initialization...")
	main_file_path = true_folder_path + '\\\\' + main_python_file_name
	text_to_write = 'sys.path.append("' + str(true_folder_path_escaped) + '")'

	# Put script in list_of_lines, and edit line 27.
	main_script = open(main_file_path, "r")
	list_of_lines = main_script.readlines()

	# Look for marker:
	line_num = findMarker("## Marker for path initialization ##", list_of_lines)
	list_of_lines[line_num] = str("## Marker for path initialization ##" + "\n" + text_to_write + "\n")

	# Write above previous file to include new line.
	main_script = open(main_file_path, "w")
	main_script.writelines(list_of_lines)
	main_script.close()

	gui = GenericDialogPlus("Initialization successful")
	gui.hideCancelButton()
	gui.addMessage("Initialization was successful, you can now run the OrganelleFinder program from the plugins dropdown")
	gui.addMessage("RESTART FIJI")
	gui.showDialog()

	