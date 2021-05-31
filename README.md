# Fiji_Organelle_Identifier
Fiji/ImageJ script that identifies organelles in EM images. Specifically developed for finding mitochondria and lipid droplets in skeletal muscle images.

To learn more about the tool and all it's features please refer to the [wiki](https://github.com/NickiShaw/Fiji_Organelle_Identifier/wiki).

:bangbang: **It is recommended that you read the wiki to understand this tool better before use.** :bangbang:


## Installation
First make sure that Fiji is installed (find the install [here](https://imagej.net/Fiji/Downloads), make note of the location the download as you will need to add the files to this area (below this path is `C:\Users\name\Fiji.app`). Note that 'ImageJ' on it's own is not sufficient to run this program.

Open Fiji and navigate to `help > Update...`, when the `ImageJ Updater` window opens select `Manage update sites`, then the below window should open:
![Manage update sites](https://github.com/NickiShaw/Fiji_Organelle_Identifier/blob/main/Images/update.jpg)
Select `IJ-OpenCV-Plugins` from the menu and close the `Manage update sites` window, select `Apply Changes` in the `ImageJ Updater` window.

Download the [Organelle_Finder](https://github.com/NickiShaw/Fiji_Organelle_Identifier/tree/main/Organelle_Finder) folder, then move the entire folder into the plugins folder. The path you are looking for should look like: `C:\Users\name\Fiji.app\plugins`.

Restart Fiji, now the "Organelle_Finder" should be located in the plugins dropdown in the menu. In the Fiji menu go to Plugins > Organelle_Finder and select `OrganelleFinder Initialize`, this will run an initialization which will point the scripts to the location of the 'Organelle_Finder' folder you created. If the folder cannot be located automatically, you will have to navigate to the 'Organelle_Finder' folder in the popup window (example shown below).

<img src="https://github.com/NickiShaw/Fiji_Organelle_Identifier/blob/main/Images/Initializer_window_1.jpg" alt="Initialization manual search window" width="200"/>

In this case toggle to the 'Organelle_Finder' as shown below:
![Initialization manual search window in directory](https://github.com/NickiShaw/Fiji_Organelle_Identifier/blob/main/Images/Initializer_window_2.jpg)

If the initialization is successful, the following window will appear:
![Initialization passed window](https://github.com/NickiShaw/Fiji_Organelle_Identifier/blob/main/Images/Initializer_passed.jpg)

If the initialization failed, the following window will appear. This indicates that the file path you gave was incorrect, or not all the necessary files are in the 'Organelle_Finder' folder. Check that the colder contains:
1. OrganelleFinder_helper.py
2. Run_OrganelleFinder.py

![Initialization passed window](https://github.com/NickiShaw/Fiji_Organelle_Identifier/blob/main/Images/Initializer_failed.jpg)



You must have an image open for the program to run, the program automatically takes the top-most image available (i.e. the most recently clicked-on image).

:bangbang: **The input image must be 8-bit (grayscale)** :bangbang:

## Features

Please refer to the [wiki](https://github.com/NickiShaw/Fiji_Organelle_Identifier/wiki) for details on usage.

## Bugs/New features

If you encounter any bugs please email the bug report and the image to *nashaw@uwaterloo.ca* for troubleshooting help. Please also indicate in the email if this request is urgent.

If you would like new features please also email *nashaw@uwaterloo.ca* with as much detail as possible for what you would like developed. Please also indicate in the email if this request is urgent.

# Workflow (temp.)
![Manage update sites](https://github.com/NickiShaw/Fiji_Organelle_Identifier/blob/main/Images/workflow.jpg)
