# Fiji_Organelle_Identifier
Fiji/ImageJ script that identifies organelles in EM images. Specifically developed for finding mitochondria and lipid droplets in skeletal muscle images.

## Installation
First make sure that Fiji is installed, make note of the location the download as you will need to add the files to this area (below this path is `C:\Users\name\Fiji.app`).

Open Fiji and navigate to `help > Update...`, when the `ImageJ Updater` window opens select `Manage update sites`, then the below window should open:
![Manage update sites](https://github.com/NickiShaw/Fiji_Organelle_Identifier/Images/update.jpg)
Select `IJ-OpenCV-Plugins` from the menu and close the `Manage update sites` window, select `Apply Changes` in the `ImageJ Updater` window.

Download the "Organelle_Finder" folder, then move the entire folder into the plugins folder. The path you are looking for should look like: `C:\Users\name\Fiji.app\plugins`.

Restart Fiji, now the "Organelle_Finder" should be located in the plugins dropdown in the menu (scroll to the bottom section of Plugins).

You must have an image open for the program to run, the program automatically takes the top-most image available (i.e. the most recently clicked-on image).
