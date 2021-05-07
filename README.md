# Fiji_Organelle_Identifier
Fiji/ImageJ script that identifies organelles in EM images. Specifically developed for finding mitochondria and lipid droplets in skeletal muscle images.

To learn more about the tool and all it's features please refer to the [wiki](https://github.com/NickiShaw/Fiji_Organelle_Identifier/wiki).

:bangbang: **It is recommended that you read the wiki to understand this tool better before use.** :bangbang:


## Installation
First make sure that Fiji is installed (find the install [here](https://imagej.net/Fiji/Downloads), make note of the location the download as you will need to add the files to this area (below this path is `C:\Users\name\Fiji.app`).

Open Fiji and navigate to `help > Update...`, when the `ImageJ Updater` window opens select `Manage update sites`, then the below window should open:
![Manage update sites](https://github.com/NickiShaw/Fiji_Organelle_Identifier/blob/main/Images/update.jpg)
Select `IJ-OpenCV-Plugins` from the menu and close the `Manage update sites` window, select `Apply Changes` in the `ImageJ Updater` window.

Download the [Organelle_Finder](https://github.com/NickiShaw/Fiji_Organelle_Identifier/tree/main/Organelle_Finder) folder, then move the entire folder into the plugins folder. The path you are looking for should look like: `C:\Users\name\Fiji.app\plugins`.

Restart Fiji, now the "Organelle_Finder" should be located in the plugins dropdown in the menu (scroll to the bottom section of Plugins).

You must have an image open for the program to run, the program automatically takes the top-most image available (i.e. the most recently clicked-on image).


## Bugs/New features

If you encounter any bugs please email the bug report and the image to *nashaw@uwaterloo.ca* for troubleshooting help. Please also indicate in the email if this request is urgent.

If you would like new features please also email *nashaw@uwaterloo.ca* with as much detail as possible for what you would like developed. Please also indicate in the email if this request is urgent.
