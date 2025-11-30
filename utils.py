import spectral.io.envi as envi
from datetime import datetime
import os
import sys

def hex_to_rgb(hex_color):
    # Remove the '#' if it exists
    hex_color = hex_color.lstrip('#')
    
    # Convert the hex string to RGB values
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))


def create_unique_folder(base_path=".", prefix="folder_"):
    # Get the current time and date in a formatted string
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the folder name using the prefix and current time
    folder_name = f"{prefix}{time_str}"
    
    # Create the full path for the new folder
    folder_path = os.path.join(base_path, folder_name)
    
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)  # exist_ok=True will not raise an error if the folder exists
    
    return folder_path


def enviread(datafile, hdrfile):
   info = envi.open(hdrfile, datafile)
   data = info.load()
   return data, info


# this function is used to load files with relative paths
# it is necessary in order to make the bundled .exe file work
def get_resource_path(relative_path):
    """ Get the absolute path to a resource, works for both development and PyInstaller bundle """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)




