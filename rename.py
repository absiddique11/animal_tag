import os

def rename_images(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only jpg files
    images = [file for file in files if file.lower().endswith('.jpg')]

    # Sort images to maintain order
    images.sort()

    # Rename files
    for index, filename in enumerate(images):
        new_name = f"{index + 101:04d}.jpg"  # Starting from 0101
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} to {new_name}")

# Example usage
folder_path = '/home/mdabubakrsiddique/Documents/animal_tag/tagVideos/im'
rename_images(folder_path)

