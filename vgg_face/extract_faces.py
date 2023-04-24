import mtcnn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def extract_face(file_path, input_shape=(224, 224)):
    pixels = plt.imread(file_path)

    # create a face detector to get bounding box of face
    detector = mtcnn.MTCNN()
    results = detector.detect_faces(pixels)

    # get coordinates of bounding box
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height

    # Get only face from image
    face = pixels[y1:y2, x1:x2]

    # Resize to proper input shape
    image = Image.fromarray(face)
    image = image.resize(input_shape)
    face_array = np.asarray(image)
    
    return face_array


def create_faces_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for celeb in os.listdir(input_dir):
        celeb_dir = os.path.join(input_dir, celeb)
        face_dir = os.path.join(output_dir, celeb)
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
        for filename in os.listdir(celeb_dir):
            path = os.path.join(celeb_dir, filename)
            face = extract_face(path)
            # save face array to file
            plt.imsave(f"{face_dir}\{filename}", face)

        

if __name__ == "__main__":
    # if you keep the paths as follows they should be one directory above the repo
    input_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Celebrity_Faces_Dataset'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Celebrity_Faces_Cropped_Dataset'))

    create_faces_dataset(input_dir, output_dir)