import face_recognition
import cv2
import numpy as np

# Load a sample picture and learn how to recognize it.
image_paths = ["./donaldTrump.jpeg", "./Joe_Biden.jpg",
               "./elaySason.jpg"]

recognized_face_encodings = []
for path in image_paths:
    cur_image = face_recognition.load_image_file(path)
    cur_encoding = face_recognition.face_encodings(cur_image)[0]
    recognized_face_encodings.append(cur_encoding)

recognized_face_names = [
    "Donald Trunp",
    "Joe Biden",
    "Elay Sason"
]

process_current_frame = True
face_locations = []
face_encodings = []
face_names = []

vidCapture = cv2.VideoCapture(0)
print('Press esc to end the program')
while True:

    # Press escape to quit video. 27 is the ASCII code of escape key
    if cv2.waitKey(1) == 27:
        break
    # Read one frame from the video
    ret, frame = vidCapture.read()

    # make the image 75% smaller in order to make the processing quicker
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Transforming form BGR to RGB to make the image compatible with face_recognition
    rgb_smaller_frame = smaller_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_current_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_smaller_frame)
        face_encodings = face_recognition.face_encodings(rgb_smaller_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Checking for potential matches with recognized known faces
            matches = face_recognition.compare_faces(recognized_face_encodings, face_encoding)
            name = "Unknown"

           
            face_distances = face_recognition.face_distance(recognized_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = recognized_face_names[best_match_index]

            face_names.append(name)

    process_current_frame = not process_current_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 4, bottom - 4), font, 1, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Image', frame)

# Release handle to the webcam
vidCapture.release()
cv2.destroyAllWindows()
