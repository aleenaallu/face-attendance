import cv2
import face_recognition

# Load images of known faces and encode them
image_of_person1 = face_recognition.load_image_file("person1.jpeg")
image_of_person2 = face_recognition.load_image_file("person2.jpeg")

encoding_of_person1 = face_recognition.face_encodings(image_of_person1)[0]
encoding_of_person2 = face_recognition.face_encodings(image_of_person2)[0]

# Create a list of known face encodings and their corresponding names
known_face_encodings = [encoding_of_person1, encoding_of_person2]
known_face_names = ["shifas", "nishad"]

# Initialize some variables
attendance = {}  # Dictionary to store attendance data

# Start the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Recognize faces
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Compare face_encoding with known_face_encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        if True in matches:
            matched_index = matches.index(True)
            name = known_face_names[matched_index]

            # Update attendance dictionary
            if name not in attendance:
                attendance[name] = True

        # Draw rectangle around the face and label it with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame with detected faces
    cv2.imshow('Face Attendance System', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Save the attendance data to a file or database
with open('attendance.txt', 'w') as file:
    for name, present in attendance.items():
        file.write(f"{name}: {'Present' if present else 'Absent'}\n")