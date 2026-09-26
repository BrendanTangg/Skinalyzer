import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

model_path = 'models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_face_result = None

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))
    global latest_face_result
    latest_face_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("No stream :(")
        exit()

    start_time = time.monotonic()
    last_timestamp_ms = -1

    while (True):
        ret, frame = stream.read()
        if not ret:
            print ("No more stream :(")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic()-start_time)*1000)

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_face_result and latest_face_result.face_landmarks:
            h, w = frame.shape[:2]

            face_landmarks = latest_face_result.face_landmarks[0]

            # Get image dimensions
            h, w = frame.shape[:2]

            # MediaPipe landmarks for approximate cheek locations
            left_cheek_lm = face_landmarks[50]
            right_cheek_lm = face_landmarks[280]

            # Convert normalized coordinates to pixel coordinates
            left_cheek_x = int(left_cheek_lm.x * w)
            left_cheek_y = int(left_cheek_lm.y * h)

            right_cheek_x = int(right_cheek_lm.x * w)
            right_cheek_y = int(right_cheek_lm.y * h)

            #Draw cheek points
            cv2.circle(
                frame,
                (left_cheek_x, left_cheek_y),
                8,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                (right_cheek_x, right_cheek_y),
                8,
                (0, 0, 255),
                -1
            )

            # Store all landmark x and y coordinates
            x_coords = []
            y_coords = []

            for lm in face_landmarks:
                x_coords.append(int(lm.x * w))
                y_coords.append(int(lm.y * h))

            # Find the boundaries of the face
            x1 = max(0, min(x_coords))
            x2 = min(w, max(x_coords))
            y1 = max(0, min(y_coords))
            y2 = min(h, max(y_coords))

            # Crop out the face
            face_region = frame[y1:y2, x1:x2]

            # Make sure the crop actually contains pixels
            if face_region.size > 0:

                # Convert face to grayscale
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

                # Slightly blur the image to reduce camera noise
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                # Detect changes in pixel intensity
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)

                # Calculate texture score
                texture_score = laplacian.var()

                # Display the texture score
                cv2.putText(
                    frame,
                    f"Texture: {texture_score:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Draw a rectangle around the area being analyzed
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            for lm in face_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
cv2.destroyAllWindows()