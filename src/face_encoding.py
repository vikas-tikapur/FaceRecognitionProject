import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    base_options = python.BaseOptions(
        model_asset_path="models/face_detector.tflite"
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )

    detector = vision.FaceDetector.create_from_options(options)
    result = detector.detect(mp_image)

    if not result.detections:
        print("No face detected")
        return

    for detection in result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_face("data/test_images/person1.jpg")