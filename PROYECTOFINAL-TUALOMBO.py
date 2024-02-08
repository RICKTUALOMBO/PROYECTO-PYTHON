import cv2
import torch
import numpy as np
from sort import Sort
import matplotlib.path as mplPath

ZONE1 = np.array([
[639, 100],
[538, 155],
[693, 223],
[760, 155],
])

ZONE2 = np.array([
[129, 67],
[9, 136],
[101, 286],
[383, 217],
[390, 117],
])



def get_center(bbox):
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center

def load_model():
    model = torch.hub.load("ultralytics/yolov5", model="yolov5n", pretrained=True)
    return model

def get_bboxes(preds: object):
    df = preds.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.50]
    df = df[df["name"] == "person"]
    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)

def is_valid_detection(xc, yc, zone):
    return mplPath.Path(zone).contains_point((xc, yc))

def detector(cap: object):
    model = load_model()

    # Inicializar el tracker SORT
    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        preds = model(frame)
        bboxes = get_bboxes(preds)

        pred_confidences = preds.xyxy[0][:, 4].cpu().numpy()

        # Aplicar el tracker a las bounding boxes
        trackers = tracker.update(bboxes)

        detections_zone1 = 0
        detections_zone2 = 0
        detections_zone3 = 0

        for i, box in enumerate(trackers):
            xc, yc = get_center(box)
            # Convertir a enteros
            xc, yc = int(xc), int(yc)

            # Dibujar bounding boxes
            cv2.rectangle(img=frame, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 255, 0), thickness=2)

            # Dibujar centro de bounding boxes
            cv2.circle(img=frame, center=(xc, yc), radius=5, color=(255, 0, 0), thickness=-1)

            # Dibujar el ID de la persona
            cv2.putText(img=frame, text=f"id: {int(box[4])}, conf: {pred_confidences[i]:.2f}", org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 255), thickness=2)

            # Conteo de personas en cada zona
            if is_valid_detection(xc, yc, ZONE1):
                detections_zone1 += 1
            elif is_valid_detection(xc, yc, ZONE2):
                detections_zone2 += 1


        # Mostrar el conteo en pantalla
        cv2.putText(img=frame, text=f"Area 1: {detections_zone1}", org=(100, 200), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2)
        cv2.putText(img=frame, text=f"Area 2: {detections_zone2}", org=(100, 250), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2)

        cv2.polylines(img=frame, pts=[ZONE1], isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.polylines(img=frame, pts=[ZONE2], isClosed=True, color=(0, 255, 0), thickness=3)

        cv2.imshow("frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    cap = cv2.VideoCapture("C:/Users/59398/Desktop/Proyecto/video.webm")
    detector(cap)