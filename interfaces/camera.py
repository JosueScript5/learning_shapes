import cv2
import face_recognition
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

def main():
    window_name = "Webcam - Détection visage avec Arial"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Chemin vers ta police Arial.ttf
    font_path = os.path.join("police", "Arial.ttf")
    font = ImageFont.truetype(font_path, 14)  # Taille 14 pour le texte

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur, impossible d'accéder à la webcam")
        return

    last_location = None
    face_detected = False
    message_to_display = "Humain"

    print("Appuyez sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Traitement de détection
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small[:, :, ::-1]
        locations = face_recognition.face_locations(rgb_small)

        if locations:
            top, right, bottom, left = locations[0]
            last_location = (top * 4, right * 4, bottom * 4, left * 4)
            face_detected = True
        else:
            face_detected = False

        if last_location and face_detected:
            top, right, bottom, left = last_location

            # Dessine un rectangle vert autour du visage
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Convertir l'image OpenCV en PIL
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Coordonnées du texte
            text_x = left
            text_y = bottom + 10

            # Taille du texte
            bbox = font.getbbox(message_to_display)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Dessiner un rectangle noir sous le visage
            draw.rectangle(
                [(text_x - 5, text_y - 5), (text_x + text_w + 5, text_y + text_h + 5)],
                fill=(0, 0, 0)
            )

            # Écrire le texte en blanc
            draw.text((text_x, text_y), message_to_display, font=font, fill=(255, 255, 255))

            # Reconvertir en OpenCV
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
