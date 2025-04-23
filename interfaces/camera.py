import cv2

def main():
    # Ouvre la webcam (index 0 pour la webcam par défaut)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur Impossible d'accéder à la webcam")
        return

    print("Appuyez sur 'q' pour quitter")

    while True:
        # Capture une image de la webcam
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de l'image")
            break

        # Affiche l'image dans une fenêtre
        cv2.imshow("Ma Webcam", frame)

        # Quitte si on appuie sur la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libère la caméra et ferme les fenêtres
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
