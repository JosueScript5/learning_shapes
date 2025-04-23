import face_recognition

# Charger l'image de la personne connue
image_connue = face_recognition.load_image_file("images/visages/Messi.jpeg")
encodage_connu = face_recognition.face_encodings(image_connue)[0]

# Charger l'image à tester
image_inconnue = face_recognition.load_image_file("images/visages/MessiBarça.jpeg")
encodages_inconnus = face_recognition.face_encodings(image_inconnue)

# Vérifier s'il y a au moins un visage détecté dans l'image inconnue
if len(encodages_inconnus) == 0:
    print("Aucun visage trouvé dans l'image inconnue")
else:
    encodage_inconnu = encodages_inconnus[0]

    # Comparer les visages
    correspondance = face_recognition.compare_faces([encodage_connu], encodage_inconnu)
    distance = face_recognition.face_distance([encodage_connu], encodage_inconnu)[0]

    if correspondance[0]:
        print(f"✅ Visage reconnu (distance {distance:.2f})")
    else:
        print(f"❌ Visage non reconnu (distance {distance:.2f})")
