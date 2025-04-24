import os
import shutil
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

# Préparation du modèle
def generate_shape(shape, img_size=200):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    if shape == 'circle':
        cv2.circle(img, (img_size//2, img_size//2), img_size//3, 255, -1)
    elif shape == 'Carré':
        s = img_size//3
        start = img_size//2 - s//2
        cv2.rectangle(img, (start, start), (start+s, start+s), 255, -1)
    elif shape == 'triangle':
        pts = np.array([
            [img_size//2, img_size//2 - img_size//3],
            [img_size//2 - img_size//3, img_size//2 + img_size//3],
            [img_size//2 + img_size//3, img_size//2 + img_size//3]
        ])
        cv2.drawContours(img, [pts], 0, 255, -1)
    return img

def extract_features(img):
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

# Génération des données d'entraînement
features, labels = [], []
for shape in ['circle', 'Carré', 'triangle']:
    for _ in range(200):
        img = generate_shape(shape)
        features.append(extract_features(img))
        labels.append(shape)
X = np.array(features)
y = np.array(labels)

# Split train/test (ici on ne fera que l'entraînement)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Entraînement
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# CustomTkinter
# Créer le dossier images s'il n'existe pas
IMAGES_DIR = os.path.join(os.getcwd(), 'images')
if not os.path.isdir(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('blue')

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Reconnaissance de formes')
        self.geometry('600x600')

        # Bouton pour sélectionner une image
        self.btn_select = ctk.CTkButton(
            self, text='Sélectionner une image', command=self.select_image
        )
        self.btn_select.pack(pady=20)

        # Label pour afficher l'image
        self.label_image = ctk.CTkLabel(self, text='')
        self.label_image.pack(pady=10)

        # Label pour le résultat
        self.label_result = ctk.CTkLabel(self, text='', font=ctk.CTkFont(size=16))
        self.label_result.pack(pady=20)

    def select_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[('Images', '*.jpg *.jpeg *.png')]
        )
        if not filepath:
            return
        # Copier dans le dossier images
        dest_path = os.path.join(IMAGES_DIR, os.path.basename(filepath))
        shutil.copy2(filepath, dest_path)

        # Charger et afficher l'image
        img_bgr = cv2.imread(dest_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (200, 200))

        im_pil = Image.fromarray(img_bgr)
        im_pil = im_pil.resize((300, 300))
        photo = ImageTk.PhotoImage(im_pil)
        self.label_image.configure(image=photo)
        self.label_image.image = photo

        # Prédiction
        feat = extract_features(img_resized).reshape(1, -1)
        pred = clf.predict(feat)[0]
        proba = clf.predict_proba(feat).max()
        self.label_result.configure(
            text=f"Prédiction {pred}\nConfiance {proba:.1%}"
        )

if __name__ == '__main__':
    app = App()
    app.mainloop()
