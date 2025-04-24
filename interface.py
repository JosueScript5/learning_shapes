import customtkinter as ctk
import cv2
from tkinter import filedialog, messagebox
from PIL import Image
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from shapes import build_dataset, extract_features

# Configuration de l'apparence
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ShapeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reconnaissance de Formes")
        self.geometry("700x500")
        self.resizable(False, False)

        # Variables
        self.algo_var = ctk.StringVar(value="SVM")
        self.model = None

        # Layout principal
        self.frame_left = ctk.CTkFrame(self)
        self.frame_left.pack(side="left", fill="y", padx=20, pady=20)

        # Ajouter un cadre intérieur pour padding (équivalent p-5)
        self.inner_frame = ctk.CTkFrame(self.frame_left, fg_color="transparent")
        self.inner_frame.pack(padx=30, pady=30, fill="both", expand=True)

        self.frame_right = ctk.CTkFrame(self)
        self.frame_right.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Widgets gauche (placés dans le cadre intérieur avec padding)
        ctk.CTkLabel(self.inner_frame, text="Choisir l'algorithme", font=ctk.CTkFont(size=14))\
            .pack(pady=(0,10))
        ctk.CTkOptionMenu(self.inner_frame, variable=self.algo_var,
                          values=["SVM", "Régression Logistique", "k-NN"])\
            .pack(pady=(0,20))
        ctk.CTkButton(self.inner_frame, text="Entraîner le modèle", command=self.train_model)\
            .pack(pady=(0,10))
        self.acc_label = ctk.CTkLabel(self.inner_frame, text="", font=ctk.CTkFont(size=12))
        self.acc_label.pack(pady=(0,20))
        ctk.CTkButton(self.inner_frame, text="Charger et prédire", command=self.predict_image)\
            .pack(pady=(0,10))
        self.pred_label = ctk.CTkLabel(self.inner_frame, text="", font=ctk.CTkFont(size=12), justify="left")
        self.pred_label.pack(pady=(0,10))

        # Widget droit pour afficher l'image (sans texte par défaut)
        self.image_label = ctk.CTkLabel(self.frame_right, text="")
        self.image_label.pack(fill="both", expand=True)

        # Nouveau : Label pour les noms des formes
        self.shapes_list_label = ctk.CTkLabel(
            self.frame_right,
            text="Types de formes disponibles\nCircle, Ellipse, Triangle, Carré, Pentagon,\nHexagon, Heptagon, Octagon, Nonagon, Decagon",
            font=ctk.CTkFont(size=12),
            justify="center"
        )
        self.shapes_list_label.pack(pady=10)

    def train_model(self):
        X, y = build_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        algo = self.algo_var.get()
        if algo == "SVM":
            self.model = SVC(kernel='rbf', probability=True)
        elif algo == "Régression Logistique":
            self.model = LogisticRegression(max_iter=500)
        else:
            self.model = KNeighborsClassifier()
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.acc_label.configure(text=f"Précision {acc*100:.2f}%")

    def predict_image(self):
        if not self.model:
            messagebox.showwarning("Erreur", "Entraînez d'abord le modèle")
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image")
            return
        img_resized = cv2.resize(img, (300,300))
        feat = extract_features(img_resized).reshape(1,-1)
        pred = self.model.predict(feat)[0]
        proba = self.model.predict_proba(feat).max()
        # Affichage du texte sur deux lignes
        self.pred_label.configure(text=f"Forme {pred}\n\nConfiance {proba:.1%}")
        # Affichage de l'image existante
        img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_color)
        img_ctk = ctk.CTkImage(light_image=img_pil, size=(300,300))
        self.image_label.configure(image=img_ctk)
        self.image_label.image = img_ctk

if __name__ == "__main__":
    app = ShapeApp()
    app.mainloop()
