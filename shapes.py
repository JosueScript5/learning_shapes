import cv2
import numpy as np

# Génération de formes synthétiques
# Ajout de 10 formes géométriques courantes
# circle, ellipse, triangle, carré, pentagon, hexagon,
# heptagon, octagon, nonagon, decagon

def generate_shape(shape, img_size=200):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (img_size//2, img_size//2)
    radius = img_size//3

    if shape == 'circle':
        cv2.circle(img, center, radius, 255, -1)

    elif shape == 'ellipse':
        cv2.ellipse(img, center, (radius, int(radius*0.6)), 0, 0, 360, 255, -1)

    else:
        # Polygones réguliers du triangle (3 côtés) au décagone (10 côtés)
        mapping = {
            'triangle': 3,
            'carré': 4,
            'pentagon': 5,
            'hexagon': 6,
            'heptagon': 7,
            'octagon': 8,
            'nonagon': 9,
            'decagon': 10
        }
        sides = mapping.get(shape)
        if sides:
            pts = []
            for i in range(sides):
                theta = 2 * np.pi * i / sides - np.pi/2
                x = int(center[0] + radius * np.cos(theta))
                y = int(center[1] + radius * np.sin(theta))
                pts.append([x, y])
            pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(img, pts, 255)
    return img

# Extraction de caractéristiques
def extract_features(img):
    m = cv2.moments(img)
    hu = cv2.HuMoments(m).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

# Construction du dataset
def build_dataset(n_per_class=200):
    shapes = ['circle', 'ellipse', 'triangle', 'carré', 'pentagon',
              'hexagon', 'heptagon', 'octagon', 'nonagon', 'decagon']
    X, y = [], []
    for label in shapes:
        for _ in range(n_per_class):
            img = generate_shape(label)
            feat = extract_features(img)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)
