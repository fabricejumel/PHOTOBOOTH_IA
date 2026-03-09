import cv2
import os
import time
import base64
import subprocess
from datetime import datetime
import requests
import numpy as np
import mediapipe as mp
import sys
import argparse

# ---------- CONFIG ----------
WEBUI_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"


PROMPT= (
    "comic book style, ligne claire art, clean line art, cel shading, "
    "group of people transformed into giant mecha robots, "
    "metallic armor, glowing eyes, mechanical arms, robot helmets, "
    "professional robot transformation scene"
)

NEGATIVE_PROMPT = (
    "photorealistic, 3d render, blurry, deformed, ugly, low quality, sketch, watercolor, "
    "messy lines, painterly, anime style, wrinkles, uniform wall, plain background, "
    "empty room, no robot, contemporary furniture, bad anatomy, extra limbs, "
    "deformed hands, mutated, disfigured"
)

# 🔥 Format caméra
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
WIDTH, HEIGHT = 1280, 720

# 🔥 NOUVEAUX PARAMÈTRES OPTIMISÉS
DENOISING_STRENGTH = 0.62
CFG_SCALE = 7.5
STEPS = 28
SEED = -1
SAMPLER_NAME = "DPM++ 2M Karras"

# 🔥 LOGO TEMPLATE
LOGO_TEMPLATE_PATH = "./logo/CPE_template_A6_paysage_small.png"

# 🔥 DÉTECTION GESTE - durées requises
GESTURE_HOLD_DURATION = 1.0  # 2 secondes pour valider un geste
GESTURE_COOLDOWN = 0.1       # 1 seconde de cooldown après geste

# 🔥 Écran 3840x1080 + barres de titre
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1920
TITLEBAR_HEIGHT = 52
TITLEBAR_HEIGHT = 52

# Hauteur disponible (sans barres)
AVAILABLE_HEIGHT = SCREEN_HEIGHT - 2 * TITLEBAR_HEIGHT
WEBCAM_HEIGHT = AVAILABLE_HEIGHT // 3
STABLEDIFF_HEIGHT = AVAILABLE_HEIGHT * 1 // 3

# Largeur proportionnelle 16:9
WEBCAM_WIDTH = (WEBCAM_HEIGHT * 16) // 9
STABLEDIFF_WIDTH = (STABLEDIFF_HEIGHT * 16) // 9

# 🔥 CONTROLNET SDXL
CONTROLNET_MODELS = {
    "openpose": "kohya_controllllite_xl_openpose_anime [7e5349e5]",
}

PRINTER_NAME = "HP_Color_LaserJet_5700"
PRINTER_OPTIONS = [
    "-o", "media=A6", "-o", "InputSlot=Tray2", "-o", "mediaType=HP-Brochure-Glossy-200g",
    "-o", "orientation-requested=4", "-o", "fit-to-page", "-o", "print-quality=5"
]

OUT_DIR = "../result_API_1111"
os.makedirs(OUT_DIR, exist_ok=True)

# SUPPRESSION WARNINGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# GLOBALES
last_final_imgs = []
last_final_paths = []
state = "waiting_victory"
gesture_message = "Faites le Signe V pour demarrer"
NO_PRINT = False
N_IMAGES = 1
logo_template = None
printing = False

# 🔥 GESTION GESTES STABLES
victory_start_time = None
thumbs_start_time = None
last_gesture_time = 0

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

def load_logo_template():
    """Charge le template logo avec transparence"""
    global logo_template
    if os.path.exists(LOGO_TEMPLATE_PATH):
        logo_template = cv2.imread(LOGO_TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
        if logo_template is not None:
            print(f"✓ Logo template chargé: {logo_template.shape}")
        else:
            print(f"⚠ Erreur chargement {LOGO_TEMPLATE_PATH}")
    else:
        print(f"⚠ Logo template introuvable: {LOGO_TEMPLATE_PATH}")

def apply_logo_overlay(image):
    """Applique le template logo transparent sur l'image"""
    global logo_template
    
    if logo_template is None:
        return image
    
    # Dimensions de l'image
    img_h, img_w = image.shape[:2]
    logo_h, logo_w = logo_template.shape[:2]
    
    # Redimensionne le logo si nécessaire (garde ratio)
    if logo_w != img_w or logo_h != img_h:
        logo_resized = cv2.resize(logo_template, (img_w, img_h), interpolation=cv2.INTER_AREA)
    else:
        logo_resized = logo_template
    
    # Applique overlay avec canal alpha si disponible
    if logo_resized.shape[2] == 4:  # RGBA
        alpha = logo_resized[:, :, 3] / 255.0
        
        # Convertit image BGR en BGRA si nécessaire
        if image.shape[2] == 3:
            result = image.copy()
        else:
            result = image[:, :, :3].copy()
        
        # Blend avec alpha
        for c in range(3):
            result[:, :, c] = (alpha * logo_resized[:, :, c] + 
                              (1 - alpha) * result[:, :, c])
        
        return result
    else:
        # Pas de transparence, simple overlay
        return cv2.addWeighted(image, 0.7, logo_resized[:, :, :3], 0.3, 0)

def find_first_camera():
    print("TEST CAMERAS (2,0,4,1,3)...")
    test_order = [2, 0, 4, 1, 3, 5, 6, 7, 8, 9]
    
    for i in test_order:
        try:
            print(f"  Test {i}...", end=" ")
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                print("NON")
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None and frame.size > 0:
                print("OK")
                return i
            print("no frame")
        except:
            print("crash")
            continue
    
    print("AUCUNE")
    return None

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def detect_gestures(frame):
    """Détecte les gestes Victory (V) et Thumbs Up"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = mp_hands.process(rgb_frame)
    
    victory_detected = False
    thumbs_up_detected = False
    
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
            
            # 🔥 SIGNE V (index + majeur levés)
            index_tip_y = landmarks[8][1]
            middle_tip_y = landmarks[12][1]
            ring_tip_y = landmarks[16][1]
            pinky_tip_y = landmarks[20][1]
            index_base_y = landmarks[5][1]
            middle_base_y = landmarks[9][1]
            
            # Index et majeur levés, annulaire et auriculaire baissés
            if (index_tip_y < index_base_y - 40 and 
                middle_tip_y < middle_base_y - 40 and
                ring_tip_y > landmarks[13][1] and  # Annulaire baissé
                pinky_tip_y > landmarks[17][1]):   # Auriculaire baissé
                victory_detected = True
            
            # 🔥 POUCE (thumb levé)
            thumb_tip_y = landmarks[4][1]
            thumb_base_y = landmarks[1][1]
            index_tip_y_thumb = landmarks[8][1]
            
            # Pouce levé ET index baissé (sinon conflit avec V)
            if (thumb_tip_y < thumb_base_y - 60 and 
                index_tip_y_thumb > landmarks[6][1]):  # Index pas levé
                thumbs_up_detected = True
    
    return victory_detected, thumbs_up_detected

def check_stable_gesture(victory_now, thumbs_now):
    """🔥 Vérifie si un geste est maintenu assez longtemps - PRIORITÉ EXPLICITE"""
    global victory_start_time, thumbs_start_time, last_gesture_time, state
    
    current_time = time.time()
    
    # Cooldown entre gestes
    if current_time - last_gesture_time < GESTURE_COOLDOWN:
        return None, 0.0, thumbs_now, victory_now
    
    # 🔥 PRIORITÉ SELON L'ÉTAT
    if state == "ready_print":
        # En attente d'impression: THUMBS prioritaire
        if thumbs_now and victory_now:
            victory_now = False  # Ignore victory si thumbs détecté
    
    # Gestion THUMBS
    if thumbs_now:
        if thumbs_start_time is None:
            thumbs_start_time = current_time
            print("DEBUG: THUMBS détecté - début compteur")
        held_duration = current_time - thumbs_start_time
        
        if held_duration >= GESTURE_HOLD_DURATION:
            # Geste validé !
            print(f"DEBUG: THUMBS VALIDÉ après {held_duration:.2f}s")
            thumbs_start_time = None
            victory_start_time = None
            last_gesture_time = current_time
            return "thumbs", held_duration, thumbs_now, victory_now
        else:
            # En cours de maintien
            victory_start_time = None  # Reset l'autre geste
            return "thumbs_holding", held_duration, thumbs_now, victory_now
    else:
        if thumbs_start_time is not None:
            print("DEBUG: THUMBS relâché - reset compteur")
        thumbs_start_time = None
    
    # Gestion VICTORY
    if victory_now:
        if victory_start_time is None:
            victory_start_time = current_time
            print("DEBUG: VICTORY détecté - début compteur")
        held_duration = current_time - victory_start_time
        
        if held_duration >= GESTURE_HOLD_DURATION:
            # Geste validé !
            print(f"DEBUG: VICTORY VALIDÉ après {held_duration:.2f}s")
            victory_start_time = None
            thumbs_start_time = None
            last_gesture_time = current_time
            return "victory", held_duration, thumbs_now, victory_now
        else:
            # En cours de maintien
            return "victory_holding", held_duration, thumbs_now, victory_now
    else:
        if victory_start_time is not None:
            print("DEBUG: VICTORY relâché - reset compteur")
        victory_start_time = None
    
    return None, 0.0, thumbs_now, victory_now

def draw_gesture_circle(frame, x, y, color, size=35):
    """Cercle coloré simple"""
    cv2.circle(frame, (x, y), size, color, 6)

def draw_progress_bar(frame, x, y, w, h, progress):
    """Barre de progression pour le maintien du geste"""
    # Fond gris
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), -1)
    # Remplissage vert selon progression
    filled_w = int(w * progress)
    cv2.rectangle(frame, (x, y), (x + filled_w, y + h), (0, 255, 0), -1)
    # Bordure
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

def draw_gesture_overlay(frame, w, h, gesture_status, gesture_progress, thumbs_detected, victory_detected):
    """🔥 Affiche les gestes détectés en BLANC à côté des cercles"""
    global state, gesture_message, NO_PRINT, N_IMAGES, printing
    
    overlay = frame.copy()
    
    if state == "waiting_victory":
        cv2.rectangle(overlay, (0, 0), (w, h//3), (150, 100, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, gesture_message, (30, 50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, "MAINTENIR SIGNE V 2s", (w//2-250, h-50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
        
        # 🔥 Affiche VICTORY détecté en BLANC
        if victory_detected:
            cv2.putText(frame, "V", (w-170, 95), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        
        # Affiche progression du geste
        if gesture_status == "victory_holding":
            draw_gesture_circle(frame, w-100, 80, (255, 255, 0))  # Jaune = en cours
            progress = gesture_progress / GESTURE_HOLD_DURATION
            draw_progress_bar(frame, w-200, 120, 180, 30, progress)
            cv2.putText(frame, f"{gesture_progress:.1f}s", (w-80, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if NO_PRINT:
            cv2.putText(frame, f"[NO_PRINT x{N_IMAGES}]", (15, h-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    elif state == "ready_print":
        cv2.rectangle(overlay, (0, 0), (w, h//3), (150, 100, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        if printing:
            cv2.putText(frame, "IMPRESSION EN COURS", (30, 60), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 165, 0), 3)
            cv2.putText(frame, "Attendez", (30, 110), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "IMAGE(S) PRETE(S), FAIS UN GESTE !!!", (30, 55), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, "POUCE 2s = IMPRIMER", (30, 95), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
            cv2.putText(frame, "SIGNE V 2s = NOUVELLE PHOTO", (30, 130), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            
            # 🔥 Affiche THUMBS détecté en BLANC
            if thumbs_detected:
                cv2.putText(frame, "POUCE", (w-220, 175), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
            
            # 🔥 Affiche VICTORY détecté en BLANC
            if victory_detected:
                cv2.putText(frame, "V", (w-170, 95), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
            
            # Affiche progression des gestes
            if gesture_status == "thumbs_holding":
                draw_gesture_circle(frame, w-100, 160, (255, 255, 0))  # Jaune
                progress = gesture_progress / GESTURE_HOLD_DURATION
                draw_progress_bar(frame, w-250, 200, 220, 30, progress)
                cv2.putText(frame, f"{gesture_progress:.1f}s", (w-80, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            elif gesture_status == "victory_holding":
                draw_gesture_circle(frame, w-100, 80, (255, 255, 0))  # Jaune
                progress = gesture_progress / GESTURE_HOLD_DURATION
                draw_progress_bar(frame, w-200, 120, 180, 30, progress)
                cv2.putText(frame, f"{gesture_progress:.1f}s", (w-80, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if NO_PRINT:
            cv2.putText(frame, "[SIMULATION]", (15, h-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def print_images():
    """🔥 Imprime webcam original + toutes les images IA"""
    global last_final_paths, NO_PRINT, N_IMAGES, printing
    
    if printing:
        print("IMPRESSION DEJA EN COURS")
        return
    
    # 🔥 Prend original (_input) + N_IMAGES IA
    expected = N_IMAGES + 1
    if len(last_final_paths) < expected:
        print(f"Images manquantes: {len(last_final_paths)}/{expected}")
        return
    
    printing = True
    
    print(f"\nPRINTER {'SIMULATION' if NO_PRINT else 'HP'} {PRINTER_NAME}")
    
    # 🔥 Prend les N_IMAGES+1 dernières (original + toutes les IA)
    images_to_print = last_final_paths[-expected:]
    
    if NO_PRINT:
        for i, img in enumerate(images_to_print, 1):
            label = "WEBCAM" if "_input" in img else f"IA{i-1}"
            print(f"  {label}: {os.path.basename(img)} OK")
            time.sleep(0.3)
        print(f"SIMULATION OK - {expected} images (webcam + {N_IMAGES} IA)\n")
        time.sleep(1)
    else:
        for i, img in enumerate(images_to_print, 1):
            label = "WEBCAM" if "_input" in img else f"IA"
            # 🔥 MODE SILENCIEUX COMPLET
            cmd = ["lp", "-d", PRINTER_NAME, "-o", "job-sheets=none", "-s"] + PRINTER_OPTIONS + [img]
            result = subprocess.run(cmd,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   stdin=subprocess.DEVNULL)  # 🔥 Aussi stdin
            print(f"  {label} {i}/{expected} {'OK' if result.returncode == 0 else 'ERREUR'}")
            time.sleep(0.5)
        print("Attente fin impression...")
        time.sleep(3)
    
    printing = False
    print("IMPRESSION TERMINEE\n")


def call_api_images(frame_bgr, capture_ts):
    """🔥 Génère les images IA avec logo + sauvegarde l'original AVEC logo aussi"""
    global N_IMAGES, last_final_paths
    try:
        # 🔥 Sauvegarde l'original AVEC logo
        frame_with_logo = apply_logo_overlay(frame_bgr)
        orig_path = os.path.join(OUT_DIR, f"{capture_ts}_input.png")
        cv2.imwrite(orig_path, frame_with_logo)
        print(f"Original + Logo: {orig_path}")
        last_final_paths.append(orig_path)
        
        # Image pour l'API (SANS logo)
        img_path = os.path.join(OUT_DIR, f"{capture_ts}_resized.png")
        cv2.imwrite(img_path, frame_bgr)
        
        payload = {
            "prompt": PROMPT, 
            "negative_prompt": NEGATIVE_PROMPT,
            "init_images": [img_to_b64(img_path)], 
            "width": WIDTH, 
            "height": HEIGHT,
            "denoising_strength": DENOISING_STRENGTH, 
            "seed": SEED,
            "steps": STEPS,
            "cfg_scale": CFG_SCALE,
            "sampler_name": SAMPLER_NAME,
            "batch_size": 1, 
            "n_iter": N_IMAGES,
        }
        
        print(f"API → {N_IMAGES} images (CFG={CFG_SCALE}, Denoise={DENOISING_STRENGTH})...")
        r = requests.post(WEBUI_URL, json=payload, timeout=180)
        
        if r.status_code != 200:
            print(f"API ERREUR {r.status_code}")
            return None
        
        result = r.json()
        if len(result.get("images", [])) < N_IMAGES:
            print(f"Moins de {N_IMAGES} images")
            return None
        
        paths = []
        for i, b64_img in enumerate(result["images"][:N_IMAGES]):
            if "," in b64_img: 
                b64_img = b64_img.split(",", 1)[1]
            
            # 🔥 Sauvegarde temporaire SANS logo
            temp_path = os.path.join(OUT_DIR, f"{capture_ts}_IA{i+1}_temp.png")
            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(b64_img))
            
            # 🔥 Charge et applique le logo
            img = cv2.imread(temp_path)
            img_with_logo = apply_logo_overlay(img)
            
            # 🔥 Sauvegarde finale AVEC logo
            out_path = os.path.join(OUT_DIR, f"{capture_ts}_IA{i+1}.png")
            cv2.imwrite(out_path, img_with_logo)
            
            # Supprime temporaire
            os.remove(temp_path)
            
            paths.append(out_path)
            last_final_paths.append(out_path)
            print(f"IA{i+1} + Logo: {out_path}")
        
        return paths
    except Exception as e:
        print(f"API: {e}")
        return None

def countdown_flash_live(cap):
    print("3-2-1...")
    
    start = time.time()
    while time.time() - start < 4.0:
        ret, frame = cap.read()
        if not ret: return None
        
        display = frame.copy()
        elapsed = time.time() - start
        remaining = 4.0 - elapsed
        
        blue_overlay = np.ones_like(display) * [150, 100, 0]
        display = cv2.addWeighted(blue_overlay.astype(np.uint8), 0.4, display.astype(np.uint8), 0.6, 0)
        
        if remaining > 3.0:
            text = "PREPAREZ-VOUS"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 8)[0]
            text_x = (CAMERA_WIDTH - text_size[0]) // 2
            cv2.putText(display, text, (text_x, CAMERA_HEIGHT//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 8)
                       
        elif remaining > 0.5:
            count = str(max(1, int(remaining)))
            text_size = cv2.getTextSize(count, cv2.FONT_HERSHEY_SIMPLEX, 15, 20)[0]
            text_x = (CAMERA_WIDTH - text_size[0]) // 2
            text_y = (CAMERA_HEIGHT + text_size[1]) // 2
            cv2.putText(display, count, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 15, (255, 255, 255), 20)
                       
        else:
            text = "PHOTO!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 10)[0]
            text_x = (CAMERA_WIDTH - text_size[0]) // 2
            text_y = (CAMERA_HEIGHT + text_size[1]) // 2
            cv2.putText(display, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)
            cv2.imshow("Webcam", display)
            time.sleep(0.5)
            return frame
        
        cv2.imshow("Webcam", display)
        if cv2.waitKey(30) & 0xFF == ord('q'): return None
    
    return None

def main(device, no_print=False, n_images=1):
    global NO_PRINT, N_IMAGES, state, gesture_message, last_final_imgs, printing
    
    NO_PRINT = no_print
    N_IMAGES = n_images
    
    # 🔥 CHARGE LE LOGO AU DÉMARRAGE
    load_logo_template()
    
    # AUTO-DETECT
    if device == 'auto':
        device = find_first_camera()
        if device is None:
            print("PAS DE CAMERA")
            return
    
    print(f"Camera {device} | {'NO_PRINT' if NO_PRINT else 'PRINT'} | {N_IMAGES} IA")
    print(f"Impression: WEBCAM + {N_IMAGES} IA = {N_IMAGES+1} photos total")
    print(f"Prompt: Comic Book Ligne Claire | CFG={CFG_SCALE} | Denoise={DENOISING_STRENGTH}")
    print(f"Geste: maintenir {GESTURE_HOLD_DURATION}s pour valider")
    print(f"Ecran 3840x1080 | Webcam {WEBCAM_WIDTH}x{WEBCAM_HEIGHT} | SD {STABLEDIFF_WIDTH}x{STABLEDIFF_HEIGHT}")
    
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"{device} impossible")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("READY | Maintenir Signe V 2s=Photo | Maintenir POUCE 2s=Print | Q=Quit")
    
    # CENTRAGE 3840x1080
    total_windows_height = WEBCAM_HEIGHT + STABLEDIFF_HEIGHT + 2 * TITLEBAR_HEIGHT
    y_start = 0 # max(0, (SCREEN_HEIGHT - total_windows_height) // 2)
    
    webcam_x = (SCREEN_WIDTH - WEBCAM_WIDTH) // 2
    stablediff_x = (SCREEN_WIDTH - STABLEDIFF_WIDTH) // 2
    
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Webcam", WEBCAM_WIDTH, WEBCAM_HEIGHT)
    cv2.moveWindow("Webcam", webcam_x, y_start)
    
    cv2.namedWindow("Image StableDiffusion", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Image StableDiffusion", STABLEDIFF_WIDTH, STABLEDIFF_HEIGHT)
    cv2.moveWindow("Image StableDiffusion", stablediff_x, y_start + WEBCAM_HEIGHT + 2 * TITLEBAR_HEIGHT)
    
    last_final_imgs = [None] * N_IMAGES
    current_display_img = None
    processing_frame = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        victory_now, thumbs_now = detect_gestures(frame)
        gesture_status, gesture_progress, thumbs_detected, victory_detected = check_stable_gesture(victory_now, thumbs_now)
        
        key = cv2.waitKey(1) & 0xFF
        
        # STATES
        if state == "waiting_victory":
            if gesture_status == "victory" or key == 32:
                print("SIGNE V VALIDÉ - COUNTDOWN!")
                state = "countdown"
                
        elif state == "countdown":
            frame_final = countdown_flash_live(cap)
            if frame_final is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"{ts} | Generation {N_IMAGES} IA...")
                
                processing_frame = frame_final.copy()
                blue_overlay = np.ones_like(processing_frame) * [150, 100, 0]
                processing_frame = cv2.addWeighted(blue_overlay.astype(np.uint8), 0.4, processing_frame.astype(np.uint8), 0.6, 0)
                
                text = "EN COURS DE TRAITEMENT, 15s"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)[0]
                text_x = (CAMERA_WIDTH - text_size[0]) // 2
                text_y = (CAMERA_HEIGHT + text_size[1]) // 2
                cv2.putText(processing_frame, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
                
                cv2.imshow("Webcam", processing_frame)
                cv2.waitKey(1)
                
                paths = call_api_images(frame_final, ts)
                if paths and len(paths) == N_IMAGES:
                    last_final_imgs = [cv2.imread(path) for path in paths]
                    current_display_img = last_final_imgs[-1]
                    cv2.imshow("Image StableDiffusion", current_display_img)
                    
                    state = "ready_print"
                    gesture_message = "IMAGE(S) PRETE(S), FAIS UN GESTE !!!"
                    print(f"READY")
                else:
                    state = "waiting_victory"
                
        elif state == "ready_print":
            if not printing and (gesture_status == "thumbs" or key == ord('a')):
                print("POUCE VALIDÉ - PRINT!")
                print_images()
            elif printing:
                if gesture_status or key == ord('a'):
                    print("Impression en cours, patientez...")
            
            if not printing and (gesture_status == "victory" or key == 32):
                print("SIGNE V VALIDÉ - NOUVELLE PHOTO!")
                state = "countdown"
                gesture_message = "Faites le Signe V pour demarrer"
        
        # DISPLAY
        if state != "countdown":
            display = draw_gesture_overlay(frame.copy(), CAMERA_WIDTH, CAMERA_HEIGHT, 
                                          gesture_status, gesture_progress, 
                                          thumbs_detected, victory_detected)
            cv2.imshow("Webcam", display)
            if current_display_img is not None:
                cv2.imshow("Image StableDiffusion", current_display_img)
        
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo Booth MDM 2025")
    parser.add_argument("--device", "-d", default='auto', help="Camera")
    parser.add_argument("--no-print", action="store_true", help="Simulation")
    parser.add_argument("--images", "-n", type=int, default=1, help="Images IA")
    args = parser.parse_args()
    
    main(args.device, args.no_print, args.images)

