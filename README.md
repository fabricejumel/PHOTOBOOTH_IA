# PHOTOBOOTH_IA

# 4 terminals
# terminator, " 3 horizontaux, 4eme vertical a coté des 3
# créer le venv venv_photobooth_venv la premiere fois et installer les requirements requirements.txt

```bash
#terminal 1, necessite l installation de comfyui et le model SDXL BASE 1 
cd ~/wp_comfui/
source comfyui-env/bin/activate
cd ~/wp_comfui/ComfyUI
python main.py --listen 0.0.0.0
```
```bash
#terminal 2
cd ~/wp_photobooth
source venv_photobooth_venv/bin/activate
python A1111_comfyui_proxy_v3.py
```

```bash
#terminal 3
cd ~/wp_photobooth
source venv_photobooth_venv/bin/activate
python CPE_FINAL_Photobooth_scn.py
```

```bash
#terminal 4
nvtop
```






#j ai cree sur certaine machine le layout photobooth pour terminator inclu dans le repo...
