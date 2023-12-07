import os
import glob
import shutil
import sys
import torch
import time

def find_latest_best_model(folder_path):
    search_path = os.path.join(folder_path, '**', 'best_model.pth')
    files = glob.glob(search_path, recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

timestamp = int(time.time())
model_dir_name = "voice-" + str(timestamp)
ttspath = os.path.expanduser("~/.local/share/tts")
save_model_path = os.path.join(ttspath, model_dir_name)
save_model_cp = os.path.join(save_model_path, "model.pth")
os.makedirs(save_model_path, exist_ok=True)

model_path = find_latest_best_model("/tmp/xtts_ft/run/training/")
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
del checkpoint["optimizer"]
for key in list(checkpoint["model"].keys()):
    if "dvae" in key:
        del checkpoint["model"][key]
torch.save(checkpoint, save_model_cp)
model_dir = os.path.dirname(model_path)
shutil.copy(os.path.join(model_dir, 'config.json'), save_model_path)
shutil.copy(os.path.join(model_dir, 'vocab.json'), save_model_path)

print("Model saved to " + save_model_path)
