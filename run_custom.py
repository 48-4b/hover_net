#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

# Constants
CONDA_ENV_NAME = "hovernet_env"
# Attempt to find the python executable in the conda environment
# This path is specific to the current installation; for general use, rely on activated env
CONDA_PYTHON_PATH = os.path.expanduser("~/miniconda3/envs/hovernet_env/bin/python")

def main():
    parser = argparse.ArgumentParser(description="Run HoverNet Inference on an image.")
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("--output_dir", default="hovernet_output", help="Directory for output.")
    parser.add_argument("--model_mode", default="fast", choices=["original", "fast"], help="Model mode (fast=PanNuke, original=CoNSeP)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU/MPS if available") # Actually the underlying script handles this
    
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_image)
    output_path = os.path.abspath(args.output_dir)
    
    # Model selection logic
    if args.model_mode == "fast":
        model_path = os.path.abspath("checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar")
        nr_types = 6
        type_info = os.path.abspath("type_info.json")
    else:
        model_path = os.path.abspath("checkpoints/hovernet_original_consep_notype_tf2pytorch.tar")
        nr_types = 0 # or 5 depending on model, CoNSeP usually has implicit types?
        # Actually original CoNSeP checkpoint in this repo might be just segmentation or have different types
        # README says: "We do not provide a segmentation and classification model for CPM17 and Kumar... CoNSeP checkpoint... use original model mode"
        # I downloaded `hovernet_original_consep_notype_tf2pytorch.tar` so it is NO TYPE.
        nr_types = 0
        type_info = "''" # Empty string pass-through issue? I should handle it.

    # Prepare input directory for run_infer.py (it expects a directory of images)
    # We will create a temporary directory or just pass the parent dir if it only contains the image?
    # run_infer.py processes ALL images in input_dir.
    # To process a single image, we should probably symlink it to a temp dir.
    temp_input_dir = os.path.join(output_path, "_temp_input")
    os.makedirs(temp_input_dir, exist_ok=True)
    
    # Symlink or copy the image
    base_name = os.path.basename(input_path)
    temp_img_path = os.path.join(temp_input_dir, base_name)
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    try:
        os.symlink(input_path, temp_img_path)
    except OSError:
         import shutil
         shutil.copy(input_path, temp_img_path)

    # Determine python executable
    python_exe = sys.executable
    if "hovernet_env" not in python_exe and os.path.exists(CONDA_PYTHON_PATH):
        print(f"Using Conda Python: {CONDA_PYTHON_PATH}")
        python_exe = CONDA_PYTHON_PATH

    env = os.environ.copy()
    env["FORCE_CPU"] = "1"
    
    cmd = [
        python_exe, "run_infer.py",
        f"--model_path={model_path}",
        f"--model_mode={args.model_mode}",
        f"--nr_types={nr_types}",
        f"--batch_size=4",
        f"--nr_inference_workers=0", # Single threaded to avoid Mac multiprocessing issues
    ]

    if args.model_mode == "fast":
         cmd.append(f"--type_info_path={type_info}")

    cmd.extend([
        "tile",
        "--save_raw_map",
        f"--input_dir={temp_input_dir}",
        f"--output_dir={output_path}"
    ])

    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)
    
    # Cleanup temp dir
    try:
        os.remove(temp_img_path)
        os.rmdir(temp_input_dir)
    except:
        pass

    print(f"\nInference complete. Results in {output_path}")
    print(f"Overlay image: {os.path.join(output_path, 'overlay', base_name + '.png')}")
    print(f"JSON data: {os.path.join(output_path, 'json', base_name + '.json')}")

if __name__ == "__main__":
    main()
