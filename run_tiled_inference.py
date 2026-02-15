"""
Tiled inference pipeline with magnification correction.

The input image appears to be ~10x magnification, but PanNuke model expects 40x.
This script:
1. Tiles the image into overlapping patches
2. Applies R/G channel swap (Cyan -> Purple color correction)
3. Upscales each patch 4x to simulate 40x magnification
4. Runs HoVerNet inference on each upscaled patch
5. Downscales results back to original resolution
6. Stitches results together with Voronoi-based cell boundary expansion
"""
import cv2
import numpy as np
import os
import sys
import subprocess
import scipy.io
import shutil
from scipy.ndimage import label as ndimage_label
from scipy.spatial import Voronoi

def run_tiled_inference(image_path, output_dir, scale_factor=4, tile_size=256, overlap=32):
    """Run inference with magnification correction."""
    
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Apply R/G channel swap (Cyan -> Purple color correction)
    img_fixed = img_rgb.copy()
    img_fixed[..., 0] = img_rgb[..., 1]  # New R = Old G
    img_fixed[..., 1] = img_rgb[..., 0]  # New G = Old R
    
    # Create tile directory
    tile_dir = os.path.join(output_dir, "_tiles_input")
    tile_output_dir = os.path.join(output_dir, "_tiles_output")
    
    if os.path.exists(tile_dir):
        shutil.rmtree(tile_dir)
    os.makedirs(tile_dir)
    
    # Generate tiles
    step = tile_size - overlap
    tiles_info = []
    tile_idx = 0
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = img_fixed[y_start:y_end, x_start:x_end]
            
            # Skip mostly-white tiles (no tissue)
            gray = cv2.cvtColor(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            tissue_frac = np.sum(gray < 200) / gray.size
            if tissue_frac < 0.1:
                continue
            
            # Upscale tile
            tile_up = cv2.resize(tile, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_CUBIC)
            
            # Save tile
            tile_path = os.path.join(tile_dir, f"tile_{tile_idx:04d}.png")
            cv2.imwrite(tile_path, cv2.cvtColor(tile_up, cv2.COLOR_RGB2BGR))
            
            tiles_info.append({
                'idx': tile_idx,
                'x': x_start,
                'y': y_start,
                'w': x_end - x_start,
                'h': y_end - y_start,
            })
            tile_idx += 1
    
    print(f"Created {tile_idx} tiles (skipped white tiles)")
    
    if tile_idx == 0:
        print("No tiles with tissue found!")
        return
    
    # Run HoVerNet inference on all tiles
    cmd = [
        sys.executable, "run_infer.py",
        f"--model_path={os.path.abspath('checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar')}",
        "--model_mode=fast",
        "--nr_types=6",
        "--batch_size=4",
        "--nr_inference_workers=0",
        f"--type_info_path={os.path.abspath('type_info.json')}",
        "tile",
        "--save_raw_map",
        f"--input_dir={os.path.abspath(tile_dir)}",
        f"--output_dir={os.path.abspath(tile_output_dir)}"
    ]
    
    env = os.environ.copy()
    env["FORCE_CPU"] = "1"
    print("Running HoVerNet inference on tiles...")
    subprocess.call(cmd, env=env)
    
    # Stitch results back
    print("Stitching results...")
    full_inst_map = np.zeros((h, w), dtype=np.int32)
    full_prob_map = np.zeros((h, w), dtype=np.float32)
    inst_offset = 0
    all_centroids = []
    all_types = []
    
    for info in tiles_info:
        mat_path = os.path.join(tile_output_dir, "mat", f"tile_{info['idx']:04d}.mat")
        if not os.path.exists(mat_path):
            continue
        
        data = scipy.io.loadmat(mat_path)
        inst_map = data['inst_map']
        
        # Downscale back to original resolution
        inst_map_small = cv2.resize(inst_map.astype(np.float32), 
                                     (info['w'], info['h']),
                                     interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        # Raw probability map
        raw_map = data['raw_map']
        prob_map = raw_map[..., 1]  # NP probability channel
        prob_map_small = cv2.resize(prob_map, (info['w'], info['h']),
                                     interpolation=cv2.INTER_LINEAR)
        
        # Place into full map with offset
        y, x = info['y'], info['x']
        tile_h, tile_w = info['h'], info['w']
        
        # Only overwrite if new probability is higher (handle overlaps)
        region_prob = full_prob_map[y:y+tile_h, x:x+tile_w]
        mask = prob_map_small > region_prob
        
        # Re-label instances with offset
        tile_inst = inst_map_small.copy()
        tile_inst[tile_inst > 0] += inst_offset
        
        full_inst_map[y:y+tile_h, x:x+tile_w][mask] = tile_inst[mask]
        full_prob_map[y:y+tile_h, x:x+tile_w] = np.maximum(region_prob, prob_map_small)
        
        max_inst = inst_map.max()
        if max_inst > 0:
            inst_offset += max_inst
        
        # Collect centroids (downscale positions)
        if 'inst_centroid' in data and data['inst_centroid'].size > 0:
            centroids = data['inst_centroid'] / scale_factor
            centroids[:, 0] += x  # X offset
            centroids[:, 1] += y  # Y offset
            all_centroids.extend(centroids.tolist())
            
            if 'inst_type' in data and data['inst_type'].size > 0:
                all_types.extend(data['inst_type'].flatten().tolist())
    
    # Count final instances
    unique_inst = np.unique(full_inst_map)
    num_inst = len(unique_inst) - (1 if 0 in unique_inst else 0)
    print(f"Total detected nuclei: {num_inst}")
    
    # Save results
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    # Save inst map
    scipy.io.savemat(os.path.join(output_dir, "results", "inst_map.mat"), {
        'inst_map': full_inst_map,
        'prob_map': full_prob_map,
    })
    
    # Create overlay visualization
    overlay = img_rgb.copy()
    
    # Draw nuclear contours
    for inst_id in unique_inst:
        if inst_id == 0:
            continue
        mask = (full_inst_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)  # Red contours
    
    # Draw centroids
    for c in all_centroids:
        cx, cy = int(c[0]), int(c[1])
        cv2.circle(overlay, (cx, cy), 2, (0, 255, 0), -1)  # Green dots
    
    cv2.imwrite(os.path.join(output_dir, "results", "overlay_nuclei.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Create Voronoi-based cell boundary visualization
    if len(all_centroids) > 3:
        cell_overlay = create_cell_visualization(img_rgb, full_inst_map, all_centroids, all_types)
        cv2.imwrite(os.path.join(output_dir, "results", "overlay_cells.png"),
                    cv2.cvtColor(cell_overlay, cv2.COLOR_RGB2BGR))
    
    # Save probability heatmap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Nuclear Segmentation ({num_inst} nuclei)")
    axes[1].axis('off')
    
    axes[2].imshow(full_prob_map, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title("Detection Probability")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results", "summary.png"), dpi=150)
    plt.close()
    
    print(f"Results saved to {output_dir}/results/")
    return full_inst_map, all_centroids


def create_cell_visualization(img_rgb, inst_map, centroids, types=None):
    """Create cell-level visualization using dilated nuclear masks."""
    h, w = img_rgb.shape[:2]
    
    # Dilate each nucleus to approximate cell boundaries
    from scipy.ndimage import binary_dilation, distance_transform_edt
    
    cell_map = np.zeros((h, w), dtype=np.int32)
    
    # Use distance transform to expand nuclei into cell regions
    fg_mask = inst_map > 0
    if fg_mask.sum() == 0:
        return img_rgb.copy()
    
    # For each nucleus, dilate it
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    unique_ids = np.unique(inst_map)
    unique_ids = unique_ids[unique_ids > 0]
    
    for uid in unique_ids:
        nuc_mask = (inst_map == uid).astype(np.uint8)
        dilated = cv2.dilate(nuc_mask, kernel, iterations=2)
        # Only assign to unclaimed pixels
        assign_mask = (dilated > 0) & (cell_map == 0)
        cell_map[assign_mask] = uid
    
    # Create overlay
    overlay = img_rgb.copy()
    
    # Color each cell region with a semi-transparent color
    colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
    ]
    
    for uid in unique_ids:
        cell_mask = cell_map == uid
        color = colors[int(uid) % len(colors)]
        overlay[cell_mask] = (
            overlay[cell_mask].astype(np.float32) * 0.6 +
            np.array(color, dtype=np.float32) * 0.4
        ).astype(np.uint8)
    
    # Draw cell boundaries
    for uid in unique_ids:
        mask = (cell_map == uid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    
    # Draw nuclear boundaries (smaller, darker)
    for uid in unique_ids:
        mask = (inst_map == uid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), 1)
    
    return overlay


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output_dir", default="tiled_output")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (default 4 for 10x->40x)")
    parser.add_argument("--tile_size", type=int, default=256) 
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    
    run_tiled_inference(args.image, args.output_dir, args.scale, args.tile_size, args.overlap)
