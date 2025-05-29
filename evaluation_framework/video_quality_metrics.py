import os
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import piq
import csv

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CSV header definition
csv_header = ["video_path", "psnr_mean", "ssim_mean", "tv_mean", "brisque_mean", "clip_iqa_mean", "num_frames", "size_mb", "duration_s", "bitrate_kbps"]

# Function to get CSV filename for a specific root directory
def get_csv_filename(root_dir):
    # Extract the directory name from the path for unique filename
    dir_name = os.path.basename(os.path.normpath(root_dir))
    return f"video_metrics_{dir_name}.csv"

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def compute_tv(img):
    # Convert numpy array to torch tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions and move to device
    tv_value = piq.total_variation(img_tensor).item()
    return tv_value

def compute_brisque(img):
    # Convert numpy array to torch tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions and move to device
    brisque_value = piq.brisque(img_tensor).item()
    return brisque_value

def compute_clip_iqa(img):
    # Convert numpy array to torch tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions and move to device
    clip_model = piq.CLIPIQA().to(device)
    clip_value = clip_model(img_tensor).item()
    return clip_value

def compute_video_metrics(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate NR metrics for first frame
    tv_list = [compute_tv(prev_frame_gray)]
    brisque_list = [compute_brisque(prev_frame_gray)]
    clip_iqa_list = [compute_clip_iqa(prev_frame_gray)]
    
    psnr_list = []
    ssim_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # FR metrics
        psnr_val = compute_psnr(prev_frame_gray, frame_gray)
        ssim_val = ssim(prev_frame_gray, frame_gray)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        # NR metrics
        tv_list.append(compute_tv(frame_gray))
        brisque_list.append(compute_brisque(frame_gray))
        clip_iqa_list.append(compute_clip_iqa(frame_gray))
        
        prev_frame_gray = frame_gray
        
    cap.release()
    
    # Get size, duration, bitrate
    size_bytes = os.path.getsize(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    bitrate = (size_bytes * 8) / duration / 1000 if duration > 0 else 0  # kbps
    
    return {
        'psnr_mean': np.mean(psnr_list) if psnr_list else None,
        'ssim_mean': np.mean(ssim_list) if ssim_list else None,
        'tv_mean': np.mean(tv_list) if tv_list else None,
        'brisque_mean': np.mean(brisque_list) if brisque_list else None,
        'clip_iqa_mean': np.mean(clip_iqa_list) if clip_iqa_list else None,
        'num_frames': len(psnr_list) + 1,
        'size_mb': size_bytes / 1024 / 1024,
        'duration_s': duration,
        'bitrate_kbps': bitrate
    }

def process_all_videos_in_dir(root_dir, csv_filename):
    results = []
    
    # Create CSV file with header
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(subdir, file)
                print(f"Processing {video_path}")
                metrics = compute_video_metrics(video_path)
                if metrics:
                    # Add video path to metrics
                    metrics['video'] = video_path
                    results.append(metrics)
                    
                    # Write to CSV as we go
                    with open(csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            video_path,
                            metrics['psnr_mean'] if metrics['psnr_mean'] is not None else "N/A",
                            metrics['ssim_mean'] if metrics['ssim_mean'] is not None else "N/A",
                            metrics['tv_mean'] if metrics['tv_mean'] is not None else "N/A",
                            metrics['brisque_mean'] if metrics['brisque_mean'] is not None else "N/A",
                            metrics['clip_iqa_mean'] if metrics['clip_iqa_mean'] is not None else "N/A",
                            metrics['num_frames'],
                            metrics['size_mb'],
                            metrics['duration_s'],
                            metrics['bitrate_kbps']
                        ])
    
    return results

# Define video directories to process
root_video_dirs = [
    '/home/tpei0009/LTX-Video/hq_emotion/',
    # Add more directories as needed
]

# Process each directory
for root_video_dir in root_video_dirs:
    if not os.path.exists(root_video_dir):
        print(f"Warning: Directory {root_video_dir} does not exist, skipping")
        continue
        
    # Get CSV filename for this directory
    csv_filename = get_csv_filename(root_video_dir)
    print(f"Processing videos in {root_video_dir}")
    print(f"Results will be saved to {csv_filename}")
    
    # Process videos and get results
    results = process_all_videos_in_dir(root_video_dir, csv_filename)

    # Print per-video metrics
    for res in results:
        print(f"{res['video']}: Size={res['size_mb']:.2f} MB, Duration={res['duration_s']:.2f} s, Bitrate={res['bitrate_kbps']:.2f} kbps, "
              f"PSNR={res['psnr_mean']:.2f if res['psnr_mean'] is not None else 'N/A'}, "
              f"SSIM={res['ssim_mean']:.4f if res['ssim_mean'] is not None else 'N/A'}, "
              f"TV={res['tv_mean']:.4f}, "
              f"BRISQUE={res['brisque_mean']:.4f}, CLIP-IQA={res['clip_iqa_mean']:.4f}, Frames={res['num_frames']}")

    # Compute and print averages
    if results:
        avg_size = np.mean([r['size_mb'] for r in results])
        avg_duration = np.mean([r['duration_s'] for r in results])
        avg_bitrate = np.mean([r['bitrate_kbps'] for r in results])
        avg_psnr = np.mean([r['psnr_mean'] for r in results if r['psnr_mean'] is not None])
        avg_ssim = np.mean([r['ssim_mean'] for r in results if r['ssim_mean'] is not None])
        avg_tv = np.mean([r['tv_mean'] for r in results if r['tv_mean'] is not None])
        avg_brisque = np.mean([r['brisque_mean'] for r in results if r['brisque_mean'] is not None])
        avg_clip_iqa = np.mean([r['clip_iqa_mean'] for r in results if r['clip_iqa_mean'] is not None])
        
        print(f"Total average Size: {avg_size:.2f} MB")
        print(f"Total average Duration: {avg_duration:.2f} s")
        print(f"Total average Bitrate: {avg_bitrate:.2f} kbps")
        print(f"Total average PSNR: {avg_psnr:.2f}")
        print(f"Total average SSIM: {avg_ssim:.4f}")
        print(f"Total average TV: {avg_tv:.4f}")
        print(f"Total average BRISQUE: {avg_brisque:.4f}")
        print(f"Total average CLIP-IQA: {avg_clip_iqa:.4f}")
        
        # Write averages to a summary CSV
        summary_csv = f"summary_{csv_filename}"
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["directory", root_video_dir])
            writer.writerow(["avg_size_mb", avg_size])
            writer.writerow(["avg_duration_s", avg_duration])
            writer.writerow(["avg_bitrate_kbps", avg_bitrate])
            writer.writerow(["avg_psnr", avg_psnr])
            writer.writerow(["avg_ssim", avg_ssim])
            writer.writerow(["avg_tv", avg_tv])
            writer.writerow(["avg_brisque", avg_brisque])
            writer.writerow(["avg_clip_iqa", avg_clip_iqa])
            writer.writerow(["total_videos", len(results)])
        
        print(f"Summary metrics saved to {summary_csv}")
    else:
        print("No videos processed.")