import os
import pandas as pd
import numpy as np
import cv2
import torch
from deepface import DeepFace
# from deepface.extendedmodels import Emotion, Age, Gender, Race
from tqdm import tqdm
import time
import json

# Enhanced CUDA configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Speed up CUDA operations
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

def analyze_face_details(frame, detector_backend='retinaface'):
    """
    Analyze a face in detail using DeepFace
    """
    try:
        # Direct approach - analyze image in one step
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['age', 'gender', 'race', 'emotion'],
            detector_backend=detector_backend,
            enforce_detection=False,
            silent=True
        )
        
        # Handle multiple faces if detected
        if not analysis:
            return None
            
        # If multiple faces detected, choose the one with highest confidence
        if isinstance(analysis, list) and len(analysis) > 0:
            # Find face with max race probability as an indicator of confidence
            max_prob = -1
            best_face = None
            
            for face in analysis:
                race_dict = face.get('race', {})
                for label, prob in race_dict.items():
                    if prob > max_prob:
                        max_prob = prob
                        best_face = face
            
            if best_face:
                # Extract dominant labels for each attribute
                gender_label = get_dominant_label(best_face.get('gender', {}))
                race_label = get_dominant_label(best_face.get('race', {}))
                emotion_label = get_dominant_label(best_face.get('emotion', {}))
                
                # Create processed result with dominant labels
                processed_face = {
                    'age': best_face.get('age', 0),
                    'gender': gender_label,
                    'dominant_race': race_label,
                    'dominant_emotion': emotion_label,
                    'race': best_face.get('race', {}),
                    'emotion': best_face.get('emotion', {})
                }
                
                result = {
                    'demographic': processed_face,
                    'facial_area': best_face.get('region', {}),
                    'confidence': max_prob
                }
                return result
        
        # Single face case
        elif isinstance(analysis, dict):
            # Extract dominant labels for each attribute
            gender_label = get_dominant_label(analysis.get('gender', {}))
            race_label = get_dominant_label(analysis.get('race', {}))
            emotion_label = get_dominant_label(analysis.get('emotion', {}))
            
            # Create processed result with dominant labels
            processed_face = {
                'age': analysis.get('age', 0),
                'gender': gender_label,
                'dominant_race': race_label,
                'dominant_emotion': emotion_label,
                'race': analysis.get('race', {}),
                'emotion': analysis.get('emotion', {})
            }
            
            result = {
                'demographic': processed_face,
                'facial_area': analysis.get('region', {}),
                'confidence': max([v for v in analysis.get('race', {}).values()] or [0])
            }
            return result
            
        return None
    
    except Exception as e:
        print(f"Error in face analysis: {e}")
        return None

def get_dominant_label(attribute_dict):
    """
    Extract the dominant label (highest probability) from an attribute dictionary
    """
    if not attribute_dict or not isinstance(attribute_dict, dict):
        return "Unknown"
        
    max_prob = -1
    max_label = "Unknown"
    
    for label, prob in attribute_dict.items():
        if isinstance(prob, (int, float)) and prob > max_prob:
            max_prob = prob
            max_label = label
            
    return max_label

def extract_video_frames(video_path, sampling_rate=0.1):
    """
    Extract frames from a video at the specified sampling rate.
    
    Args:
        video_path (str): Path to the video file
        sampling_rate (float): Fraction of frames to extract (0.0-1.0)
        
    Returns:
        list: List of frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_to_extract = max(1, int(total_frames * sampling_rate))
    step = max(1, int(total_frames / frames_to_extract))
    
    frames = []
    frame_indices = []
    
    for frame_idx in tqdm(range(0, total_frames, step), desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frames.append(frame)
        frame_indices.append(frame_idx)
    
    cap.release()
    return frames, frame_indices, fps, total_frames

def analyze_video_frames(video_path, output_dir=None, sample_rate=0.1):
    """
    Analyze video frames for facial attributes using DeepFace.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Directory to save the output
        sample_rate (float, optional): Fraction of frames to analyze
        
    Returns:
        dict: Analysis results
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return None
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Extract frames from the video
    frames, frame_indices, fps, total_frames = extract_video_frames(video_path, sample_rate)
    
    if not frames:
        print(f"No frames extracted from {video_path}")
        return None
    
    # Configure optimal backend for GPU if available
    detector_backend = 'retinaface' if torch.cuda.is_available() else 'opencv'
    
    # Analyze each frame
    print(f"Analyzing {len(frames)} frames from {video_path}")
    frame_results = []
    
    for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
        print(f"Analyzing frame {i+1}/{len(frames)} (Frame #{frame_idx})")
        
        # Pre-process frame with CUDA if available
        if torch.cuda.is_available():
            try:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device)
                frame_tensor = frame_tensor / 255.0
                frame = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
            except Exception as e:
                print(f"Error in CUDA preprocessing: {e}")
        
        # Analyze face in the frame
        face_analysis = analyze_face_details(frame, detector_backend)
        
        if face_analysis:
            result = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps if fps > 0 else 0,
                'analysis': face_analysis
            }
            frame_results.append(result)
        
        # Periodically clear CUDA cache
        if torch.cuda.is_available() and i % 10 == 0:
            torch.cuda.empty_cache()
    
    # Calculate overall demographics
    demographics = {
        'age': [],
        'gender': {'Man': 0, 'Woman': 0},
        'race': {
            'asian': 0, 'indian': 0, 'black': 0, 
            'white': 0, 'middle eastern': 0, 'latino hispanic': 0
        },
        'emotion': {
            'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
            'sad': 0, 'surprise': 0, 'neutral': 0
        },
        'frames_analyzed': len(frame_results)
    }
    
    # Extract facial landmarks for tracking
    landmark_data = []
    
    for result in frame_results:
        if 'analysis' in result and 'demographic' in result['analysis']:
            demo = result['analysis']['demographic']
            
            # Extract age
            if 'age' in demo:
                demographics['age'].append(demo['age'])
            
            # Extract gender
            if 'gender' in demo:
                gender = demo['gender']
                demographics['gender'][gender] += 1
            
            # Extract race
            if 'dominant_race' in demo:
                race = demo['dominant_race']
                demographics['race'][race] += 1
            
            # Extract emotion
            if 'dominant_emotion' in demo:
                emotion = demo['dominant_emotion']
                demographics['emotion'][emotion] += 1
            
            # Extract landmarks for CSV
            if 'landmarks' in result['analysis'] and result['analysis']['landmarks']:
                landmarks = result['analysis']['landmarks']
                landmark_row = {
                    'frame': result['frame_idx'],
                    'timestamp': result['timestamp'],
                    'confidence': result['analysis'].get('confidence', 0)
                }
                
                # Add each landmark point
                for key, (x, y) in landmarks.items():
                    landmark_row[f"{key}_x"] = x
                    landmark_row[f"{key}_y"] = y
                
                # Add emotion probabilities
                if 'emotion' in demo:
                    for emotion, score in demo['emotion'].items():
                        landmark_row[f"emotion_{emotion}"] = score
                
                landmark_data.append(landmark_row)
    
    # Calculate average demographics
    if demographics['frames_analyzed'] > 0:
        demographics['avg_age'] = sum(demographics['age']) / len(demographics['age']) if demographics['age'] else 0
        demographics['dominant_gender'] = max(demographics['gender'].items(), key=lambda x: x[1])[0] if any(demographics['gender'].values()) else "Unknown"
        demographics['dominant_race'] = max(demographics['race'].items(), key=lambda x: x[1])[0] if any(demographics['race'].values()) else "Unknown"
        demographics['dominant_emotion'] = max(demographics['emotion'].items(), key=lambda x: x[1])[0] if any(demographics['emotion'].values()) else "Unknown"
    
    # Create and save CSV file with facial landmarks and tracking data
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if landmark_data:
        landmark_df = pd.DataFrame(landmark_data)
        csv_path = os.path.join(output_dir, f"{video_name}_landmarks.csv")
        landmark_df.to_csv(csv_path, index=False)
    else:
        csv_path = None
    
    # Create overall results
    processing_time = time.time() - start_time
    
    results = {
        'video_path': video_path,
        'video_info': {
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0
        },
        'frames_analyzed': len(frames),
        'faces_detected': len(frame_results),
        'csv_path': csv_path,
        'demographics': demographics,
        'processing_time': processing_time
    }
    
    # Calculate bitrate and size
    try:
        size_bytes = os.path.getsize(video_path)
        results['video_info']['size_mb'] = size_bytes / 1024 / 1024
        duration = results['video_info']['duration']
        results['video_info']['bitrate_kbps'] = (size_bytes * 8) / duration / 1000 if duration > 0 else 0
    except Exception as e:
        print(f"Error calculating video size/bitrate: {e}")
    
    # Save combined results as JSON
    json_path = os.path.join(output_dir, f"{video_name}_analysis.json")
    with open(json_path, 'w') as f:
        # Use a custom encoder for numpy values
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        json.dump(results, f, cls=NpEncoder, indent=2)
    
    print(f"Analysis completed in {processing_time:.2f} seconds. Results saved to {json_path}")
    
    return results

def process_all_videos(directory, output_dir=None, recursive=True, sample_rate=0.1):
    """
    Process all videos in a directory and its subdirectories.
    
    Args:
        directory (str): Directory containing videos
        output_dir (str, optional): Output directory
        recursive (bool, optional): Search subdirectories recursively
        sample_rate (float, optional): Frame sampling rate for analysis
        
    Returns:
        list: List of analysis results for each video
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return []
    
    if output_dir is None:
        output_dir = os.path.join(directory, "deepface_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    videos = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    videos.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                videos.append(os.path.join(directory, file))
    
    if not videos:
        print(f"No videos found in {directory}")
        return []
    
    results = []
    
    print(f"Processing {len(videos)} videos...")
    
    for i, video in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] Processing {video}")
        video_name = os.path.splitext(os.path.basename(video))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        result = analyze_video_frames(video, video_output_dir, sample_rate)
        results.append(result)
        
        # Clear CUDA cache between videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA memory after processing video: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract facial features and demographics from videos using DeepFace")
    parser.add_argument("--input", "-i", help="Input video file or directory")
    parser.add_argument("--output", "-o", help="Output directory for analysis results")
    parser.add_argument("--recursive", "-r", action="store_true", default=True, help="Process directories recursively")
    parser.add_argument("--sample-rate", "-s", type=float, default=0.1, help="Sampling rate for frame analysis (0.0-1.0)")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Force CPU if requested
    if args.no_cuda and torch.cuda.is_available():
        print("CUDA is available but disabled by user request")
        torch.cuda.is_available = lambda: False
        device = torch.device("cpu")
    
    # Monitor GPU memory usage
    if torch.cuda.is_available():
        print(f"Initial CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Use the same directory structure as before
    root_video_dir = '/home/tpei0009/LTX-Video/hq_emotion/'
    if args.input:
        root_video_dir = args.input
    
    output_dir = os.path.join(root_video_dir, 'deepface_analysis')
    if args.output:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all videos
    total_start_time = time.time()
    
    # Process videos with os.walk like in video_quality_metrics.py
    results = []
    for subdir, dirs, files in os.walk(root_video_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(subdir, file)
                print(f"Processing {video_path}")
                
                # Create output directory for this video
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                video_output_dir = os.path.join(output_dir, video_name)
                
                # Analyze the video
                result = analyze_video_frames(video_path, video_output_dir, args.sample_rate)
                results.append(result)
                
                # Clear CUDA cache between videos
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"CUDA memory after processing video: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    total_time = time.time() - total_start_time
    
    # Print summary of results
    print(f"\nProcessed {len(results)} videos in {total_time/60:.2f} minutes.")
    
    # Calculate averages for demographic data
    if results:
        # Initialize counters
        total_age = 0
        gender_counts = {'Man': 0, 'Woman': 0}
        race_counts = {
            'asian': 0, 'indian': 0, 'black': 0, 
            'white': 0, 'middle eastern': 0, 'latino hispanic': 0
        }
        emotion_counts = {
            'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
            'sad': 0, 'surprise': 0, 'neutral': 0
        }
        
        # Count valid results
        valid_results = 0
        
        for res in results:
            if res and 'demographics' in res and res['demographics']:
                valid_results += 1
                
                # Age
                if 'avg_age' in res['demographics']:
                    total_age += res['demographics']['avg_age']
                
                # Gender - accumulate the most dominant gender for each video
                if 'dominant_gender' in res['demographics']:
                    gender_counts[res['demographics']['dominant_gender']] += 1
                
                # Race - accumulate the most dominant race for each video
                if 'dominant_race' in res['demographics']:
                    race_counts[res['demographics']['dominant_race']] += 1
                
                # Emotion - accumulate the most dominant emotion for each video
                if 'dominant_emotion' in res['demographics']:
                    emotion_counts[res['demographics']['dominant_emotion']] += 1
        
        if valid_results > 0:
            # Print demographic summaries
            print(f"\nDemographic Analysis Summary:")
            print(f"Average age across all videos: {total_age / valid_results:.2f}")
            print(f"Gender distribution: {gender_counts}")
            print(f"Race distribution: {race_counts}")
            print(f"Emotion distribution: {emotion_counts}")
            
            # Determine most common categories
            most_common_gender = max(gender_counts.items(), key=lambda x: x[1])[0] if any(gender_counts.values()) else "Unknown"
            most_common_race = max(race_counts.items(), key=lambda x: x[1])[0] if any(race_counts.values()) else "Unknown"
            most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if any(emotion_counts.values()) else "Unknown"
            
            print(f"\nMost common gender: {most_common_gender}")
            print(f"Most common race/ethnicity: {most_common_race}")
            print(f"Most common emotion: {most_common_emotion}")
            
            # Save aggregate results
            aggregate_results = {
                'total_videos': len(results),
                'valid_results': valid_results,
                'avg_age': total_age / valid_results if valid_results > 0 else 0,
                'gender_counts': gender_counts,
                'race_counts': race_counts,
                'emotion_counts': emotion_counts,
                'most_common_gender': most_common_gender,
                'most_common_race': most_common_race,
                'most_common_emotion': most_common_emotion,
                'processing_time_minutes': total_time / 60
            }
            
            with open(os.path.join(output_dir, 'aggregate_results.json'), 'w') as f:
                json.dump(aggregate_results, f, indent=2)
            
            print(f"\nAggregate results saved to {os.path.join(output_dir, 'aggregate_results.json')}")
    else:
        print("No videos processed.") 