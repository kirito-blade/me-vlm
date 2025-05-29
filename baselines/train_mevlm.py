import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, classification_report
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers import Input, Conv3D, BatchNormalization, Add, GlobalAveragePooling3D, Lambda, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from keras import backend as K
from keras.applications import MobileNetV2
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import argparse
import scipy.stats as stats
import pandas as pd
import json
import yaml

# Configuration parameters
IMAGE_ROWS, IMAGE_COLUMNS, IMAGE_DEPTH = 64, 64, 96
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
NUM_CLASSES = 3  # Changed to target number of classes
TEST_SPLIT = 0.2
RANDOM_SEED = 4

# Path configuration - easy to change when swapping datasets
# Training data
ROOT_VIDEO_DIR = '/home/tpei0009/LTX-Video/hq_emotion/'
# Validation data (SAMM dataset paths)
VALIDATION_EMOTION_PATHS = {
    'anger': '/home/tpei0009/MMNet/sam_emo/anger',
    'happiness': '/home/tpei0009/MMNet/sam_emo/happiness',
    'disgust': '/home/tpei0009/MMNet/sam_emo/disgust',
    'sadness': '/home/tpei0009/MMNet/sam_emo/sadness',
    'surprise': '/home/tpei0009/MMNet/sam_emo/surprise',
}
# Additional test dataset paths from train_miex.py
MMEW_DATASET_PATH = '/home/tpei0009/STSTNet/generated5k_mmewAU'
CAS_DATASET_PATH = '/home/tpei0009/MMNet/cas_emo'
# Unified emotion to label mapping - using 3 classes
EMOTION_TO_LABEL = {
    'positive': 0,  # happiness
    'negative': 1,  # anger, disgust, sadness, fear
    'surprise': 2,  # surprise
}

# Mapping from standard emotion categories to our 3-class system
EMOTION_MAPPING = {
    'happiness': 'positive', 
    'anger': 'negative',
    'fear': 'negative',
    'disgust': 'negative',
    'sadness': 'negative',
    'surprise': 'surprise'
}
# Cache file paths
IMAGES_NPY = '/home/tpei0009/micro-expression-recognition/microexpstcnn_images.npy'
LABELS_NPY = '/home/tpei0009/micro-expression-recognition/microexpstcnn_labels.npy'
CATEGORICAL_LABELS_NPY = '/home/tpei0009/micro-expression-recognition/microexpstcnnlabels.npy'
TRAININGSAMPLES_NPY = '/home/tpei0009/micro-expression-recognition/trainingsamples.npy'
# Weights path
WEIGHTS_PATH = "/home/tpei0009/micro-expression-recognition/weights_mevlm/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# Results path for saving plots
RESULTS_DIR = "/home/tpei0009/micro-expression-recognition/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# TensorFlow setup
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Force memory allocation
        tf.config.experimental.set_memory_growth(gpus[0], False)
        # Optional: limit memory to a specific amount
        # tf.config.set_logical_device_configuration(
        #     physical_devices[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        # )
    except RuntimeError as e:
        print(e)

# Force immediate GPU memory allocation
with tf.device('/GPU:0'):
    # Create a large tensor to force allocation
    dummy = tf.random.normal([1000, 1000])
    result = tf.matmul(dummy, dummy)
    print("Test tensor device:", result.device)
    del dummy, result

# Function definitions
def process_image(image_path):
    """Process a single image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def process_directory(directory_path):
    """Process all images in a directory."""
    training_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                processed_image = process_image(image_path)
                training_list.append(processed_image)
    return np.asarray(training_list)

def process_video(video_path, num_frames=IMAGE_DEPTH, resize_shape=(IMAGE_ROWS, IMAGE_COLUMNS)):
    """
    Reads a video file, extracts num_frames frames, resizes them, converts to grayscale.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 0
    for i in range(start_frame, min(start_frame + num_frames, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, resize_shape, interpolation=cv2.INTER_AREA)
        frames.append(resized)
    cap.release()
    frames = np.asarray(frames)
    # Ensure the shape is (num_frames, rows, cols)
    if frames.shape[0] < num_frames:
        # Pad with zeros if not enough frames
        pad_shape = (num_frames - frames.shape[0],) + frames.shape[1:]
        frames = np.concatenate([frames, np.zeros(pad_shape, dtype=frames.dtype)], axis=0)
    return frames

def process_all_videos_with_labels(root_dir, emotion_to_label, num_frames=IMAGE_DEPTH, 
                                 resize_shape=(IMAGE_ROWS, IMAGE_COLUMNS)):
    """
    Process all videos in the directory structure and assign labels.
    """
    training_list = []
    training_labels = []
    for emotion in os.listdir(root_dir):
        emotion_folder = os.path.join(root_dir, emotion)
        if not os.path.isdir(emotion_folder):
            continue
            
        # Map standard emotion to our 3-class system
        mapped_emotion = EMOTION_MAPPING.get(emotion)
        if mapped_emotion is None:
            print(f"Warning: Emotion '{emotion}' not in mapping, skipping.")
            continue
            
        label = emotion_to_label.get(mapped_emotion)
        if label is None:
            print(f"Warning: Mapped emotion '{mapped_emotion}' not in label mapping, skipping.")
            continue
            
        for subdir, dirs, files in os.walk(emotion_folder):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(subdir, file)
                    print(f"Processing {video_path} as label {label} ({emotion} -> {mapped_emotion})")
                    frames = process_video(video_path, num_frames=num_frames, resize_shape=resize_shape)
                    training_list.append(frames)
                    training_labels.append(label)
    return training_list, training_labels

def load_samm_dataset_no_sticker(emotion_paths, emotion_to_label, image_rows=IMAGE_ROWS, 
                               image_columns=IMAGE_COLUMNS, image_depth=IMAGE_DEPTH):
    """
    Load SAMM dataset images.
    Maps standard emotion categories to our 3-class system:
    - happiness -> positive (class 0)
    - anger, fear, disgust, sadness -> negative (class 1)
    - surprise -> surprise (class 2)
    """
    training_list = []
    label_list = []
    
    for emotion, path in emotion_paths.items():
        # Map standard emotion to our 3-class system
        mapped_emotion = EMOTION_MAPPING.get(emotion)
        if mapped_emotion is None:
            print(f"Warning: Emotion '{emotion}' not in mapping, skipping.")
            continue
            
        label = emotion_to_label[mapped_emotion]
        directorylisting = os.listdir(path)
        
        for video in directorylisting:
            videopath = os.path.join(path, video)
            for root, dirs, files in os.walk(videopath):
                for dir1 in dirs:
                    video_path2 = os.path.join(videopath, dir1)
                    try:
                        video_path3 = os.listdir(video_path2)
                    except FileNotFoundError:
                        continue
                    video_path3.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1)
                    frames = []
                    for file in video_path3:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            file_path = os.path.join(video_path2, file)
                            image = cv2.imread(file_path)
                            imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
                            grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                            frames.append(grayimage)
                    while len(frames) < image_depth:
                        frames.append(np.zeros_like(frames[0]))
                    frames = np.asarray(frames)[:image_depth, :, :]
                    videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                    training_list.append(videoarray)
                    label_list.append(label)
                    
    training_list = np.asarray(training_list)
    label_list = np_utils.to_categorical(label_list, len(emotion_to_label))
    # Reshape to (N, 1, rows, cols, depth)
    training_list = np.expand_dims(training_list, axis=1)
    return training_list, label_list

def normalize_data(data):
    """Apply consistent normalization to the data."""
    for i in range(data.shape[0]):
        data[i] = (data[i] - np.mean(data[i])) / (np.std(data[i]) + 1e-8)
    return data

def load_mmew_dataset(root_dir, image_rows=IMAGE_ROWS, image_columns=IMAGE_COLUMNS, image_depth=IMAGE_DEPTH):
    """
    Load MMeW dataset from the specified root directory.
    Directory structure is expected to be:
    root_dir/
        subject_id_mmew_emotion_class/
            0.jpg, 1.jpg, 2.jpg, etc.
    
    Where emotion_class is 0, 1, 2, etc.
    
    The 3-class mapping is:
    0: positive
    1: negative
    2: surprise
    
    Returns:
        training_list: List of image arrays
        label_list: List of labels (categorical)
        emotion_to_label: Dictionary mapping emotion names to label indices
    """
    print(f"Loading MMeW dataset from {root_dir}...", flush=True)
    
    # Define emotion mapping for MMeW dataset
    # Using a consistent 3-class mapping across all datasets
    mmew_emotion_mapping = {
        '0': 'positive',   # Class 0: positive emotions
        '1': 'negative',   # Class 1: negative emotions
        '2': 'surprise',   # Class 2: surprise emotions
    }
    
    # Create label mapping - fixed for 3 classes
    emotion_to_label = {
        'positive': 0,
        'negative': 1,
        'surprise': 2
    }
    
    training_list = []
    label_list = []
    
    # List all subject directories
    subject_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for subject_dir in subject_dirs:
        # Extract emotion class from directory name (e.g., "000106_mmew_0" -> "0")
        parts = subject_dir.split('_')
        if len(parts) < 3:
            print(f"Warning: Directory {subject_dir} does not match expected format, skipping.")
            continue
        
        emotion_class = parts[-1]
        if emotion_class not in mmew_emotion_mapping:
            print(f"Warning: Unknown emotion class {emotion_class} in directory {subject_dir}, skipping.")
            continue
        
        emotion = mmew_emotion_mapping[emotion_class]
        label = emotion_to_label[emotion]
        
        # Get image files in the subject directory
        subject_path = os.path.join(root_dir, subject_dir)
        image_files = [f for f in os.listdir(subject_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else -1)
        
        if len(image_files) < image_depth:
            print(f"Warning: Directory {subject_dir} has only {len(image_files)} images, "
                  f"need at least {image_depth}. Padding with zeros.")
        
        # Load frames
        frames = []
        for i, image_file in enumerate(image_files):
            if i >= image_depth:
                break
                
            image_path = os.path.join(subject_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load image {image_path}, skipping.")
                continue
                
            # Resize and convert to grayscale
            image_resized = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            frames.append(gray_image)
        
        # Pad with zeros if not enough frames
        while len(frames) < image_depth:
            frames.append(np.zeros((image_rows, image_columns), dtype=np.uint8))
        
        # Truncate if too many frames
        frames = frames[:image_depth]
        
        # Convert frames to numpy array
        frames = np.array(frames)
        
        # Roll axes to match expected format
        videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)
        label_list.append(label)
    
    training_list = np.array(training_list)
    num_classes = len(emotion_to_label)
    label_list = np_utils.to_categorical(label_list, num_classes)
    
    # Reshape to (N, 1, rows, cols, depth)
    training_list = np.expand_dims(training_list, axis=1)
    
    print(f"Loaded {len(training_list)} samples with {num_classes} emotion classes.")
    print(f"Data shape: {training_list.shape}")
    print(f"Emotion to label mapping: {emotion_to_label}")
    
    return training_list, label_list, emotion_to_label

def load_validation_dataset(root_dir, emotion_to_label, image_rows=IMAGE_ROWS, 
                           image_columns=IMAGE_COLUMNS, image_depth=IMAGE_DEPTH):
    """
    Load validation dataset (SAMM) from the specified root directory.
    Directory structure is expected to be:
    root_dir/
        emotion_name/
            video_folders/
                frames/
                
    Maps standard emotion categories to training emotions:
    - happiness -> positive (class 0)
    - anger, fear, disgust, sadness -> negative (class 1)
    - surprise -> surprise (class 2)
    """
    print(f"Loading validation dataset from {root_dir}...", flush=True)
    
    validation_list = []
    label_list = []
    
    # Define mapping from validation emotions to our training classes
    validation_to_training_map = {
        'happiness': 'positive',  # Map happiness to class 0
        'anger': 'negative',      # Map anger to class 1
        'fear': 'negative',       # Map fear to class 1
        'disgust': 'negative',    # Map disgust to class 1
        'sadness': 'negative',    # Map sadness to class 1
        'surprise': 'surprise'    # Map surprise to class 2
    }
    
    # Process each emotion directory in validation dataset
    for val_emotion, train_emotion in validation_to_training_map.items():
        # Skip if training doesn't have this emotion class
        if train_emotion not in emotion_to_label:
            print(f"Skipping validation emotion '{val_emotion}' as it maps to '{train_emotion}' which is not in training data")
            continue
            
        # Get the label index for this emotion
        label = emotion_to_label[train_emotion]
        
        # Path to this emotion folder
        emotion_path = os.path.join(root_dir, val_emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: Emotion directory {emotion_path} not found, skipping.")
            continue
            
        # Process video directories
        video_dirs = [d for d in os.listdir(emotion_path) if os.path.isdir(os.path.join(emotion_path, d))]
        
        for video in video_dirs:
            videopath = os.path.join(emotion_path, video)
            
            # Process each sequence in the video directory
            for seq_dir in os.listdir(videopath):
                seq_path = os.path.join(videopath, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                    
                # Get frames
                frame_files = [f for f in os.listdir(seq_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1)
                
                if len(frame_files) < image_depth:
                    # Skip if not enough frames
                    print(f"Warning: Sequence {seq_path} has only {len(frame_files)} frames, "
                          f"need at least {image_depth}. Skipping.")
                    continue
                
                frames = []
                for i, frame_file in enumerate(frame_files):
                    if i >= image_depth:
                        break
                        
                    frame_path = os.path.join(seq_path, frame_file)
                    image = cv2.imread(frame_path)
                    if image is None:
                        print(f"Warning: Failed to load image {frame_path}, skipping.")
                        continue
                        
                    # Resize and convert to grayscale
                    image_resized = cv2.resize(image, (image_rows, image_columns), 
                                              interpolation=cv2.INTER_AREA)
                    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                    frames.append(gray_image)
                
                # Pad with zeros if not enough frames
                while len(frames) < image_depth:
                    frames.append(np.zeros((image_rows, image_columns), dtype=np.uint8))
                
                # Truncate if too many frames
                frames = frames[:image_depth]
                
                # Convert frames to numpy array
                frames = np.array(frames)
                
                # Roll axes to match expected format
                videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                validation_list.append(videoarray)
                label_list.append(label)
    
    validation_list = np.array(validation_list)
    if len(validation_list) == 0:
        raise ValueError("No validation samples could be loaded")
        
    num_classes = len(emotion_to_label)
    label_list = np_utils.to_categorical(label_list, num_classes)
    
    # Reshape to (N, 1, rows, cols, depth)
    validation_list = np.expand_dims(validation_list, axis=1)
    
    print(f"Loaded {len(validation_list)} validation samples.")
    print(f"Validation data shape: {validation_list.shape}")
    
    return validation_list, label_list

def load_cas_test_dataset(root_dir, emotion_to_label, image_rows=IMAGE_ROWS, 
                         image_columns=IMAGE_COLUMNS, image_depth=IMAGE_DEPTH):
    """
    Load CAS test dataset from the specified root directory.
    """
    print(f"Loading CAS test dataset from {root_dir}...", flush=True)
    
    # Define emotion mapping for test dataset
    test_emotion_mapping = {
        'happiness': 'positive',    # Map to class 0
        'repression': 'negative',   # Map to class 1
        'surprise': 'surprise'      # Map to class 2
    }
    
    test_list = []
    label_list = []
    skipped_count = 0
    loaded_count = 0
    
    # Process each emotion directory
    for test_emotion, train_emotion in test_emotion_mapping.items():
        # Skip if training doesn't have this emotion class
        if train_emotion not in emotion_to_label:
            print(f"Skipping test emotion '{test_emotion}' as it maps to '{train_emotion}' which is not in training data")
            continue
            
        # Get the label index for this emotion
        label = emotion_to_label[train_emotion]
        
        # Path to this emotion folder
        emotion_path = os.path.join(root_dir, test_emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: Emotion directory {emotion_path} not found, skipping.")
            continue
            
        print(f"Processing emotion: {test_emotion} -> {train_emotion} (label {label})")
        
        # Process video directories
        video_dirs = [d for d in os.listdir(emotion_path) if os.path.isdir(os.path.join(emotion_path, d))]
        
        for video in video_dirs:
            videopath = os.path.join(emotion_path, video)
            
            # Process each sequence in the video directory
            for seq_dir in os.listdir(videopath):
                seq_path = os.path.join(videopath, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                    
                # Get frames
                frame_files = [f for f in os.listdir(seq_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else -1)
                
                if len(frame_files) == 0:
                    # Skip if no frames found
                    skipped_count += 1
                    continue
                
                frames = []
                for i, frame_file in enumerate(frame_files):
                    if i >= image_depth:
                        break
                        
                    frame_path = os.path.join(seq_path, frame_file)
                    image = cv2.imread(frame_path)
                    if image is None:
                        continue
                        
                    # Resize and convert to grayscale
                    image_resized = cv2.resize(image, (image_rows, image_columns), 
                                              interpolation=cv2.INTER_AREA)
                    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                    frames.append(gray_image)
                
                # Skip if we couldn't load any frames
                if len(frames) == 0:
                    skipped_count += 1
                    continue
                
                # Pad with zeros if not enough frames (similar to SAMM dataset loading)
                while len(frames) < image_depth:
                    frames.append(np.zeros((image_rows, image_columns), dtype=np.uint8))
                    
                # Truncate if too many frames
                frames = frames[:image_depth]
                
                # Convert frames to numpy array
                frames = np.array(frames)
                
                # Roll axes to match expected format
                videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                test_list.append(videoarray)
                label_list.append(label)
                loaded_count += 1
                
                print(f"Loaded sequence {seq_path} with {len([f for f in os.listdir(seq_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])} frames")
    
    test_list = np.array(test_list)
    if len(test_list) == 0:
        raise ValueError("No test samples could be loaded")
        
    num_classes = len(emotion_to_label)
    label_list = np_utils.to_categorical(label_list, num_classes)
    
    # Reshape to (N, 1, rows, cols, depth)
    test_list = np.expand_dims(test_list, axis=1)
    
    print(f"Loaded {loaded_count} test samples, skipped {skipped_count} sequences.")
    print(f"Test data shape: {test_list.shape}")
    
    return test_list, label_list

def residual_block(x, filters, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same'):
    """Create a residual block for ResNet."""
    shortcut = x
    
    # First convolution
    x = Conv3D(filters, kernel_size, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution
    x = Conv3D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection (identity or projection)
    if stride != (1, 1, 1) or shortcut.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut to main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def build_model(input_shape=(1, IMAGE_ROWS, IMAGE_COLUMNS, IMAGE_DEPTH), num_classes=NUM_CLASSES, model_type='original'):
    """Build the selected model architecture."""
    
    if model_type == 'original':
        # Original 3D CNN model
        model = Sequential()
        model.add(Convolution3D(
            32, (3, 3, 3),
            input_shape=input_shape,
            activation='relu',
            padding='same'
        ))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        
    elif model_type == 'resnet3d':
        # ResNet-inspired 3D CNN for video classification
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        
        # ResNet blocks
        # First block group - 64 filters
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        
        # Second block group - 128 filters
        x = residual_block(x, 128, stride=(2, 2, 2))
        x = residual_block(x, 128)
        
        # Third block group - 256 filters (optional, depending on input size)
        x = residual_block(x, 256, stride=(2, 2, 2))
        x = residual_block(x, 256)
        
        # Global pooling and classification
        x = GlobalAveragePooling3D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=x)
        
    elif model_type == 'mobilenet':
        # Using a 2D MobileNetV2 on frame sequences
        # Reshape input to use 2D MobileNetV2 on individual frames
        # This requires reshaping the 3D data to 2D+time approach
        
        # Create a sequence model that processes each frame with MobileNetV2
        inputs = Input(shape=input_shape)
        
        # Reshape to process frames individually (batch_size, frames, h, w, c)
        # Extract the middle frame as representative frame (simplification)
        # For a more comprehensive approach, you'd process multiple frames
        frame = Lambda(lambda x: x[:, 0, :, :, IMAGE_DEPTH//2])(inputs)
        
        # Use MobileNetV2 as feature extractor (input needs to be RGB)
        frame = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=-1), 3, axis=-1))(frame)
        frame = Lambda(lambda x: tf.image.resize(x, (224, 224)))(frame)
        
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # Apply MobileNetV2 to the frame
        x = base_model(frame)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=x)
    
    elif model_type == 'simple3dcnn':
        model = Sequential()
        model.add(Convolution3D(
            16, (3, 3, 3),
            input_shape=input_shape,
            activation='relu',
            padding='same'
        ))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        metrics=['accuracy']
    )
    return model

def plot_training_history(history, save_path):
    """Plot and save training history."""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training plots saved to {save_path}")

# Function to load configuration from a file
def load_config(config_path):
    """Load configuration from a YAML or JSON file."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")
    return config

# Function to create a default config dictionary
def create_default_config(args):
    """Create a default configuration dictionary from arguments."""
    config = {
        "paths": {
            "root_video_dir": args.root_video_dir or ROOT_VIDEO_DIR,
            "weights_path": args.weights_path or WEIGHTS_PATH,
            "results_dir": args.results_dir or RESULTS_DIR,
            "images_npy": args.images_npy or IMAGES_NPY,
            "labels_npy": args.labels_npy or LABELS_NPY,
            "categorical_labels_npy": args.categorical_labels_npy or CATEGORICAL_LABELS_NPY,
            "trainingsamples_npy": args.trainingsamples_npy or TRAININGSAMPLES_NPY
        },
        "training": {
            "batch_size": args.batch_size or BATCH_SIZE,
            "epochs": args.epochs or EPOCHS,
            "learning_rate": args.learning_rate or LEARNING_RATE,
            "test_split": args.test_split or TEST_SPLIT,
            "random_seed": args.random_seed or RANDOM_SEED
        },
        "model": {
            "type": args.model_type or 'original',
            "image_rows": args.image_rows or IMAGE_ROWS,
            "image_columns": args.image_columns or IMAGE_COLUMNS,
            "image_depth": args.image_depth or IMAGE_DEPTH
        },
        "emotions": {}
    }
    
    # Handle emotion paths
    if args.emotion_paths:
        # Parse emotion paths from comma-separated string
        # Format: emotion1=/path/to/emotion1,emotion2=/path/to/emotion2
        emotion_paths = {}
        for item in args.emotion_paths.split(','):
            if '=' in item:
                emotion, path = item.split('=', 1)
                emotion_paths[emotion.strip()] = path.strip()
        config["emotions"]["paths"] = emotion_paths
    else:
        config["emotions"]["paths"] = VALIDATION_EMOTION_PATHS
    
    # Add dataset mode configuration
    config["dataset_mode"] = args.dataset_mode or 'samm_only'
    config["mmew_dataset_path"] = args.mmew_dataset_path or MMEW_DATASET_PATH
    config["cas_dataset_path"] = args.cas_dataset_path or CAS_DATASET_PATH
    
    # Set SAMM dataset path
    if args.samm_dataset_path:
        config["samm_dataset_path"] = args.samm_dataset_path
    else:
        # Use individual emotion paths for SAMM if available, otherwise use default
        config["samm_dataset_path"] = None  # Will use VALIDATION_EMOTION_PATHS
    
    # Create emotion to label mapping from the emotion paths
    config["emotions"]["mapping"] = {emotion: idx for idx, emotion in enumerate(config["emotions"]["paths"].keys())}
    
    return config

def main(config):
    """Main function to train the model with flexible dataset support."""
    # Extract configuration
    model_type = config["model"]["type"]
    dataset_mode = config.get("dataset_mode", "samm_only")
    num_classes = NUM_CLASSES  # Use global setting
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    random_seed = config["training"]["random_seed"]
    learning_rate = config["training"]["learning_rate"]
    test_split = config["training"]["test_split"]
    image_rows = config["model"]["image_rows"]
    image_columns = config["model"]["image_columns"] 
    image_depth = config["model"]["image_depth"]
    test_only = config.get("test_only", False)
    pretrained_weights = config.get("pretrained_weights", None)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if dataset_mode == "mmew_train_samm_val_cas_test":
        # train_miex.py style: Train on MMeW, validate on SAMM, test on CAS
        print("Using MMeW-SAMM-CAS dataset mode (train_miex.py style)...", flush=True)
        
        # Load MMeW dataset for training
        train_data, train_labels, emotion_to_label = load_mmew_dataset(
            config["mmew_dataset_path"],
            image_rows=image_rows,
            image_columns=image_columns,
            image_depth=image_depth
        )
        
        # Update num_classes based on MMeW dataset
        num_classes = len(emotion_to_label)
        print(f"Number of classes detected from MMeW: {num_classes}")
        
        # Load SAMM validation dataset
        if config.get("samm_dataset_path"):
            samm_path = config["samm_dataset_path"]
        else:
            samm_path = '/home/tpei0009/MMNet/sam_emo'  # Default SAMM path
            
        val_data, val_labels = load_validation_dataset(
            samm_path,
            emotion_to_label,
            image_rows=image_rows,
            image_columns=image_columns,
            image_depth=image_depth
        )
        
        # Load CAS test dataset
        test_data, test_labels = load_cas_test_dataset(
            config["cas_dataset_path"],
            emotion_to_label,
            image_rows=image_rows,
            image_columns=image_columns,
            image_depth=image_depth
        )
        
        print(f"Dataset splits - Training (MMeW): {train_data.shape[0]}, "
              f"Validation (SAMM): {val_data.shape[0]}, Test (CAS): {test_data.shape[0]}")
        
        # Normalize all data
        print("Normalizing datasets...", flush=True)
        train_data = normalize_data(train_data)
        val_data = normalize_data(val_data)
        test_data = normalize_data(test_data)
        print("Normalization complete", flush=True)
        
    else:
        # Original SAMM-only mode
        print("Using SAMM-only dataset mode (original)...", flush=True)
        emotion_paths = config["emotions"]["paths"]
        emotion_to_label = config["emotions"]["mapping"]
        num_classes = len(emotion_to_label)
        
        # Load SAMM dataset
        samm_images, samm_labels = load_samm_dataset_no_sticker(
            emotion_paths, emotion_to_label, 
            image_rows=image_rows, image_columns=image_columns, image_depth=image_depth
        )
        
        # Split SAMM dataset into validation and test sets
        print("Splitting SAMM dataset into validation and test sets...", flush=True)
        val_data, test_data, val_labels, test_labels = train_test_split(
            samm_images, samm_labels, test_size=0.5, random_state=random_seed
        )
        
        # Process SAMM validation and test images
        if val_data.shape[-1] != image_depth:
            val_data = np.transpose(val_data, (0, 1, 3, 4, 2))
            test_data = np.transpose(test_data, (0, 1, 3, 4, 2))
        
        # Normalize the data
        val_data = normalize_data(val_data)
        test_data = normalize_data(test_data)
        
        # For SAMM-only mode, we don't have separate training data
        train_data = None
        train_labels = None
    
    if test_only and pretrained_weights:
        # Test-only mode with pretrained weights
        print(f"Running in test-only mode with pretrained weights: {pretrained_weights}", flush=True)
        
        # Build the base model architecture
        print(f"Building {model_type} model architecture with {num_classes} classes...", flush=True)
        with tf.device('/device:GPU:0'):
            model = build_model(
                input_shape=(1, image_rows, image_columns, image_depth),
                num_classes=num_classes,
                model_type=model_type
            )
        
        # Load pretrained weights
        print(f"Loading pretrained weights from {pretrained_weights}...", flush=True)
        model.load_weights(pretrained_weights)
        print("Weights loaded successfully")
        
    else:
        # Training mode
        if dataset_mode == "mmew_train_samm_val_cas_test" and train_data is not None:
            # Train on MMeW, validate on SAMM
            print(f"Building {model_type} model with {num_classes} classes...", flush=True)
            with tf.device('/device:GPU:0'):
                model = build_model(
                    input_shape=(1, image_rows, image_columns, image_depth),
                    num_classes=num_classes,
                    model_type=model_type
                )
            model.summary()
            
            # Handle pretrained weights (if specified but not in test-only mode)
            if pretrained_weights:
                print(f"Loading pretrained weights for fine-tuning: {pretrained_weights}", flush=True)
                model.load_weights(pretrained_weights)
            
            # Setup callbacks
            filepath = config["paths"]["weights_path"]
            checkpoint = ModelCheckpoint(
                filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
            )
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
            callbacks_list = [checkpoint, early_stopping, reduce_lr]
            
            # Train model using SAMM for validation
            print("Starting model training with SAMM validation...", flush=True)
            history = model.fit(
                train_data, train_labels,
                validation_data=(val_data, val_labels),
                callbacks=callbacks_list, batch_size=batch_size, epochs=epochs, shuffle=True
            )
            print("Finished model training!", flush=True)
            
            # Plot training history
            print("Generating training plots...", flush=True)
            plot_path = os.path.join(config["paths"]["results_dir"], f"training_history_{timestamp}.png")
            plot_training_history(history, plot_path)
            
        else:
            print("Training mode requires MMeW dataset. Use dataset_mode='mmew_train_samm_val_cas_test' for training.")
            return None
    
    # Evaluation - this happens in both training and test-only modes
    print("Evaluating on test set...", flush=True)
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    
    # Also evaluate on validation set for comparison
    val_loss, val_accuracy = model.evaluate(val_data, val_labels)
    print(f"Validation accuracy: {val_accuracy:.4f}, Validation loss: {val_loss:.4f}")
    
    # Generate confusion matrix
    print("Generating confusion matrix...", flush=True)
    predictions = model.predict(test_data)
    predictions_labels = np.argmax(predictions, axis=1)
    test_labels_argmax = np.argmax(test_labels, axis=1)
    cm = confusion_matrix(test_labels_argmax, predictions_labels)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate F1 scores and other metrics
    print("Calculating F1 scores and other metrics...", flush=True)
    f1_per_class = f1_score(test_labels_argmax, predictions_labels, average=None)
    f1_weighted = f1_score(test_labels_argmax, predictions_labels, average='weighted')
    f1_macro = f1_score(test_labels_argmax, predictions_labels, average='macro')
    
    print(f"F1 score per class: {f1_per_class}")
    print(f"Weighted F1 score: {f1_weighted}")
    print(f"Macro F1 score: {f1_macro}")
    
    # Detailed classification report
    print("Classification Report:")
    if dataset_mode == "mmew_train_samm_val_cas_test":
        target_names = list(emotion_to_label.keys())
    else:
        target_names = list(config["emotions"]["mapping"].keys())
    
    try:
        report = classification_report(test_labels_argmax, predictions_labels, target_names=target_names)
        print(report)
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        # Fall back to using indices as labels
        report = classification_report(test_labels_argmax, predictions_labels)
        print(report)

    # T-test: Compare model performance against chance level
    class_probabilities = model.predict(test_data)
    max_probabilities = np.max(class_probabilities, axis=1)
    chance_level = 1/num_classes
    t_stat, p_value = stats.ttest_1samp(max_probabilities, chance_level)
    print(f"T-test against chance level: t={t_stat:.4f}, p={p_value:.4f}")
    print(f"Mean confidence: {np.mean(max_probabilities):.4f}, Chance level: {chance_level:.4f}")
    
    # Store per-class metrics for statistical tests
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels_argmax, predictions_labels
    )
    
    # ANOVA: Compare F1 scores across different emotion categories
    results_df = pd.DataFrame({
        'emotion': target_names,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'support': support
    })
    print("Per-class performance:")
    print(results_df)
    
    # One-way ANOVA across emotion categories
    try:
        class_data = []
        for i in range(num_classes):
            mask = test_labels_argmax == i
            if np.sum(mask) > 0:
                class_data.append(class_probabilities[mask, i])
        
        if len(class_data) > 1:
            f_stat, p_anova = stats.f_oneway(*class_data)
            print(f"ANOVA across emotion categories: F={f_stat:.4f}, p={p_anova:.4f}")
        else:
            print("Not enough groups for ANOVA test.")
    except Exception as e:
        print(f"Error performing ANOVA: {e}")

    # Save results to file
    results_path = os.path.join(config["paths"]["results_dir"], f"metrics_results_{dataset_mode}_{timestamp}.txt")
    with open(results_path, 'w') as f:
        f.write(f"Dataset mode: {dataset_mode}\n")
        f.write(f"F1 score per class: {f1_per_class}\n")
        f.write(f"Weighted F1 score: {f1_weighted}\n")
        f.write(f"Macro F1 score: {f1_macro}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels_argmax, predictions_labels, target_names=target_names))
        f.write(f"\nT-test against chance level: t={t_stat:.4f}, p={p_value:.4f}\n")
        f.write(f"Mean confidence: {np.mean(max_probabilities):.4f}, Chance level: {chance_level:.4f}\n")
        try:
            f.write("\nANOVA across emotion categories:\n")
            f.write(f"F={f_stat:.4f}, p={p_anova:.4f}\n")
        except:
            f.write("\nANOVA could not be performed\n")
        f.write("\nPer-class performance:\n")
        f.write(results_df.to_string())
    print(f"Metrics saved to {results_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(config["paths"]["results_dir"], f"confusion_matrix_{dataset_mode}_{timestamp}.txt")
    np.savetxt(cm_path, cm, fmt='%d', delimiter=',')
    print(f"Confusion matrix saved to {cm_path}")
    
    if not test_only:
        # Save final model (only if training was performed)
        model_path = os.path.join(config["paths"]["results_dir"], f"final_model_{dataset_mode}_{timestamp}.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train micro-expression recognition model.')
    # Mode selection
    parser.add_argument('--test_only', action='store_true', help='Run in test-only mode (no training)')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights to load')
    
    # Model and training parameters
    parser.add_argument('--model_type', type=str, 
                        choices=['original', 'resnet3d', 'mobilenet', 'simple3dcnn'], 
                        help='Model type to use.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--test_split', type=float, help='Test split ratio')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    
    # Image dimensions
    parser.add_argument('--image_rows', type=int, help='Image height')
    parser.add_argument('--image_columns', type=int, help='Image width')
    parser.add_argument('--image_depth', type=int, help='Image depth (frames)')
    
    # Paths
    parser.add_argument('--root_video_dir', type=str, help='Root directory for training videos')
    parser.add_argument('--weights_path', type=str, help='Path to save model weights')
    parser.add_argument('--results_dir', type=str, help='Directory to save results')
    parser.add_argument('--images_npy', type=str, help='Path to cache image data')
    parser.add_argument('--labels_npy', type=str, help='Path to cache label data')
    parser.add_argument('--categorical_labels_npy', type=str, help='Path to cache categorical labels')
    parser.add_argument('--trainingsamples_npy', type=str, help='Path to cache training samples count')
    
    # Emotion configuration
    parser.add_argument('--emotion_paths', type=str, 
                        help='Comma-separated list of emotion=path pairs, e.g., "anger=/path/to/anger,happiness=/path/to/happiness"')
    
    # Dataset selection and testing mode (from train_miex.py functionality)
    parser.add_argument('--dataset_mode', type=str, 
                        choices=['samm_only', 'mmew_train_samm_val_cas_test', 'custom'], 
                        default='samm_only',
                        help='Dataset mode: samm_only (original), mmew_train_samm_val_cas_test (train_miex style), or custom')
    parser.add_argument('--mmew_dataset_path', type=str, default=MMEW_DATASET_PATH,
                        help='Path to MMeW training dataset')
    parser.add_argument('--cas_dataset_path', type=str, default=CAS_DATASET_PATH,
                        help='Path to CAS test dataset')
    parser.add_argument('--samm_dataset_path', type=str, 
                        help='Path to SAMM validation dataset (defaults to VALIDATION_EMOTION_PATHS)')
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config(args)
    
    # Add test_only and pretrained_weights to config
    config["test_only"] = args.test_only
    if args.pretrained_weights:
        config["pretrained_weights"] = args.pretrained_weights
    
    # Optionally save the configuration for reproducibility
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_save_path = os.path.join(config["paths"]["results_dir"], f"config_{timestamp}.json")
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_save_path}")
    
    main(config)