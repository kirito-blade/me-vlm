# train_flipped.py - Train on test dataset and test on SAMM dataset
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import argparse
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers import Input, Conv3D, BatchNormalization, Add, GlobalAveragePooling3D, Lambda, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from keras.applications import MobileNetV2
from keras import backend as K

# Configuration parameters
IMAGE_ROWS, IMAGE_COLUMNS, IMAGE_DEPTH = 64, 64, 2
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Path configuration - FLIPPED compared to original
TRAINING_DATASET_PATH = '/home/tpei0009/MMNet/cas_emo'  # Previously test dataset, now training
RESULTS_DIR = "/home/tpei0009/micro-expression-recognition/results_cas_to_samm_3"
WEIGHTS_PATH = "/home/tpei0009/micro-expression-recognition/weights_flipped/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
TEST_DATASET_PATH = '/home/tpei0009/MMNet/sam_emo'  # Previously training dataset, now test

os.makedirs(RESULTS_DIR, exist_ok=True)

# TensorFlow setup
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def normalize_data(data):
    for i in range(data.shape[0]):
        data[i] = (data[i] - np.mean(data[i])) / (np.std(data[i]) + 1e-8)
    return data

def residual_block(x, filters, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same'):
    shortcut = x
    
    x = Conv3D(filters, kernel_size, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv3D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    
    if stride != (1, 1, 1) or shortcut.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def build_model(input_shape=(1, IMAGE_ROWS, IMAGE_COLUMNS, IMAGE_DEPTH), num_classes=3, model_type='original'):
    if model_type == 'original':
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
        inputs = Input(shape=input_shape)
        
        x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        
        x = residual_block(x, 128, stride=(2, 2, 2))
        x = residual_block(x, 128)
        
        x = residual_block(x, 256, stride=(2, 2, 2))
        x = residual_block(x, 256)
        
        x = GlobalAveragePooling3D()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=x)
    
    elif model_type == 'mobilenet':
        inputs = Input(shape=input_shape)
        frame = Lambda(lambda x: x[:, 0, :, :, IMAGE_DEPTH//2])(inputs)
        frame = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=-1), 3, axis=-1))(frame)
        frame = Lambda(lambda x: tf.image.resize(x, (224, 224)))(frame)
        
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training plots saved to {save_path}")

def load_training_dataset(root_dir, image_rows=IMAGE_ROWS, 
                         image_columns=IMAGE_COLUMNS, image_depth=IMAGE_DEPTH):
    """
    Load training dataset (previously test dataset) from the specified root directory.
    Expected directory structure:
    root_dir/
        happiness/
            video_folders/
                frames/
        repression/
            video_folders/
                frames/
        surprise/
            video_folders/
                frames/
    """
    print(f"Loading training dataset from {root_dir}...", flush=True)
    
    training_list = []
    label_list = []
    
    # Define emotion mapping
    emotion_mapping = {
        'happiness': 0,    # positive emotion
        'repression': 1,   # negative emotion
        'surprise': 2      # surprise
    }
    
    emotion_to_label = {
        'positive': 0,
        'negative': 1,
        'surprise': 2
    }
    
    # Process each emotion directory
    for emotion, label in emotion_mapping.items():
        emotion_path = os.path.join(root_dir, emotion)
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
                
                # Skip if we couldn't load enough frames
                if len(frames) < image_depth:
                    continue
                    
                # Truncate if too many frames
                frames = frames[:image_depth]
                
                # Convert frames to numpy array
                frames = np.array(frames)
                
                # Roll axes to match expected format
                videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                training_list.append(videoarray)
                label_list.append(label)
    
    training_list = np.array(training_list)
    if len(training_list) == 0:
        raise ValueError("No training samples could be loaded")
        
    num_classes = len(emotion_to_label)
    label_list = np_utils.to_categorical(label_list, num_classes)
    
    # Reshape to (N, 1, rows, cols, depth)
    training_list = np.expand_dims(training_list, axis=1)
    
    print(f"Loaded {len(training_list)} training samples.")
    print(f"Training data shape: {training_list.shape}")
    
    return training_list, label_list, emotion_to_label

def load_test_dataset(root_dir, emotion_to_label, image_rows=IMAGE_ROWS, 
                     image_columns=IMAGE_COLUMNS, image_depth=IMAGE_DEPTH):
    """
    Load test dataset (previously training dataset/SAMM) from the specified root directory.
    """
    print(f"Loading test dataset from {root_dir}...", flush=True)
    
    # Define mapping from SAMM emotions to our 3 classes
    samm_emotion_mapping = {
        'happiness': 'positive',  # Map happiness to class 0
        'anger': 'negative',      # Map anger to class 1
        'fear': 'negative',       # Map fear to class 1
        'disgust': 'negative',    # Map disgust to class 1
        'sadness': 'negative',    # Map sadness to class 1
        'surprise': 'surprise'    # Map surprise to class 2
    }
    
    test_list = []
    label_list = []
    
    # Process each emotion directory in test dataset
    for samm_emotion, train_emotion in samm_emotion_mapping.items():
        # Skip if training doesn't have this emotion class
        if train_emotion not in emotion_to_label:
            print(f"Skipping test emotion '{samm_emotion}' as it maps to '{train_emotion}' which is not in training data")
            continue
            
        # Get the label index for this emotion
        label = emotion_to_label[train_emotion]
        
        # Path to this emotion folder
        emotion_path = os.path.join(root_dir, samm_emotion)
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
                
                # Skip if we couldn't load enough frames
                if len(frames) < image_depth:
                    continue
                    
                # Truncate if too many frames
                frames = frames[:image_depth]
                
                # Convert frames to numpy array
                frames = np.array(frames)
                
                # Roll axes to match expected format
                videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
                test_list.append(videoarray)
                label_list.append(label)
    
    test_list = np.array(test_list)
    if len(test_list) == 0:
        raise ValueError("No test samples could be loaded")
        
    num_classes = len(emotion_to_label)
    label_list = np_utils.to_categorical(label_list, num_classes)
    
    # Reshape to (N, 1, rows, cols, depth)
    test_list = np.expand_dims(test_list, axis=1)
    
    print(f"Loaded {len(test_list)} test samples.")
    print(f"Test data shape: {test_list.shape}")
    
    return test_list, label_list

def train_and_evaluate(args):
    # Extract configuration
    model_type = args.model_type
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    test_split = args.test_split
    random_seed = args.random_seed
    image_rows = args.image_rows
    image_columns = args.image_columns
    image_depth = args.image_depth
    dataset_path = args.dataset_path
    test_dataset_path = args.test_dataset_path
    test_only = args.test_only
    pretrained_weights = args.pretrained_weights
    
    # Load training dataset (previously test dataset)
    training_data, training_labels, emotion_to_label = load_training_dataset(
        dataset_path,
        image_rows=image_rows,
        image_columns=image_columns,
        image_depth=image_depth
    )
    
    # Get number of classes from the dataset
    num_classes = len(emotion_to_label)
    print(f"Number of classes detected: {num_classes}")
    
    # Split training data for validation
    train_data, val_data, train_labels, val_labels = train_test_split(
        training_data, training_labels, test_size=test_split, random_state=random_seed
    )
    
    print(f"Dataset splits - Training: {train_data.shape[0]}, Validation: {val_data.shape[0]}")
    
    # Load test dataset (previously training dataset/SAMM)
    test_data, test_labels = load_test_dataset(
        test_dataset_path,
        emotion_to_label,
        image_rows=image_rows,
        image_columns=image_columns,
        image_depth=image_depth
    )
    
    # Normalize all data
    print("Normalizing datasets...", flush=True)
    train_data = normalize_data(train_data)
    val_data = normalize_data(val_data)
    test_data = normalize_data(test_data)
    print("Normalization complete", flush=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if test_only and pretrained_weights:
        # Test-only mode with pretrained weights
        print(f"Running in test-only mode with pretrained weights: {pretrained_weights}", flush=True)
        
        # Build the base model architecture
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
        # Build model
        print(f"Building {model_type} model with {num_classes} classes...", flush=True)
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
        filepath = WEIGHTS_PATH
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        callbacks_list = [checkpoint, early_stopping, reduce_lr]
        
        # Train model
        print("Starting model training...", flush=True)
        history = model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            callbacks=callbacks_list, batch_size=batch_size, epochs=epochs, shuffle=True
        )
        print("Finished model training!", flush=True)
        
        # Plot training history
        print("Generating training plots...", flush=True)
        plot_path = os.path.join(RESULTS_DIR, f"training_history_{timestamp}.png")
        plot_training_history(history, plot_path)
    
    # Evaluation - this happens in both training and test-only modes
    print("Evaluating on test set...", flush=True)
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    
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
    target_names = list(emotion_to_label.keys())
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
    
    # ANOVA: Compare F1 scores across different emotion categories
    # Create dataframe for ANOVA
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels_argmax, predictions_labels
    )
    
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
    results_path = os.path.join(RESULTS_DIR, f"metrics_results_{timestamp}.txt")
    with open(results_path, 'w') as f:
        f.write(f"F1 score per class: {f1_per_class}\n")
        f.write(f"Weighted F1 score: {f1_weighted}\n")
        f.write(f"Macro F1 score: {f1_macro}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels_argmax, predictions_labels, target_names=target_names))
        f.write(f"\nT-test against chance level: t={t_stat:.4f}, p={p_value:.4f}\n")
        f.write(f"Mean confidence: {np.mean(max_probabilities):.4f}, Chance level: {chance_level:.4f}\n")
        f.write("\nANOVA across emotion categories:\n")
        try:
            f.write(f"F={f_stat:.4f}, p={p_anova:.4f}\n")
        except:
            f.write("ANOVA could not be performed\n")
        f.write("\nPer-class performance:\n")
        f.write(results_df.to_string())
    print(f"Metrics saved to {results_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.txt")
    np.savetxt(cm_path, cm, fmt='%d', delimiter=',')
    print(f"Confusion matrix saved to {cm_path}")
    
    if not test_only:
        # Save final model (only if training was performed)
        model_path = os.path.join(RESULTS_DIR, f"final_model_{timestamp}.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train micro-expression recognition model on CAS dataset and test on SAMM dataset (flipped roles).')
    # Mode selection
    parser.add_argument('--test_only', action='store_true', help='Run in test-only mode (no training)')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights to load')
    
    # Model and training parameters
    parser.add_argument('--model_type', type=str, default='original', 
                        choices=['original', 'resnet3d', 'mobilenet', 'simple3dcnn'], 
                        help='Model type to use.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--test_split', type=float, default=TEST_SPLIT, help='Test split ratio')
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED, help='Random seed')
    
    # Image dimensions
    parser.add_argument('--image_rows', type=int, default=IMAGE_ROWS, help='Image height')
    parser.add_argument('--image_columns', type=int, default=IMAGE_COLUMNS, help='Image width')
    parser.add_argument('--image_depth', type=int, default=IMAGE_DEPTH, help='Image depth (frames)')
    
    # Dataset paths
    parser.add_argument('--dataset_path', type=str, default=TRAINING_DATASET_PATH, 
                        help='Path to training dataset directory (CAS)')
    parser.add_argument('--test_dataset_path', type=str, default=TEST_DATASET_PATH, 
                        help='Path to test dataset directory (SAMM)')
    
    args = parser.parse_args()
    
    # Start training and evaluation
    train_and_evaluate(args)