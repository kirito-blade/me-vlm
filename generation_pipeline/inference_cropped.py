import os
import cv2  # Assuming OpenCV is installed for cropping and resizing
import dlib  # For facial landmark detection
import pandas as pd
import PIL
import torch
import torchvision.transforms as transforms
from PIL import Image
# from diffusers import StableDiffusionInstructPix2PixPipeline, DiffusionPipeline

# Define the path to the CSV file
attributes_file = "/home/tpei0009/ACE/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.csv"

# Initialize the model pipeline
# model_id = "/home/tpei0009/instructme2me/instruct-me2me"  # Local path to the model
# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
# pipe = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix").to('cuda')

# generator = torch.Generator("cuda").manual_seed(0)
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Initialize dlib's pre-trained face detector and shape predictor (for facial landmarks)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/tpei0009/STSTNet/shape_predictor_68_face_landmarks.dat")  # Adjust path

# Function to load and resize an image
def load_image_from_file(file_path, size=(256, 256)):
    try:
        image = PIL.Image.open(file_path)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(size)  # Resize to 256x256
        return image
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Parse the CSV file and filter images where smile attribute is -1
def get_images_with_smile_neg_one(file_path):
    images = []
    df = pd.read_csv(file_path)
    
    # Assuming the first column is the image filename and the columns for smile and gender attributes are labeled correctly
    for index, row in df.iterrows():
        image_id = row['image_id']  # Image file name (e.g., '0.jpg')
        smile_attribute = row['Smiling']  # Smile attribute (adjust index if necessary)
        
        if smile_attribute == -1:  # Check if smile attribute is -1
            images.append((image_id, row['Male']))  # Store the image_id and gender attribute
    return images

# Function to crop the face centered on the nose using facial landmarks
def crop_face_centered_on_nose(image, landmarks):
    # The landmark for the nose tip is index 30 (nose tip)
    nose_x, nose_y = landmarks.part(30).x, landmarks.part(30).y
    
    # Shrink the crop area around the nose, e.g., crop a 512x512 region
    small_crop_size = 512  # Smaller area around the nose to focus on the face
    half_small_crop = small_crop_size // 2
    
    # Get the smaller crop box around the nose
    x1 = max(0, nose_x - half_small_crop)
    y1 = max(0, nose_y - half_small_crop)
    x2 = min(image.shape[1], nose_x + half_small_crop)
    y2 = min(image.shape[0], nose_y + half_small_crop)
    
    # Crop the image to the smaller region
    cropped_face = image[y1:y2, x1:x2]
    
    # Convert the cropped region to a PIL image
    cropped_face = Image.fromarray(cropped_face)
    
    # Define the transformation: resize to 256x256
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Apply the transformation and move to GPU
    cropped_face = transform(cropped_face).unsqueeze(0).to('cuda')
    
    return cropped_face

# Function to detect face and crop if file is found but cropping is needed
def handle_missing_image(image_name):
    image_path = f"/home/tpei0009/instructme2me/Cropped_celebahq/{image_name}"
    
    # Try to load the image using OpenCV
    img_cv2 = cv2.imread(image_path)
    
    if img_cv2 is None:
        print(f"Image not found: {image_path}")
        return None
    
    # Convert the image to RGB format
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a PIL image
    img_pil = Image.fromarray(img_cv2)
    
    # Move the image to GPU
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to('cuda')
    
    return img_tensor

# Function to generate images with emotion type
def generate_image(image_list, emo_type):
    # Define parameters for the pipeline
    prompt = (
        "A professional static headshot of a person showing a sadness micro-expression transitioning to neutral. "
        "The expression involves slight lip corners turning down, minimal inner eyebrow raising, and a subtle downturn of the mouth. "
        # "The micro-expression appears and disappears quickly with only minor changes in facial muscles. "
        # "The background is plain and unobtrusive, ensuring full focus on the face. "
        # "The lighting is even, with no shadows or harsh contrasts, maintaining a 4K resolution for clarity. "
        # "The frame is locked, with no movement or blurring, capturing the subtle transition in expression."
    )
    negative_prompt = (
        "Blink eye, blurry, out of focus, double exposure, motion blur, head movement, camera shake, extreme expressions, "
        "distorted features, unrealistic skin, artistic effects, filters, overlays, multiple faces, side angles, tilted head, "
        "poor lighting, shadows, low resolution, pixelation, compression artifacts, makeup, accessories, jewelry, visible clothing, "
        "background details, hand gestures, body movement, animation style, cartoon, illustration, 3d render"
    )

    # Process each image in a loop
    for image_name, male_attribute in image_list:
        image_path = f"/home/tpei0009/ACE/CelebAMask-HQ/CelebA-HQ-img/{image_name}"
        vid_name = os.path.splitext(image_name)[0]  # Get name without extension
        image = load_image_from_file(image_path)
        
        # Handle case where file is not found or needs cropping
        if image is None:
            image = handle_missing_image(image_name)
            if image is None:
                continue  # Skip if the image cannot be processed

        # Determine gender_id based on male_attribute
        gender_id = 'm' if male_attribute == 1 else 'f' if male_attribute == -1 else 'unknown'

        # Generate the edited image
        # edited_image = pipe(
        #     prompt,
        #     image=image,
        #     num_inference_steps=num_inference_steps,
        #     image_guidance_scale=image_guidance_scale,
        #     guidance_scale=guidance_scale,
        #     generator=generator,
        # ).images[0]
        video = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=256,
            height=256,
            num_frames=161,
            num_inference_steps=80,
        ).frames[0]
        # Save the generated image with gender_id included in the path
        output_path = f"/home/tpei0009/LTX-Video/hq_emotion/{emo_type}/{gender_id}_{vid_name}.mp4"
        export_to_video(video, output_path, fps=24)
        # video.save(output_path)
        # video.save(output_path)
        print(f"Saved edited image: {output_path}")

# Load the images where smile attribute is -1
image_list = get_images_with_smile_neg_one(attributes_file)
emotion_list = ['sadness']#, 'sadness', 'disgust', 'surprise', 'fear']

for i in range(len(emotion_list)):
    generate_image(image_list, emotion_list[i])
