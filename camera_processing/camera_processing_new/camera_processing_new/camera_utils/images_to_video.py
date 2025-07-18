import cv2
import glob
import os

# Directory containing images
folder = "./visual_observations/realsensecamera"
# Get all .png files in the folder
files = glob.glob(os.path.join(folder, "*.png"))

save_folder = "./visual_observations/videos"
# Create the save folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Group images by demo number
groups = {}
for filepath in files:
    filename = os.path.basename(filepath)
    # Expect filename format: "demoNumber_timestamp.png" (e.g., "0_92384792374.png")
    parts = filename.split('_')
    if len(parts) != 2:
        continue
    demo = parts[0]
    try:
        timestamp = float(parts[1].split('.')[0])
    except ValueError:
        continue
    groups.setdefault(demo, []).append((timestamp, filepath))

# Set frames per second for the video
fps = 20  # adjust as needed

# Process each demo group to create a video
for demo, images in groups.items():
    # Sort images by timestamp
    images.sort(key=lambda x: x[0])
    
    # Read the first image to determine dimensions
    first_image = cv2.imread(images[0][1], cv2.IMREAD_COLOR)
    if first_image is None:
        print(f"Error reading {images[0][1]}")
        continue
    # Convert from BGR to RGB if needed (depending on your processing requirements)
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    height, width, _ = first_image.shape

    # Define the output video filename with .mp4 extension
    video_filename = os.path.join(save_folder, f"demo{demo}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    # Check if the VideoWriter opened correctly
    if not video.isOpened():
        print(f"Error: VideoWriter could not open {video_filename}")
        continue

    print(f"Creating video for demo {demo} with {len(images)} images...")
    for _, img_path in images:
        # Read image and convert to RGB
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.write(frame)
    video.release()
    print(f"Video for demo {demo} saved as {video_filename}")

print("All videos have been created.")
