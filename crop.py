# Extra packages
import cv2

# Default part of python
import random
import os

def capture_crop(self, frame, point):
    # Define logger
    self.logger.debug(f"Running function capture_crop({point})")

    # Prepare local variables
    x, y = point

    # Add a random offset to the coordinates, but ensure they remain within the image bounds
    # DONE: Implement Milesight functionality
    frame_width, frame_height = self.video_file_object.get_frame_shape()

    # Check if any of the dimensions is smaller than crop_size and if so upscale the image to prevent crops smaller than desired crop_size
    if frame_height < self.crop_size or frame_width < self.crop_size:
        # Calculate the scaling factor to upscale the image
        scaling_factor = self.crop_size / min(frame_height, frame_width)

        # Calculate the new dimensions for the upscaled frame
        new_width = int(round(frame_width * scaling_factor))
        new_height = int(round(frame_height * scaling_factor))

        # Upscale the frame using cv2.resize with Lanczos upscaling algorithm
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Get the new frame size
    frame_height, frame_width = frame.shape[:2]

    # Calculate the coordinates for the area that will be cropped
    x_offset = random.randint(-self.offset_range, self.offset_range)
    y_offset = random.randint(-self.offset_range, self.offset_range)
    x1 = max(0, min(((x - self.crop_size // 2) + x_offset), frame_width - self.crop_size))
    y1 = max(0, min(((y - self.crop_size // 2) + y_offset), frame_height - self.crop_size))
    x2 = max(self.crop_size, min(((x + self.crop_size // 2) + x_offset), frame_width))
    y2 = max(self.crop_size, min(((y + self.crop_size // 2) + y_offset), frame_height))

    # Crop the image
    crop = frame[y1:y2, x1:x2]

    # Convert to correct color space
    crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    if crop_img.shape[2] == 3:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

    # Return the cropped image and the coordinates for future reference
    return crop_img, x1, y1, x2, y2


def generate_frames(self, frame, success, tag, index, frame_number_start):
    # Define logger
    self.logger.debug(f"Running function generate_frames({index})")

    # Prepare name elements
    filename_parts = tag[:-4].split("_")
    recording_identifier = "_".join(filename_parts[:-3])
    timestamp = "_".join(filename_parts[-3:])

    # Define local variables
    crop_counter = 1
    frame_skip_loc = self.frame_skip

    # Calculate the frame skip variable based on the limited number of frames per visit
    if self.frames_per_visit > 0:
        frame_skip_loc = int((self.visit_duration * self.fps) // self.frames_per_visit)
        if frame_skip_loc < 1:
            frame_skip_loc = 1

    # Loop through the video and crop y images every n-th frame
    frame_count = 0
    image_paths = []

    while success:
        # Crop images every n-th frame
        if int(frame_count % frame_skip_loc) == 0:
            for i, point in enumerate(self.points_of_interest_entry[index][0]):
                if self.cropped_frames == 1:
                    crop_img, x1, y1, x2, y2 = capture_crop(self, frame, point)
                    frame_number = frame_number_start + frame_count
                    roi_number = i + 1
                    visit_number = self.visit_index
                    image_name = f"{self.prefix}{recording_identifier}_{timestamp}_{roi_number}_{frame_number}_{visit_number}_{x1},{y1}_{x2},{y2}.jpg"  # Now the output images will be ordered by the ROI therefore one will be able to delete whole segments of pictures.
                    image_path = os.path.join(self.output_folder, image_name)
                    # image_path = f"./{self.output_folder}/{self.prefix}{recording_identifier}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_{i + 1}_{x1},{y1}_{x2},{y2}.jpg"
                    cv2.imwrite(image_path, crop_img)
                    image_paths.append(image_path)
                    self.image_details_dict[image_name] = [image_path, frame_number, roi_number, visit_number, 0]
            if self.whole_frame == 1:
                frame_number = frame_number_start + frame_count
                visit_number = self.visit_index
                image_name = f"{self.prefix}{recording_identifier}_{timestamp}_{frame_number}_{visit_number}_whole.jpg"
                image_path = os.path.join(self.output_folder, "whole frames", image_name)
                # image_path = f"./{self.output_folder}/whole frames/{self.prefix}{recording_identifier}_{timestamp}_{frame_number_start + frame_count}_{crop_counter}_whole.jpg"
                cv2.imwrite(image_path, frame)
            crop_counter += 1

        # If the random frame skip interval is activated add a random number to the counter or add the set frame skip interval
        if self.randomize == 1:
            if (frame_skip_loc - frame_count == 1):
                frame_count += 1
            else:
                frame_count += random.randint(1, max((frame_skip_loc - frame_count), 2))
        else:
            frame_count += frame_skip_loc

        # Read the next frame
        # DONE: Implement Milesight functionality
        frame_to_read = frame_number_start + frame_count
        success, frame = self.video_file_object.read_video_frame(frame_to_read)

        # If the frame count is equal or larger than the amount of frames that comprises the duration of the visit end the loop
        if not (frame_count < (self.visit_duration * self.fps) - 1):
            # Release the video capture object and close all windows
            # DONE: Implement Milesight functionality
            if not self.video_file_object.video_origin == "MS":
                self.video_file_object.cap.release()
            cv2.destroyAllWindows()
            break

    # Return the resulting list of image paths for future reference
    return image_paths