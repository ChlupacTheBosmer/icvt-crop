from ..database import sqlite_data

# Extra packages
import cv2

# Default part of python
import random
import os
import asyncio

def capture_crop(self, frame, point, video_file_object):
    # Define logger
    self.logger.debug(f"Running function capture_crop({point})")

    # Prepare local variables
    x, y = point

    # Add a random offset to the coordinates, but ensure they remain within the image bounds
    frame_height, frame_width,_ = frame.shape

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


def generate_frames(self, video_file_object, list_of_rois, frame_number_start, visit_duration, visit_number: int = 0,
                    frames_to_skip: int = 15, frames_per_visit: int = 0, generate_cropped_frames: bool = True,
                    generate_whole_frames: bool = False, name_prefix: str = ""):

    # Define logger
    self.logger.debug(f"Running function generate_frames({list_of_rois})")

    # Prepare name elements
    recording_identifier = video_file_object.recording_identifier
    timestamp = video_file_object.timestamp

    # Calculate the frame skip variable based on the limited number of frames per visit or use the custom set value
    if frames_per_visit > 0:
        frames_to_skip = int((visit_duration * video_file_object.fps) // frames_per_visit) if not frames_to_skip < 1 else 1

    # Loop through the video and crop y images every n-th frame
    frame_count = 0

    # Read first frame
    print(dir(video_file_object))
    print(type(video_file_object))
    frame = video_file_object.read_video_frame(frame_indices=frame_number_start, stream=False)[0][3]

    while True:
        # Crop images every n-th frame
        if int(frame_count % frames_to_skip) == 0:
            frame_number = frame_number_start + frame_count
            if generate_cropped_frames:
                for roi_number, point in enumerate(list_of_rois):

                    # Crop the frame
                    crop_img, x1, y1, x2, y2 = capture_crop(self, frame, point, video_file_object)

                    # Construct frame
                    cropped_frame = icvtFrame(crop_img, recording_identifier, timestamp, frame_number, roi_number+1,
                                              (x1, y1), (x2, y2),
                                              visit_number, name_prefix)

                    yield cropped_frame
                    # cv2.imwrite(image_path, crop_img)
                    # image_paths.append(image_path)
                    # self.image_details_dict[image_name] = [image_path, frame_number, roi_number, visit_number, 0]
            if generate_whole_frames:

                # Construct frame
                whole_frame = icvtFrame(frame, recording_identifier, timestamp, frame_number, visit_number=visit_number, name_prefix=name_prefix)
                yield whole_frame
                # cv2.imwrite(image_path, frame)

        # If the random frame skip interval is activated add a random number to the counter or add the set frame skip interval
        if self.randomize == 1:
            if (frames_to_skip - frame_count == 1):
                frame_count += 1
            else:
                frame_count += random.randint(1, max((frames_to_skip - frame_count), 2))
        else:
            frame_count += frames_to_skip

        # If the frame count is equal or larger than the amount of frames that comprises the duration of the visit end the loop
        if frame_count >= (visit_duration * video_file_object.fps) - 1:
            # Release the video capture object and close all windows
            cv2.destroyAllWindows()
            break

        # Read the next frame
        frame_to_read = frame_number_start + frame_count
        frame = video_file_object.read_video_frame(frame_to_read, False)[0][3]


class icvtFrame():
    def __init__(self, frame, recording_identifier, timestamp,
                 frame_number, roi_number: int = -1, crop_upper_left_corner=None, crop_bottom_right_corner=None,
                 visit_number: int = 0, name_prefix: str = ""):

        # Init variables
        self.frame = frame
        self.crop_upper_left_corner = crop_upper_left_corner if crop_upper_left_corner is not None else (0, 0)
        self.crop_bottom_right_corner = crop_bottom_right_corner if crop_bottom_right_corner is not None else frame.shape[:2][::-1]
        self.recording_identifier = recording_identifier
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.roi_number = roi_number
        self.visit_number = visit_number
        self.name_prefix = name_prefix
        self.visitor_detected = False
        self.is_cropped = True if roi_number >= 0 else False
        self.frame_path = None
        self.id = None

        # Define name based on whether it is a cropped or a whole frame automatically
        name_if_cropped = (f"{self.name_prefix}{self.recording_identifier}_{self.timestamp}_{self.roi_number}_"
                           f"{self.frame_number}_{self.visit_number}_{self.crop_upper_left_corner[0]},"
                           f"{self.crop_upper_left_corner[1]}_{self.crop_bottom_right_corner[0]},"
                           f"{self.crop_bottom_right_corner[1]}.jpg")
        name_if_whole = (f"{self.name_prefix}{self.recording_identifier}_{self.timestamp}_{self.frame_number}_"
                         f"{self.visit_number}_whole.jpg")
        self.name = name_if_cropped if not self.roi_number < 0 else name_if_whole

    def generate_output_path(self, output_folder):

        output_path = os.path.join(output_folder, self.name) if self.is_cropped else os.path.join(output_folder, "whole frames", self.name)

        return output_path

    def save(self, output_folder):

        output_path = self.generate_output_path(output_folder)

        try:
            cv2.imwrite(output_path, self.frame)
            self.frame_path = output_path
            return True
        except Exception as e:
            print(e)
            return False


