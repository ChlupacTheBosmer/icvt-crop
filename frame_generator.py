import asyncio
import itertools as it
import os
import random
import time
import imageio
import numpy as np
import cv2
from ..video.video_passive import VideoFilePassive
from ..crop.crop import icvtFrame
from ..yolo.yolo_frame_array import detect_visitors_in_frame_array
from ..yolo.yolo_commons import save_label_file
import threading
import queue as qut
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools


class FrameGenerator():
    def __init__(self, video_filepaths: tuple, frame_data_dict, list_of_rois, crop_size, offset_range, name_prefix, output_folder):

        self.crop_size = crop_size
        self.offset_range = offset_range
        self.name_prefix = name_prefix
        self.output_folder = output_folder

        # Create a dictionary for fast ROI lookup
        self.roi_dict = {os.path.basename(video_filepath): roi_entry for roi_entry, video_filepath in list_of_rois}

        # Create a tuple of VideoFile objects
        video_files = tuple(VideoFilePassive(filepath) for filepath in video_filepaths)

        # Iterate over video files to set ROIs
        for video_file in video_files:
            roi_entry = self.roi_dict.get(os.path.basename(video_file.filename))
            if roi_entry is not None:
                video_file.rois = roi_entry

        # Run the main body fo the generator
        self.main(1, 1, video_files, frame_data_dict)

    def main(self, nprod: int, ncon: int, video_files: tuple, frame_data_dict):

        chunks = iter(video_files)
        for chunk in iter(lambda: list(itertools.islice(chunks, nprod)), []):
            # Create a shared queue
            crop_queue = qut.Queue()
            yolo_queue = qut.Queue()


            with ThreadPoolExecutor(max_workers=nprod + ncon) as executor:
                producer_futures = []
                for video_file in chunk:
                    future = executor.submit(self.producer_task, video_file, frame_data_dict[video_file.filename][0],
                                             frame_data_dict[video_file.filename][1], crop_queue)
                    producer_futures.append(future)

                consumer_futures = []
                for n in range(ncon):
                    future = executor.submit(self.consumer_task, n, crop_queue, yolo_queue)
                    consumer_futures.append(future)

                detector_futures = []
                for n in range(ncon):
                    future = executor.submit(self.detector_task, n, frame_data_dict[video_file.filename][2], yolo_queue)
                    detector_futures.append(future)

                for future in as_completed(producer_futures):
                    print(f"<{future}> completed.")

                for _ in range(ncon):
                    crop_queue.put(None)

    def chunk_tuple(self, frame_numbers_tuple1, frame_numbers_tuple2, chunk_size):
        for i in range(0, len(frame_numbers_tuple1), chunk_size):
            end = i + chunk_size
            yield (frame_numbers_tuple1[i:end], frame_numbers_tuple2[i:end], end - i)

    def producer_task(self, video_object_file, frame_indices, visit_indices, queue):
        filename = video_object_file.filename

        frame_batch_size = 500  # Maximum chunk size

        # Iterate through chunks
        for frame_numbers_chunk, visit_numbers_chunk, actual_chunk_size in self.chunk_tuple(frame_indices, visit_indices, frame_batch_size):

            # Pre-allocate 4D array
            frame_shape = video_object_file.get_frame_shape()
            frames_array = np.zeros((actual_chunk_size, *frame_shape, 3), dtype=np.uint8)

            if video_object_file.video_origin == "MS":
                frame_generator = video_object_file.read_frames_imageio(frame_indices)
            else:
                frame_generator = video_object_file.read_frames_decord(frame_indices)
            for idx, frame_list in enumerate(frame_generator):
                _, _, _, frame, _ = frame_list
                frames_array[idx] = frame
            meta_data = {
                'frame_numbers': frame_numbers_chunk,
                'frame_visits': visit_numbers_chunk,
                'video_name': filename
            }
            queue.put((frames_array, meta_data))
            print(f"Producer <{filename}> added batch <{frame_numbers_chunk[0][0]} - {frame_numbers_chunk[-1:][0]}> to queue.")
        del video_object_file
        del frame_indices
        del frame_generator

    def process_frame(self, frame_list, rois):
        recording_identifier, timestamp, frame_number, frame = frame_list
        cropped_frames = list(self.crop_frame(rois, frame, [recording_identifier, timestamp, frame_number], 640, 100))
        return cropped_frames

    def consumer_task(self, name, crop_queue, yolo_queue):

        print(f"Consumer {name} created.")
        while True:

            frames_array, meta_data = crop_queue.get()
            if frames_array is None:
                print(f"Consumer {name} finished.")
                break
            frame_numbers = meta_data['frame_numbers']
            visit_numbers = meta_data['visit_numbers']
            video_filename = meta_data['video_name']
            rois = self.roi_dict[video_filename]
            print(f"Consumer {name} got element <{frame_numbers[0]} - {frame_numbers[-1:]}>")

            cropped_frames = self.get_cropped_frames(frames_array, meta_data, self.crop_size, self.offset_range)
            for batch_array, meta_data in cropped_frames:
                print(f"FUCK YEEES")
                yolo_queue.put((frames_array, meta_data))

    def detector_task(self, name, visitor_category_dict, yolo_queue):

        print(f"Detector {name} created.")
        while True:
            frames_array, meta_data = yolo_queue.get()
            if frames_array is None:
                print(f"Detector {name} finished.")
                break
            frame_numbers = meta_data['frame_numbers']
            video_filename = meta_data['video_name']
            coords = meta_data['coords']
            print(f"Detector {name} got element <{frame_numbers[0]} - {frame_numbers[-1:]}>")

            detection_metadata = detect_visitors_in_frame_array(frames_array, meta_data, os.path.join('resources', 'yolo', 'best.pt'))
            for idx, frame_number, roi_number, visit_number, detection, _, boxes, *_ in enumerate(detection_metadata):
                frame_name = f"{self.name_prefix}_{video_filename}_{roi_number}_{frame_number}_{visit_number}_{coords[0]}_{coords[1]}.jpg"
                output_path = os.path.join(self.output_folder, "visitor") if detection > 0 else os.path.join(self.output_folder, "empty")
                frame_path = os.path.join(output_path, frame_name)
                try:
                    cv2.imwrite(frame_path, frames_array[idx])
                except Exception as e:
                    print(e)
                label_path = os.path.join(output_path, f"{frame_name[:-4]}.txt")
                save_label_file(label_path, boxes, visitor_category_dict[visit_number])



    def crop_frame(self, rois, frame, frame_metadata, crop_size, offset_range):

        recording_identifier, timestamp, frame_number = frame_metadata

        for i, point in enumerate(rois):

            # Prepare local variables
            x, y = point

            # Add a random offset to the coordinates, but ensure they remain within the image bounds
            frame_height, frame_width, _ = frame.shape

            # Check if any of the dimensions is smaller than crop_size and if so
            # upscale the image to prevent crops smaller than desired crop_size
            if frame_height < crop_size or frame_width < crop_size:
                # Calculate the scaling factor to upscale the image
                scaling_factor = crop_size / min(frame_height, frame_width)

                # Calculate the new dimensions for the upscale frame
                new_width = int(round(frame_width * scaling_factor))
                new_height = int(round(frame_height * scaling_factor))

                # Upscale the frame using cv2.resize with Lanczos up-scaling algorithm
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Get the new frame size
            frame_height, frame_width, _ = frame.shape

            # Calculate the coordinates for the area that will be cropped
            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)
            x1 = max(0, min(((x - crop_size // 2) + x_offset), frame_width - crop_size))
            y1 = max(0, min(((y - crop_size // 2) + y_offset), frame_height - crop_size))
            x2 = max(crop_size, min(((x + crop_size // 2) + x_offset), frame_width))
            y2 = max(crop_size, min(((y + crop_size // 2) + y_offset), frame_height))

            # Crop the image
            crop = frame[y1:y2, x1:x2]

            # Convert to correct color space
            crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if crop_img.shape[2] == 3:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

            cropped_frame = icvtFrame(crop_img, recording_identifier, timestamp, frame_number, i + 1,
                                      (x1, y1), (x2, y2))

            yield cropped_frame

    def get_cropped_frames(self, batch_frames, metadata, rois, crop_size=640, offset_range=100):
        _, frame_height, frame_width, _ = batch_frames.shape
        num_frames = metadata['frame_numbers']
        num_visits = metadata['visit_numbers']

        # Loop over each ROI
        for i, point in enumerate(rois):
            x, y = point

            # Pre-allocate cropped frames for this ROI
            cropped_frames = np.empty((num_frames, crop_size, crop_size, 3), dtype=batch_frames.dtype)

            meta_data = []

            # Loop over each frame
            for idx, frame_number in enumerate(num_frames):

                # Get frame dimensions here
                frame_height, frame_width, _ = batch_frames[idx].shape

                # Check if any of the dimensions are smaller than crop_size
                if frame_height < crop_size or frame_width < crop_size:
                    scaling_factor = crop_size / min(frame_height, frame_width)
                    new_width = int(round(frame_width * scaling_factor))
                    new_height = int(round(frame_height * scaling_factor))

                    # Upscale the frame using cv2.resize with Lanczos up-scaling algorithm
                    batch_frames[idx] = cv2.resize(batch_frames[idx], (new_width, new_height),
                                                   interpolation=cv2.INTER_LANCZOS4)

                    # Update dimensions
                    frame_height, frame_width, _ = batch_frames[frame_number].shape

                # Get the random offset
                x_offset = random.randint(-offset_range, offset_range)
                y_offset = random.randint(-offset_range, offset_range)

                # Calculate the coordinates for the area that will be cropped
                x1 = max(0, min(((x - crop_size // 2) + x_offset), frame_width - crop_size))
                y1 = max(0, min(((y - crop_size // 2) + y_offset), frame_height - crop_size))
                x2 = max(crop_size, min(((x + crop_size // 2) + x_offset), frame_width))
                y2 = max(crop_size, min(((y + crop_size // 2) + y_offset), frame_height))

                # Crop the frame
                cropped_frames[idx] = batch_frames[idx, y1:y2, x1:x2]

                # Create meta info
                meta_data.append({
                    'video_name': metadata['video_name'],
                    'frame_numbers': num_frames,
                    'visit_numbers': num_visits,
                    'roi_number': i + 1,
                    'coords': [(x1, y1), (x2, y2)]
                })

            yield cropped_frames, meta_data


def generate_frame_indices(total_frames, frames_to_skip) -> tuple:

    # Generate a list of indices of every n-th frame
    frame_indices = tuple(range(1, total_frames, frames_to_skip))

    # Make sure that the last index doesn't exceed the total_frames
    #frame_indices = [idx for idx in frame_indices if idx < total_frames]

    return frame_indices


if __name__ == "__main__":
    this = FrameGenerator()






