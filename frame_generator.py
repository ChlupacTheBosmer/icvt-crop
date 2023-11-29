import os
import random
import numpy as np
import cv2
from ..video.video_passive import VideoFilePassive
from ..yolo.yolo_frame_array import detect_visitors_in_frame_array
from ..yolo.yolo_commons import save_label_file
import queue as qut
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools


class FrameGenerator():
    def __init__(self, video_filepaths: tuple, frame_data_dict, list_of_rois, crop_size, offset_range, name_prefix, output_folder, model_path: str = os.path.join('resources', 'yolo', 'best.pt')):

        self.crop_size = crop_size
        self.offset_range = offset_range
        self.name_prefix = name_prefix
        self.output_folder = output_folder
        self.model_path = model_path

        # Create a dictionary for fast ROI lookup
        if list_of_rois is not None:
            self.roi_dict = {os.path.basename(video_filepath): roi_entry for roi_entry, video_filepath in list_of_rois}
        else:
            self.roi_dict = {os.path.basename(video_filepath): [None] for video_filepath in video_filepaths}

        # Create a tuple of VideoFile objects
        video_files = tuple(VideoFilePassive(filepath) for filepath in video_filepaths)

        # Run the main body fo the generator
        self.main(1, 1, video_files, frame_data_dict)

    def main(self, nprod: int, ncon: int, video_files: tuple, frame_data_dict):

        try:

            # Create a shared queue
            crop_queue = qut.Queue()
            yolo_queue = qut.Queue()

            ndet = 1

            with ThreadPoolExecutor(max_workers=nprod + ncon + ndet) as executor:
                chunks = iter(video_files)
                for chunk in iter(lambda: list(itertools.islice(chunks, nprod)), []):
                    #print("Checking videos in this chunk.")
                    producer_futures = []
                    for video_file in chunk:
                        #print("Does this video have visits?")
                        if frame_data_dict.get(video_file.filename) is not None:
                            #print("Yes - creating producer per video.")
                            future = executor.submit(self.producer_task, video_file,
                                                     frame_data_dict[video_file.filename][0],
                                                     frame_data_dict[video_file.filename][1], crop_queue)
                            producer_futures.append(future)
                    #print(f" Producer futures: {producer_futures}")

                    consumer_futures = []
                    for n in range(ncon):
                        #print(f"Creating <{n}> consumer")
                        future = executor.submit(self.consumer_task, n, crop_queue, yolo_queue)
                        consumer_futures.append(future)
                    #print(f" Producer futures: {consumer_futures}")

                    detector_futures = []
                    for n in range(ndet):
                        #print(f"Creating <{n}> detector")
                        future = executor.submit(self.detector_task, n, frame_data_dict, yolo_queue)
                        detector_futures.append(future)
                    #print(f" Producer futures: {detector_futures}")

                    for future in as_completed(producer_futures):
                        #print(f"<{future}> completed.")
                        pass

                    for _ in range(ncon):
                        crop_queue.put((None, None))
        except Exception as e:
            raise


    def chunk_tuple(self, frame_numbers_tuple1, frame_numbers_tuple2, chunk_size):
        for i in range(0, len(frame_numbers_tuple1), chunk_size):
            end = i + chunk_size
            yield (frame_numbers_tuple1[i:end], frame_numbers_tuple2[i:end], end - i)

    def producer_task(self, video_object_file, frame_indices, visit_indices, queue):
        #print("(P) - Producer successfully created")
        try:
            filename = video_object_file.filename

            frame_batch_size = 100  # Maximum chunk size

            # Iterate through chunks
            for frame_numbers_chunk, visit_numbers_chunk, actual_chunk_size in self.chunk_tuple(frame_indices, visit_indices, frame_batch_size):

                # Pre-allocate 4D array
                frame_height, frame_width = video_object_file.get_frame_shape()
                frames_array = np.zeros((actual_chunk_size, frame_width, frame_height, 3), dtype=np.uint8)

                if video_object_file.video_origin == "MS":
                    frame_generator = video_object_file.read_frames_imageio(frame_numbers_chunk)
                else:
                    frame_generator = video_object_file.read_frames_decord(frame_numbers_chunk)
                for idx, frame_list in enumerate(frame_generator):
                    _, _, _, frame, _ = frame_list
                    frames_array[idx] = frame
                    #print(f"(P) - Adding a frame into array <{idx}>")
                #print("(P) - Array created")

                meta_data = {
                    'frame_numbers': frame_numbers_chunk,
                    'visit_numbers': visit_numbers_chunk,
                    'video_name': filename
                }
                #print("(P) - Metadata packed")
                queue.put((frames_array, meta_data))
                #print("(P) - Package added to the queue")
                #print(f"(P) - Producer <{filename}> added batch <{frame_numbers_chunk[0]} - {frame_numbers_chunk[-1:][0]}> to queue.")
        except Exception as e:
            print(f"(P) ERROR: - {e}")

    def consumer_task(self, name, crop_queue, yolo_queue):
        try:
            print(f"(C) - Consumer {name} created.")
            while True:
                print(f"(C) - Consumer {name} entered the loop.")
                frames_array, meta_data = crop_queue.get()
                print(f"(C) - Consumer {name} got item from the queue.")
                if frames_array is None:
                    print(f"(C) - Consumer {name} finished.")
                    yolo_queue.put((None, None))
                    yolo_queue.put((None, None))
                    break
                frame_numbers = meta_data['frame_numbers']
                video_filename = meta_data['video_name']
                rois = self.roi_dict[video_filename]
                #print(f"(C) - Consumer {name} got element <{frame_numbers[0]} - {frame_numbers[-1:]}>")
                cropped_frames = self.get_cropped_frames(frames_array, meta_data, rois, self.crop_size, self.offset_range)
                for batch_array, meta_data in cropped_frames:
                    yolo_queue.put((batch_array, meta_data))
                    print(f"(C) - Consumer added item into the queue")
        except Exception as e:
            print(f"(C) ERROR: - {e}")

    def detector_task(self, name, visitor_category_dict, yolo_queue):
        try:
            print(f"(D) - Detector {name} created.")
            while True:
                print(f"(D) - Detector {name} entered the loop.")
                frames_array, meta_data = yolo_queue.get()
                print(f"(D) - Detector {name} got item from the queue.")
                if frames_array is None:
                    print(f"(D) - Detector {name} finished.")
                    break

                frame_numbers = meta_data['frame_numbers']
                video_filename = meta_data['video_name']
                coords = meta_data['coords']
                print(f"(D) - Detector {name} got element <{frame_numbers[0]} - {frame_numbers[-1:]}>")

                detection_metadata = detect_visitors_in_frame_array(frames_array, meta_data, self.model_path)
                for idx, (frame_number, roi_number, visit_number, detection, _, boxes, *_) in enumerate(detection_metadata):
                    frame_name = f"{self.name_prefix}_{video_filename}_{roi_number}_{frame_number}_{visit_number}_{coords[idx][0]}_{coords[idx][1]}.jpg"
                    output_path = os.path.join(self.output_folder, "visitor") if detection > 0 else os.path.join(self.output_folder, "empty")
                    frame_path = os.path.join(output_path, frame_name)
                    try:
                        cv2.imwrite(frame_path, frames_array[idx])
                        if detection > 0:
                            label_path = os.path.join(output_path, f"{frame_name[:-4]}.txt")
                            save_label_file(label_path, boxes, visitor_category_dict[video_filename][2][visit_number])
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(f"(D) ERROR: - {e}")

    def get_cropped_frames(self, batch_frames, metadata, rois, crop_size=640, offset_range=100):
        #print("(G) - Generator initiated")
        _, frame_height, frame_width, _ = batch_frames.shape
        num_frames = metadata['frame_numbers']
        num_visits = metadata['visit_numbers']
        print(rois[0])
        # If rois is None aka no cropping should be done
        if rois[0] is None:
            print(rois[0])
            # Pre-allocate cropped frames for this ROI
            cropped_frames = np.empty((len(num_frames), frame_height, frame_width, 3), dtype=batch_frames.dtype)

            # Create meta info
            meta_data = {
                'video_name': metadata['video_name'],
                'frame_numbers': num_frames,
                'visit_numbers': num_visits,
                'roi_number': 0,
                'coords': []
            }
            print("test1")
            for idx, frame_number in enumerate(num_frames):
                frame_height, frame_width, *_ = batch_frames[idx].shape
                cropped_frames[idx] = batch_frames[idx, 0:frame_height, 0:frame_width]
                meta_data['coords'].append(((0, 0), (frame_width, frame_height)))
                print("test2")

            yield cropped_frames, meta_data
        else:
            # Loop over each ROI
            for i, point in enumerate(rois):
                #print("(G) - Generator entered the loop")
                x, y = point

                # Pre-allocate cropped frames for this ROI
                cropped_frames = np.empty((len(num_frames), crop_size, crop_size, 3), dtype=batch_frames.dtype)
                #print("(G) - Generator pre-allocated the array")

                # Create meta info
                meta_data = {
                    'video_name': metadata['video_name'],
                    'frame_numbers': num_frames,
                    'visit_numbers': num_visits,
                    'roi_number': i + 1,
                    'coords': []
                }

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
                    #print(f"(G) - Generator cropped a frame <{idx}>")

                    meta_data['coords'].append(((x1, y1), (x2, y2)))
                #print("(G) - Generator packed metadata and frames")

            yield cropped_frames, meta_data
