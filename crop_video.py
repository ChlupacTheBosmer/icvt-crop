from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.fx.all as vfx

def crop_video(video_path, point, crop_size, output_path):

    # Load the video
    video = VideoFileClip(video_path)

    # prepare coords
    x, y = point

    # Calculate crop region coordinates
    half_size = crop_size // 2
    x1 = max(x - half_size, 0)
    y1 = max(y - half_size, 0)
    x2 = min(x + half_size, video.w)
    y2 = min(y + half_size, video.h)

    # Apply cropping effect
    cropped_video = video.fx(vfx.crop, x1=x1, y1=y1, x2=x2, y2=y2)

    # Save the cropped video
    cropped_video.write_videofile(output_path, codec='libx264')

if __name__ == '__main__':
    # Define inputs
    video_path = r"D:\Dílna\Kutění\Python\Metacentrum\metacentrum\videos\GR2_L2_LavSto2_20220524_09_29.mp4"
    center_x = 925   # Example center x-coordinate
    center_y = 332  # Example center y-coordinate
    crop_size = 640  # Example crop size

    output_path = 'output_cropped_video.mp4'
    crop_video(video_path, (center_x, center_y), crop_size, output_path)