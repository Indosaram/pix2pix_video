import util.util as util
import torchvision.transforms as transforms
import cv2
import subprocess
import shlex
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def save_tensor(tensor, path, text="", text_xpos=985, text_ypos=30, text_color=(255,255,255)):
    """Saving a Torch image tensor into an image (with text)"""
    img_nda = util.tensor2im(tensor.data[0])
    img_pil = Image.fromarray(img_nda)

    if text != "":
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 50)
        draw.text((text_xpos, text_ypos), text, text_color, font=font)

    img_pil.save(path)

def zoom_in(input_tensor, w=1280, h=736, zoom_level=4):
    """Zoom-in: Crop borders and resize a tensor"""
    img_nda = util.tensor2im(input_tensor.data[0])
    img_nda = img_nda[zoom_level:-zoom_level:,2*zoom_level:-2*zoom_level,:]
    img_nda = cv2.resize(img_nda, (w,h), interpolation=cv2.INTER_LINEAR)
    return transforms.functional.to_tensor(img_nda).reshape((1,3,h,w))

def next_frame_prediction(generator, input_tensor):
    """Just one forward pass through the generator"""
    output_tensor = generator.inference(input_tensor, None, None)
    return output_tensor

def extract_frames_from_video(video_path, frame_dir, output_shape=(1280, 736), ffmpeg_verbosity=16):
    """Extract all frames from a video
      - scale down the frames to match the desired height
      - crop to the desired width
    ex: 1920x1080 --> 1308x736 --> 1280x736
    """
    width, height = output_shape
    command = """ffmpeg -v %d -i %s -q:v 2 -vf "scale=iw*%d/ih:%d, crop=%d:%d" %s/frame-%%06d.jpg -hide_banner""" % (
        ffmpeg_verbosity,
        video_path,
        height,
        height,
        width,
        height,
        frame_dir
    )
    print(command)
    print("extracting the frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

def video_from_frame_directory(frame_dir, video_path, frame_file_glob=r"frame-%05d.jpg", framerate=24, ffmpeg_verbosity=16, crop_to_720p=True):
    """Build a mp4 video from a directory frames"""
    command = """ffmpeg -v %d -framerate %d -i %s -ss 1 -q:v 2%s %s""" % (
        ffmpeg_verbosity,
        framerate,
        frame_dir + "/" + frame_file_glob,
        ' -filter:v "crop=1280:720:0:16"' if crop_to_720p else "",
        video_path
    )
    print(command)
    print("building video from frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

