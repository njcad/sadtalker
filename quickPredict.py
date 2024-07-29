"""
For live prediction in the loop.
Nate Cadicamo
"""

# libraries and modules
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from pathlib import Path


# define a class 
class TalkingPortrait():
    def __init__(self):
        """
        Initialize models and variables we will use throughout the prediction process.
        """

        # define device. GPU highly necessary
        self.device = "cuda"

        # set preprocess mode to default crop. TODO: other modes?
        self.preprocess = "crop"

        # set image size to default 256. TODO: other size?
        self.size = 256

        # set pose_style: default to 0
        self.pose_style = 0

        # set batch size, default to 2. TODO: increase batch size?
        self.batch_size = 4

        # yaw, pitch, roll. default to None
        self.input_yaw = None
        self.input_pitch = None
        self.input_roll = None

        # eyeblink and pose both default to None
        self.ref_eyeblink = None
        self.ref_eyeblink_coeff_path = None
        self.ref_pose = None
        self.ref_pose_coeff_path = None

        # initialize image data, set up in set_image()
        self.source_image = None
        self.first_coeff_path = None
        self.crop_pic_path = None
        self.crop_info = None

        # define results directory
        self.results_dir = "results"

         # define sadtalker model paths 
        # sadtalker_paths = init_path("checkpoints", os.path.join("src","config"))
        current_root_path = os.path.split(sys.argv[0])[0]
        sadtalker_paths = init_path("./checkpoints", os.path.join(current_root_path, "src/config"), self.size, False, self.preprocess)

        # initialize models 
        self.preprocess_model = CropAndExtract(sadtalker_paths, self.device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, self.device) 
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, self.device)
        

    def set_image(self, source_image):
        """
        Grab and save 3DMM extraction for source image.
        """

        # save source image
        self.source_image = source_image

        # define results directory for first frame
        results_dir = self.results_dir
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        first_frame_dir = os.path.join(results_dir, "first_frame_dir")
        os.makedirs(first_frame_dir)

        # crop image and extract 3dmm from image
        self.first_coeff_path, self.crop_pic_path, self.crop_info = self.preprocess_model.generate(
            source_image, 
            first_frame_dir, 
            self.preprocess, 
            source_image_flag=True,
            pic_size=self.size
        )

        # error checking
        if self.first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return
        

    def run(self, driven_audio):
        """
        To be used in loop, fed new audio and returning video.
        """

        # first, run audio to coeff model
        batch = get_data(
            self.first_coeff_path,
            driven_audio,
            self.device,
            self.ref_eyeblink_coeff_path,
            still=False
        )
        coeff_path = self.audio_to_coeff.generate(
            batch, 
            self.results_dir, 
            self.pose_style, 
            self.ref_pose_coeff_path
        )

        # then, run coefficient to video model
        data = get_facerender_data(
            coeff_path,
            self.crop_pic_path,
            self.first_coeff_path,
            driven_audio,
            self.batch_size,
            self.input_yaw,
            self.input_pitch,
            self.input_roll,
            expression_scale=1.0,
            still_mode=False,
            preprocess=self.preprocess,
            size=self.size
        )
        result = self.animate_from_coeff.generate(
            data, 
            self.results_dir, 
            self.source_image, 
            self.crop_info,
            enhancer=None, 
            background_enhancer=None,
            preprocess=self.preprocess,
            img_size=self.size
        )

        # save the output
        shutil.move(result, self.results_dir+'.mp4')
        # print('The generated video is named:', self.results_dir+'.mp4')

        # return the output
        return self.results_dir+'.mp4'
        

def get_image(gen=False):
    """
    Grab an image for Talking Portrait. 
    Default: user specified input image from file.
    TODO: enable call API for image gen, or source from a bank.
    """

    # default case: no generation, just sourcing from file
    if not gen:
        default_image_path = '../samples/newton.png'
        user_image_path = input("Source image path from current directory: ").strip()
        if user_image_path:
            image_path = user_image_path
        else:
            image_path = default_image_path
        return image_path

    # generative case: API call to generate image
    if gen:
        # TODO: implement functionality
        pass


def get_audio(gen=False):
    """
    Grab audio file (.wav or .m4a) for Talking Portrait. 
    Default: user specified input audio from file.
    TODO: enable call API for audio gen, or source from a bank.
    """

    # default case: no generation, just sourcing from file
    if not gen:
        default_audio_path = '../samples/newton.m4a'      
        user_audio_path = input("Source audio path from current directory: ").strip()
        if user_audio_path:
            audio_path = user_audio_path
        else:
            audio_path = default_audio_path
        return audio_path

    # generative case: API call to generate audio
    if gen:
        # TODO: implement functionality
        pass
    


def main():
    """
    Driver for Talking Portrait class.
    """

    # pull arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image file.')
    parser.add_argument('--audio', type=str, required=True, help='Path to the input audio file.')
    args = parser.parse_args()

    # instantiate TalkingPortrait instance
    talker = TalkingPortrait()

    # declare image
    # image_path = get_image()
    image_path = args.image
    
    # set image once
    image_start_time = time.time()
    talker.set_image(image_path)
    image_end_time = time.time()
    # print(f"Time to set image in TalkingPortrait model = {image_end_time - image_start_time}")

    # declare audio
    audio_path = args.audio

    # get vid path
    vid_path = talker.run(audio_path)
    return vid_path

    

    # # allow user to continually feed audio
    # while True:       
    #     audio_path = get_audio()

    #     # run with this audio 
    #     talker_start_time = time.time()
    #     talker.run(audio_path)
    #     talker_end_time = time.time()
    #     print(f"Time to generate TalkingPortrait = {talker_end_time - talker_start_time}")

    #     # if user wants to continue
    #     cont = input("Continue with new audio file? Y/N: ").strip()
    #     if cont != "Y":
    #         break
    

if __name__ == '__main__':
    main()


