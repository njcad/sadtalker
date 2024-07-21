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
from src.facerender.animate_PERFRAME import AnimateFromCoeff
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
        device = "cuda"

        # define sadtalker model paths 
        sadtalker_paths = init_path("checkpoints", os.path.join("src","config"))

        # initialize models 
        self.preprocess_model = CropAndExtract(sadtalker_paths, device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, device) 
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

        # set pose_style: default to 0
        self.pose_style = 0

        # set batch size. TODO: increase batch size?
        self.batch_size = 2

        # yaw, pitch, roll. default to None
        self.input_yaw_list = None
        self.input_pitch_list = None
        self.input_roll_list = None

        # eyeblink and pose both default to None
        self.ref_eyeblink = None
        self.ref_pose = None

        # initialize image data, set up in set_image()
        self.source_image = None
        self.first_coeff_path = None
        self.crop_pic_path = None
        self.crop_info = None

        # define results directory
        self.results_dir = "results"

    def set_image(self, source_image: Path):
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
        preprocess = "full"
        self.first_coeff_path, self.crop_pic_path, self.crop_info = self.preprocess_model.generate(
            source_image, 
            first_frame_dir, 
            preprocess, 
            source_image_flag=True
        )

        # error checking
        if self.first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return
        

    def run(self, driven_audio: Path):
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
            preprocess="full",
        )
        self.animate_from_coeff.generate(
            data, 
            self.results_dir, 
            self.source_image, 
            self.crop_info,
            enhancer="gfpgan", 
            background_enhancer=None,
            preprocess="full"
        )

        # save the output
        output = "/tmp/out.mp4"
        mp4_path = os.path.join(self.results_dir, [f for f in os.listdir(self.results_dir) if "enhanced.mp4" in f][0])
        shutil.copy(mp4_path, output)

        return Path(output)


def main():
    """
    Driver for Living Portrait class.
    """

    # declare image
    image_path = Path('../samples/newton.png')
    user_image_path = Path(input("Source image path from current directory: "))
    if user_image_path != "":
        image_path = user_image_path

    # instantiate TalkingPortrait method
    talker = TalkingPortrait()
    talker.set_image(image_path)

    # predict and display
    audio_path = Path('../samples/newton.m4a')
    vid_path = talker.run(audio_path)
    


if __name__ == '__main__':
    main()


