# edit number CUDA in file /home/server/server_denoise/FullSubNet-plus/speech_enhance/utils/utils.py
import os
import shutil
import sys
import toml
import uuid
import time
from tempfile import gettempdir
import warnings

warnings.filterwarnings("ignore")

TMP_PATH = f"{gettempdir()}/"  # Temp files, need clear

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from audio_zen.utils import initialize_module


def start(
    to_list_files: list,
):
    temp_file_dir_in = f"denoise_in_{uuid.uuid4().hex}"
    temp_file_dir_in_path = f"{TMP_PATH}{temp_file_dir_in}/"
    temp_file_dir_out = f"denoise_out_{uuid.uuid4().hex}"
    temp_file_dir_out_path = f"{TMP_PATH}{temp_file_dir_out}/"
    os.mkdir(temp_file_dir_in_path)
    os.mkdir(temp_file_dir_out_path)

    start_load = time.time()

    for copy_file in to_list_files:
        new_name = f"{str(copy_file).split('/')[-1]}"  # get old file name
        # shutil.move(copy_file, temp_file_dir_in_path)
        shutil.copy(copy_file, f"{temp_file_dir_in_path}{new_name}")

    dataset_dir_list = [temp_file_dir_in_path]
    configuration = toml.load("config/inference.toml")
    checkpoint_path = "/root/.cache/audio_enhance/models/best_model.tar"
    output_dir = temp_file_dir_out_path
    if len(dataset_dir_list) > 0:
        print(
            f"use specified dataset_dir_list: {dataset_dir_list}, instead of in config"
        )
        configuration["dataset"]["args"]["dataset_dir_list"] = dataset_dir_list

    inferencer_class = initialize_module(
        configuration["inferencer"]["path"], initialize=False
    )
    inferencer = inferencer_class(configuration, checkpoint_path, output_dir)
    inferencer()

    # enhanced_0194

    new_dir_file = f"{temp_file_dir_out_path}enhanced_0194/"
    files = os.listdir(new_dir_file)
    file_out_path_name = map(lambda name: os.path.join(new_dir_file, name), files)

    print("Copy TMP dir")
    # copy to temp dir
    for move_finish_file in file_out_path_name:
        new_name = f"{str(move_finish_file).split('/')[-1]}"  # get old file name
        shutil.move(move_finish_file, f"{TMP_PATH}{new_name}")
    print("remove directory")
    # remove directory
    shutil.rmtree(temp_file_dir_in_path)  # ignore_errors=True
    shutil.rmtree(temp_file_dir_out_path)  # ignore_errors=True

    finish_load = round(time.time() - start_load, 2)
    print(f"Load time: {finish_load}")

    return list(map(lambda name: os.path.join(TMP_PATH, name), files))
