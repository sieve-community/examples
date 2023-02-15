from os import listdir, path
import numpy as np
import cv2, os, audio
import subprocess
from tqdm import tqdm
from glob import glob
import torch
from models import Wav2Lip
import platform
import time


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images, predictions):
	batch_size = 8
	
	results = []
	pady1, pady2, padx1, padx2 = (0, 15, 0, 0)
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results

wav2lip_batch_size = 128


def datagen(frames, mels, faces, start_frame):
	frames = frames[start_frame:start_frame + len(faces)]
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	face_det_results = face_detect(frames, faces) # BGR2RGB for CNN face detection

	for i, m in enumerate(mels):
		idx = i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (96, 96))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, 96//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, 96//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

class Model():
    def __init__(self, checkpoint_path):
        def _load(checkpoint_path):
            if device == 'cuda':
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            return checkpoint

        def load_model(path):
            print("Loading Wav2Lip model...")
            model = Wav2Lip()
            print("Load checkpoint from: {}".format(path))
            checkpoint = _load(path)
            s = checkpoint["state_dict"]
            new_s = {}
            for k, v in s.items():
                new_s[k.replace('module.', '')] = v
            model.load_state_dict(new_s)

            model = model.to(device)
            return model.eval()
        self.crop = [0, -1, 0, -1]

        self.model = load_model(checkpoint_path)
        print("Model loaded successfully!")
    
    def predict(self, face, audio_file, outfile, resize_factor, faces, start_frame):
        t = time.time()
        if not os.path.isfile(face):
            raise ValueError('--face argument must be a valid path to video/image file')
       
        video_stream = cv2.VideoCapture(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        if not audio_file.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i \'{}\' -strict -2 {}'.format(audio_file, 'temp/temp.wav')

            subprocess.call(command, shell=True)
            audio_file = 'temp/temp.wav'

        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape, time.time() - t)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        print('Reading video frames...')

        full_frames = []
        while len(full_frames) < len(mel_chunks):
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

                y1, y2, x1, x2 = self.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

        print("Number of frames available for inference: "+str(len(full_frames)))

        # full_frames = full_frames[:len(mel_chunks)]

        batch_size = wav2lip_batch_size
        gen = datagen(full_frames.copy(), mel_chunks, faces, start_frame)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                print ("Model loaded")

                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi', 
                                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = p[1*(p.shape[0]//2):p.shape[0]]
                y1 = y1 + (1 * ((y2 - y1) // 2))
                p = cv2.resize(p, (x2 - x1, (y2 - y1)))
                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()
        print("running model finished", time.time() - t)

        command = 'ffmpeg -y -i \'{}\' -i \'{}\' -strict -2 -q:v 1 {}'.format(audio_file, 'temp/result.avi', outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')
        print("running ffmpeg finished", time.time() - t)

        return outfile
