.PHONY: convert_to_melspec

PYTHON=python3

convert_to_melspec: INPUT_DIR=./data/raw/birdsong-recognition/train_audio
convert_to_melspec: OUTPUT_DIR=./data/processed/mel_specs_train
convert_to_melspec:
	$(PYTHON) ./src/data/convert_to_melspec.py $(INPUT_DIR) $(OUTPUT_DIR)

resample_audio: INPUT_DIR=./data/raw/birdsong-recognition/train_audio
resample_audio: OUTPUT_DIR=./data/processed/resampled_train_audio
resample_audio: RESAMPLE_RATE=44100
resample_audio:
	$(PYTHON) ./src/data/resample_audio.py $(INPUT_DIR) $(OUTPUT_DIR) --resample_rate $(RESAMPLE_RATE)

prepare_data: INPUT_DIR=./data/raw/birdsong-recognition/train_audio
prepare_data: OUTPUT_DIR=./data/processed/prepared_data
prepare_data: TARGET_SAMPLING_RATE=44100
prepare_data: MAX_DURATION=60
prepare_data: N_MELS=128
prepare_data: N_FFT=2048
prepare_data: HOP_LENGTH=2048
prepare_data: N_JOBS=28
prepare_data:
	$(PYTHON) ./src/data/resample_audio.py $(INPUT_DIR) $(OUTPUT_DIR) --target_sampling_rate $(TARGET_SAMPLING_RATE)\
																	  --max_duration $(MAX_DURATION)\
																	  --n_mels $(N_MELS)\
																	  --n_fft $(N_FFT)\
																	  --hop_length $(HOP_LENGTH)\
																	  --n_jons $(N_JOBS)
