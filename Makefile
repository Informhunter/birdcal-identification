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

