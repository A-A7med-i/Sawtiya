from src.constant.constant import *
from src.pipeline.pipeline import AudioPipeline
from src.utils.helper import collate_audio_batch


if __name__ == "__main__":
    pipeline = AudioPipeline(
        root_dir=BASE_AUDIO_DIR,
        emotion_map=EMOTION_MAP,
        intensity_map=INTENSITY_MAP,
        label_map=EMOTION_LABEL,
        train_split=TRAIN_SPLIT,
        test_split=TEST_SPLIT,
        sample_rate=SAMPLE_RATE,
        silence_db=SILENCE_THRESHOLD_DB,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        collate_fn=collate_audio_batch,
        time_stretch_prob=AUG_TIME_STRETCH_PROB,
        pitch_shift_prob=AUG_PITCH_SHIFT_PROB,
        noise_prob=AUG_NOISE_PROB,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        checkpoint_path=MODEL_CHECKPOINT_PATH,
    )

    pipeline.train()
