import os
import io
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

# Configuration
DATASET_NAME = "MikhailT/hifi-tts"
TARGET_SR = 24000
TARGET_DURATIONS = {
    "10min": 10 * 60,
    "30min": 30 * 60,
    "1hour": 60 * 60,
    "3hours": 3 * 60 * 60
}

# Setup Folders
BASE_DIR = r"..\HiFi-TTS"
FOLDERS = list(TARGET_DURATIONS.keys())
for folder in FOLDERS:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

def resample_linear(audio_array, orig_sr, target_sr):
    """Resamples audio using numpy linear interpolation."""
    if orig_sr == target_sr:
        return audio_array
    
    duration = len(audio_array) / orig_sr
    new_length = int(duration * target_sr)
    
    old_indices = np.arange(len(audio_array))
    new_indices = np.linspace(0, len(audio_array) - 1, new_length)
    
    resampled_audio = np.interp(new_indices, old_indices, audio_array)
    return resampled_audio

def process_and_save(audio_array, orig_sr, text, filename, current_duration):
    """Processes audio to 24kHz, Mono, 16-bit PCM and saves to overlapping folders."""
    # Convert to Mono
    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
        audio_array = np.mean(audio_array, axis=1)
        
    # Resample using linear interpolation
    resampled_audio = resample_linear(audio_array, orig_sr, TARGET_SR)
    
    # Normalize and convert to 16-bit PCM
    max_val = np.max(np.abs(resampled_audio))
    if max_val > 0:
        resampled_audio = resampled_audio / max_val
    audio_16bit = np.int16(resampled_audio * 32767)
    
    # Save to appropriate overlapping folders
    for folder, max_duration in TARGET_DURATIONS.items():
        if current_duration <= max_duration:
            folder_path = os.path.join(BASE_DIR, folder)
            file_path = os.path.join(folder_path, filename)
            sf.write(file_path, audio_16bit, TARGET_SR, subtype='PCM_16')
            
            # Append to Transcript file
            transcript_path = os.path.join(folder_path, "transcripts.txt")
            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write(f"{file_path}|{text}\n")

def main():
    print("Streaming dataset from Hugging Face...")
    dataset = load_dataset(DATASET_NAME, "clean", split="train", streaming=True)
    
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    target_speaker = None
    accumulated_duration = 0.0
    max_total_duration = TARGET_DURATIONS["3hours"]
    
    for item in dataset:
        speaker = item['speaker']
        
        if target_speaker is None:
            target_speaker = speaker
            print(f"Locked onto Speaker ID: {target_speaker}")
            
        if speaker != target_speaker:
            continue
            
        raw_bytes = item['audio']['bytes']
        audio_data, orig_sr = sf.read(io.BytesIO(raw_bytes))
        
        duration = item['duration']
        text = item.get('ext_normalized', item.get('text_normalized', item['text'])) 
        
        if accumulated_duration + duration > max_total_duration:
            break
            
        filename = f"{target_speaker}_{int(accumulated_duration)}.wav"
        
        process_and_save(
            audio_array=audio_data, 
            orig_sr=orig_sr, 
            text=text, 
            filename=filename, 
            current_duration=accumulated_duration
        )
        
        accumulated_duration += duration
        
        if int(accumulated_duration) % 300 == 0:
            print(f"Collected: {accumulated_duration / 60:.2f} minutes...")

    print(f"Finished! Total audio collected: {accumulated_duration / 3600:.2f} hours.")

if __name__ == "__main__":
    main()