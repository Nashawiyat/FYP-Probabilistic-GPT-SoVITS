import os
import wave
import shutil
import numpy as np

# --- Configuration ---
source_wavs_dir = r"..\LJSpeech-1.1\LJSpeech-1.1\wavs"
processed_root = r"..\LJSpeech-1.1\LJSpeech_Processed"
existing_folder = "3hours"  # Already processed folder to reuse clips from
target_folder = "6hours"
target_duration_seconds = 6 * 3600 

TARGET_SR = 24000  # Target sample rate (24 kHz)
SOURCE_SR = 22050  # LJSpeech original sample rate


def resample_wav(input_path, output_path, source_sr=SOURCE_SR, target_sr=TARGET_SR):
    """
    Resample a WAV file to target sample rate (24 kHz), mono, 16-bit PCM.
    Uses numpy linear interpolation for resampling.
    """
    with wave.open(input_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Convert raw bytes to numpy array
    if sampwidth == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
    elif sampwidth == 1:
        samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64) - 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Convert to mono if stereo
    if n_channels == 2:
        samples = (samples[0::2] + samples[1::2]) / 2.0

    # Resample using linear interpolation
    if framerate != target_sr:
        duration = len(samples) / framerate
        n_target_samples = int(duration * target_sr)
        x_original = np.linspace(0, duration, len(samples), endpoint=False)
        x_target = np.linspace(0, duration, n_target_samples, endpoint=False)
        samples = np.interp(x_target, x_original, samples)

    # Clip and convert back to 16-bit PCM
    samples = np.clip(samples, -32768, 32767).astype(np.int16)

    # Write output
    with wave.open(output_path, 'wb') as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(target_sr)
        wf_out.writeframes(samples.tobytes())

    return len(samples) / target_sr


def get_wav_duration(filepath):
    """Get duration of a WAV file in seconds."""
    with wave.open(filepath, 'rb') as wf:
        return wf.getnframes() / wf.getframerate()


def build_6hour_folder():
    existing_dir = os.path.join(processed_root, existing_folder)
    target_dir = os.path.join(processed_root, target_folder)
    os.makedirs(target_dir, exist_ok=True)

    # Get list of already-processed files from 3hours folder
    existing_files = sorted([
        f for f in os.listdir(existing_dir) if f.endswith(".wav")
    ])
    print(f"Found {len(existing_files)} files in '{existing_folder}' folder.")

    # Copy all 3hours files into 6hours folder
    total_duration = 0.0
    copied_ids = set()

    for i, filename in enumerate(existing_files):
        src = os.path.join(existing_dir, filename)
        dst = os.path.join(target_dir, filename)

        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        dur = get_wav_duration(dst)
        total_duration += dur
        copied_ids.add(os.path.splitext(filename)[0])

        if (i + 1) % 200 == 0:
            print(f"  Copied {i + 1}/{len(existing_files)} files "
                  f"({total_duration / 3600:.2f} hrs so far)")

    print(f"Copied {len(existing_files)} files from '{existing_folder}'. "
          f"Duration so far: {total_duration / 3600:.2f} hours.")

    # Get all source WAV files sorted, skip already-copied ones
    all_source_files = sorted([
        f for f in os.listdir(source_wavs_dir) if f.endswith(".wav")
    ])

    remaining_files = [
        f for f in all_source_files
        if os.path.splitext(f)[0] not in copied_ids
    ]
    print(f"Remaining source files to process: {len(remaining_files)}")

    # Resample and add new files until we reach 6 hours
    new_files_added = 0
    for filename in remaining_files:
        if total_duration >= target_duration_seconds:
            break

        src_path = os.path.join(source_wavs_dir, filename)
        dst_path = os.path.join(target_dir, filename)

        if os.path.exists(dst_path):
            # Already exists (shouldn't happen but just in case)
            dur = get_wav_duration(dst_path)
        else:
            dur = resample_wav(src_path, dst_path)

        total_duration += dur
        new_files_added += 1

        if new_files_added % 200 == 0:
            print(f"  Processed {new_files_added} new files "
                  f"({total_duration / 3600:.2f} hrs so far)")

    final_count = len(existing_files) + new_files_added
    print(f"\nDone! '{target_folder}' folder now has {final_count} files.")
    print(f"Total duration: {total_duration / 3600:.2f} hours "
          f"({total_duration:.1f} seconds)")


if __name__ == "__main__":
    build_6hour_folder()