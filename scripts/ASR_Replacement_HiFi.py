import os

base_dir = r"..\HiFi-TTS" # Change to approrpiate path
master_transcript = os.path.join(base_dir, "transcripts.txt")
folders_to_process = ["10min", "30min", "1hour", "3hours"]

speaker_name = "HiFi_Speaker"
language = "en"

# HPC server path (change to appropriate path)
REMOTE_BASE_PATH = "/mnt/data/home/aa2249/FYP/HiFi-TTS"

def generate_hifi_lists():
    # Load transcript
    transcripts = {}
    if not os.path.exists(master_transcript):
        print(f"Error: Master transcript not found at {master_transcript}")
        return

    with open(master_transcript, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|', 1)
            if len(parts) == 2:
                # Extract just the filename
                filename = os.path.basename(parts[0].replace('\\', '/'))
                text = parts[1]
                transcripts[filename] = text

    # Iterate through each duration folder to generate its specific .list file
    for folder in folders_to_process:
        folder_path = os.path.join(base_dir, folder)
        output_filename = f"{folder}.list"
        
        if not os.path.exists(folder_path):
            print(f"Skipping: Folder '{folder}' not found.")
            continue

        lines_written = 0
        
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    if filename in transcripts:
                        text = transcripts[filename]
                        
                        linux_audio_path = f"{REMOTE_BASE_PATH}/{folder}/{filename}"
                        
                        # GPT-SoVITS Format: vocal_path|speaker_name|language|text
                        list_line = f"{linux_audio_path}|{speaker_name}|{language}|{text}\n"
                        outfile.write(list_line)
                        lines_written += 1
                        
        print(f"Success: Created {output_filename} with {lines_written} lines.")

if __name__ == "__main__":
    generate_hifi_lists()