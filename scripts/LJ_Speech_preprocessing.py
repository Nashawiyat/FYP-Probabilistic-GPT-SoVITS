import os
import csv

# Root folder of audio volume subfolders (10_mins, 30_mins, etc.)
base_audio_root = r"..\LJSpeech-1.1\LJSpeech_Processed" 
csv_path = r"..\LJSpeech-1.1\LJSpeech-1.1\metadata.csv"

# Map folder names to their intended output .list files
dataset_folders = {
    "10mins": "ljspeech_10m.list",
    "30mins": "ljspeech_30m.list",
    "1hour":  "ljspeech_1h.list",
    "3hours": "ljspeech_3h.list",
    "6hours": "ljspeech_6h.list"
}

speaker_name = "Linda"
language = "en"

def generate_multi_list():
    # Load the CSV metadata
    transcripts = {}
    print("Loading metadata...")
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) >= 3:
                transcripts[row[0]] = row[2]

    # Process each folder
    for folder_name, output_file in dataset_folders.items():
        folder_path = os.path.join(base_audio_root, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_name}: Folder not found at {folder_path}")
            continue

        lines_written = 0
        abs_folder_path = os.path.abspath(folder_path)

        with open(output_file, 'w', encoding='utf-8') as out:
            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    file_id = os.path.splitext(filename)[0]
                    
                    if file_id in transcripts:
                        normalized_text = transcripts[file_id]
                        full_audio_path = os.path.join(abs_folder_path, filename)
                        
                        # Format: vocal_path|speaker_name|language|text
                        line = f"{full_audio_path}|{speaker_name}|{language}|{normalized_text}\n"
                        out.write(line)
                        lines_written += 1
                    else:
                        print(f"Warning: No transcript for {filename} in {folder_name}")

        print(f"Done! {output_file} created with {lines_written} entries.")

if __name__ == "__main__":
    generate_multi_list()