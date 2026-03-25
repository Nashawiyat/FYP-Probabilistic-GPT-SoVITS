import os
import csv

# Change to approrpiate path
audio_root = r"..\LJSpeech-1.1\LJSpeech_Processed" 
csv_path = r"..\LJSpeech-1.1\LJSpeech-1.1\metadata.csv"

REMOTE_BASE_PATH = "C:/Temp/FYP/LJSpeech-1.1/LJSpeech_Processed"

folders_to_process = ["10min", "30min", "1hour", "3hours"]
# folders_to_process = ["6hours"]

speaker_name = "Linda"
language = "en"

def generate_multi_list():
    # Loading CSV metadata into a dictionary
    transcripts = {}
    if not os.path.exists(csv_path):
        print(f"Error: Metadata not found at {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) >= 3:
                # Key: LJ001-0001, Value: Normalized Text
                transcripts[row[0]] = row[2]

    # Iterating through each specific duration folder
    for folder in folders_to_process:
        folder_path = os.path.join(audio_root, folder)
        output_filename = f"{folder.replace(' ', '_')}.list"
        
        if not os.path.exists(folder_path):
            print(f"Skipping: Folder '{folder}' not found at {os.path.abspath(folder_path)}")
            continue

        lines_written = 0
        abs_folder_path = os.path.abspath(folder_path)

        with open(output_filename, 'w', encoding='utf-8') as out:
            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    # Remove .wav to get the ID for the CSV lookup
                    file_id = os.path.splitext(filename)[0]
                    
                    if file_id in transcripts:
                        text = transcripts[file_id]
                        
                        full_audio_path = f"{REMOTE_BASE_PATH}/{folder}/{filename}"
                        
                        # Format: vocal_path|speaker_name|language|text
                        line = f"{full_audio_path}|{speaker_name}|{language}|{text}\n"
                        out.write(line)
                        lines_written += 1
        
        print(f"Created {output_filename} with {lines_written} entries from folder '{folder}'.")

if __name__ == "__main__":
    generate_multi_list()