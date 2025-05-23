import os
import time
from datetime import datetime
from datetime import timedelta
from faster_whisper import WhisperModel

# Settings 
folder_path = r"C:\Users\USERNAME\Desktop\faster-whisper-batch" #INSERT YOUR DIRECTORY WITH AUDIO FILES TO TRANSCRIBE
model_size = "small"                                            #CAN CHOOSE "tiny" "base" "small" "medium" "large-v1" "large-v2" "large-v3" ADD ".en" for english only
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# List all .wav and .m4a files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".wav", ".m4a")):
        audio_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]
        transcript_path = os.path.join(folder_path, f"{base_name} transcript.txt")

        # Skip if transcript already exists
        if os.path.exists(transcript_path):
            print(f"Skipping {filename} (transcript already exists)")
            continue

        current_time = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
        print(f"Processing {filename} started {current_time} ...")
        

        # Get file timestamp and readable time
        timestamp = os.path.getmtime(audio_path)
        readable_time = datetime.fromtimestamp(timestamp).strftime('%B %d, %Y\n%I:%M:%S %p')

        # Transcribe
        start_time = time.time()
        segments, info = model.transcribe(audio_path, language="en")
        

        # Write transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"{readable_time}\nFaster_Whisper model used: {model_size}\n_______________________________________________________________________________\n\n")
            for segment in segments:
                real_start = datetime.fromtimestamp(timestamp + segment.start).strftime('%I:%M:%S %p')
                real_end = datetime.fromtimestamp(timestamp + segment.end).strftime('%I:%M:%S %p')
                text = segment.text.strip()
                start_hms = str(timedelta(seconds=int(segment.start)))
                end_hms = str(timedelta(seconds=int(segment.end)))
                f.write(f"{real_start} --> {real_end}:\n{start_hms} --> {end_hms}\n{text}\n\n")

        end_time = time.time() 
        worktime = str(timedelta(seconds=int(end_time - start_time)))
        print(f"Done with {filename} finished {current_time} (took {(worktime)})\n")