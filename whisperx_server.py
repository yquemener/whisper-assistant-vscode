"""
This script sets up a server that watches a directory for new or renamed .wav files
and transcribes them using WhisperX.

Features:
- Watches a specified directory for .wav files
- Reacts to both new file creations and file renames
- Transcribes audio using WhisperX
- Saves transcription results as JSON files

Configuration is loaded from a 'whisperx_config.json' file in the same directory as this script.
"""

import os
import json
import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
import whisperx
import torch

def expand_user_path(path):
    return os.path.expanduser(path)

def load_config():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, 'whisperx_config.json')
    try:
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON in configuration file: {config_path}")
        sys.exit(1)

# Load configuration
config = load_config()

# Configuration
WATCH_DIRECTORY = expand_user_path(config.get('watch_directory', ''))
DEVICE = config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = config.get('compute_type', "float16" if DEVICE == "cuda" else "float32")
BATCH_SIZE = config.get('batch_size', 16 if DEVICE == "cuda" else 1)
LANGUAGE = config.get('language', "en")
MODEL_SIZE = config.get('model_size', "large-v2")

# Load WhisperX models
print("Loading WhisperX models...")
model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)

class AudioTranscriber(FileSystemEventHandler):
    def __init__(self):
        self.pending_files = []

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".wav"):
            self.pending_files.append(event.src_path)
            self.process_pending_files()

    def on_moved(self, event):
        if not event.is_directory and event.dest_path.endswith(".wav"):
            self.pending_files.append(event.dest_path)
            self.process_pending_files()

    def process_pending_files(self):
        while self.pending_files:
            audio_path = self.pending_files.pop(0)
            self.transcribe_audio(audio_path)

    def transcribe_audio(self, audio_path):
        try:
            print(f"Preparing to transcribe {audio_path}")
            time.sleep(0.1)  # Wait for 100 milliseconds to ensure file operations are complete
            
            if not os.path.exists(audio_path):
                print(f"Error: {audio_path} does not exist.")
                return

            # Transcription with WhisperX
            audio = whisperx.load_audio(audio_path)
            if len(audio) == 0:
                print(f"Error: No audio data loaded from {audio_path}")
                return

            result = model.transcribe(audio, batch_size=BATCH_SIZE)
            
            print(f"Initial transcription result: {result}")  # Debug print
            
            # Check if 'language' is in the result
            if 'language' not in result:
                print(f"Warning: Language not detected in {audio_path}. Using default language.")
                result['language'] = LANGUAGE

            # Alignment
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            
            print(f"Alignment result: {result}")  # Debug print

            # Prepare the result
            transcription = {
                "text": " ".join([segment["text"] for segment in result["segments"]]),
                "segments": result["segments"],
                "language": result.get("language", LANGUAGE)
            }

            # Save the result
            output_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".json"
            output_path = os.path.join(os.path.dirname(audio_path), output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)

            print(f"Transcription completed: {output_path}")

        except Exception as e:
            print(f"Error transcribing {audio_path}: {str(e)}")
            print(f"Error details: {type(e).__name__}, {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if not WATCH_DIRECTORY:
        print("Error: watch_directory not specified in configuration file.")
        sys.exit(1)

    expanded_watch_directory = expand_user_path(WATCH_DIRECTORY)
    if not os.path.isdir(expanded_watch_directory):
        print(f"The directory {expanded_watch_directory} does not exist.")
        sys.exit(1)

    event_handler = AudioTranscriber()
    observer = Observer()
    observer.schedule(event_handler, expanded_watch_directory, recursive=False)
    observer.start()

    print(f"Watching directory {expanded_watch_directory}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()