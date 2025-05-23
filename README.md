# faster-whisper-batch

**Batch-transcribe audio to text** with real-world timestamps using [faster-whisper](https://github.com/guillaumekln/faster-whisper).

## 🔧 Features
- Auto-detects all `.wav` & `.m4a` files in a folder
- Skips files that already have transcripts
- Writes elapsed time and real-time of recording for each line in transcript 
- Outputs human-readable transcripts with 12-hour clock format

## Before Running
- ON LINE 8 INSERT YOUR DIRECTORY WITH AUDIO FILES TO TRANSCRIBE 
- On line 9 you can choose the model size: "tiny" "base" "small" "medium" "large-v1" "large-v2" "large-v3" ADD ".en" for english only. Fast and less accurate --> Slow and more accurate 


## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
