# Dialects

The goal of this project is to develop an ML pipeline that:

- Scrapes the [Dialect Archive](https://www.dialectsarchive.com/) page for speaker audio and transcription
- Preprocesses the audio and text files
- Creates synthetic speech using the raw scraped data using a pre-trained zero-shot TTS model
- Combines the synthetic data with real speaker audio to be used in training or finetuning

We are using the dialect archive website due to the rich variation of speaker dialects and accents in the audio files. Similar to how humans may not understand an accent the first time they hear it, an automatic speech recognition model may struggle to understand or transcribe speech in a new accent. The diversity in the data allows our model to perform better with new data. 

## Directory
- [Things to Note](#Things-to-Note)
- [Scraper](#Scraper)
- [TTS Cloning](#TTS-Cloning)
- [Synthesizer](#Synthesizer)
- [Results](#Results)

### Things to Note

We are using the Assembly AI library in 02_prepocess_audio. An API key is needed in order for this script to work as intended. Create a .env script in the root directory and create an environment variable named MY_API_KEY in order for the script to work properly. 

### Scraper

The scraper looks into Dialect Archive and extracts the transcription as a .txt file, and the audio as a .wav file. The transcription is easy to extract because most recordings are done using the same speech.

### TTS Cloning

The TTS cloner script uses a pretrained synthesizer to learn the speaker's voice. 

### Synthesizer

The synthetic voice created in the TTS Cloning script is now used to create synthetic audio using a target text file. We are testing the synthetic voice and making sure that it handles new sentences well.

### Results

Results will be obtained from outside of the pipeline. A model will be trained using real data, real data + synthetic data, synthetic data with all other variables being kept the same. 

The performance of each iteration will be compared to determine if the synthetic data improves the model's performance or not. 