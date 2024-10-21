import streamlit as st
import moviepy.editor as mp
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import openai
import tempfile
import os

# Set up API keys
openai.api_key = '22ec84421ec24230a3638d1b51e3a7dc'


# Function to upload a video file
def upload_video():
    video_file = st.file_uploader("Upload a video with improper audio", type=["mp4", "avi"])
    return video_file


# Function to extract audio from video
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path


# Function to transcribe audio using Google Speech-to-text
def transcribe_audio(audio_file):
    client = speech.SpeechClient()
    with open(audio_file, "rb") as audio:
        audio_content = audio.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)

    transcription = ''.join([result.alternatives[0].transcript for result in response.results])
    return transcription


# Function to correct transcription using GPT-4o model from Azure OpenAI
def correct_transcription(transcription):
    prompt = f"Correct the following transcription by removing filler words like 'um', 'hmm' and fix any grammatical mistakes: {transcription}"
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=prompt,
        max_tokens=500
    )
    corrected_text = response.choices[0].text.strip()
    return corrected_text


# Function to generate speech from text using Google Text-to-Speech
def generate_audio_from_text(corrected_text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=corrected_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-JennyNeural"  # Use 'Jenny' or other journey voice models from Google Text-to-Speech
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    output_audio_file = "output_audio.mp3"
    with open(output_audio_file, "wb") as out:
        out.write(response.audio_content)
    return output_audio_file


# Function to replace audio in video using MoviePy
def replace_audio_in_video(video_path, new_audio_path):
    video = mp.VideoFileClip(video_path)
    new_audio = mp.AudioFileClip(new_audio_path)
    final_video = video.set_audio(new_audio)
    output_video_path = "final_video.mp4"
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    return output_video_path


# Main Streamlit Application
def main():
    st.title("AI-Powered Video Audio Replacement")

    # Step 1: Upload video file
    video_file = upload_video()

    if video_file:
        # Save video to a temporary file
        video_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_temp_file.write(video_file.read())
        video_temp_file.close()

        # Step 2: Extract audio from video
        st.write("Extracting audio from video...")
        extracted_audio_file = extract_audio(video_temp_file.name)

        # Step 3: Transcribe the extracted audio using Google Speech-to-Text
        st.write("Transcribing audio...")
        transcription = transcribe_audio(extracted_audio_file)
        st.write(f"Original Transcription: {transcription}")

        # Step 4: Use GPT-4o to correct the transcription
        st.write("Correcting transcription with GPT-4o...")
        corrected_transcription = correct_transcription(transcription)
        st.write(f"Corrected Transcription: {corrected_transcription}")

        # Step 5: Generate new audio from corrected transcription using Google Text-to-Speech
        st.write("Generating corrected audio...")
        new_audio_file = generate_audio_from_text(corrected_transcription)

        # Step 6: Replace audio in the original video with the new corrected audio
        st.write("Replacing old audio with new audio...")
        final_video_file = replace_audio_in_video(video_temp_file.name, new_audio_file)

        # Step 7: Provide download link for the final video
        st.write("Processing complete. Download the final video below.")
        with open(final_video_file, "rb") as video_output:
            st.download_button("Download Final Video", data=video_output, file_name="final_video.mp4", mime="video/mp4")

        # Cleanup temporary files
        os.remove(video_temp_file.name)
        os.remove(extracted_audio_file)
        os.remove(new_audio_file)


if __name__ == "__main__":
    main()
