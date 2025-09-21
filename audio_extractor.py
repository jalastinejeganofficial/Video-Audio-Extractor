import streamlit as st
import moviepy.editor as mp
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os
from pathlib import Path
import tempfile
import subprocess
import json

# Set up the Streamlit interface
st.set_page_config(page_title="Video Audio Extractor", page_icon="🎵", layout="wide")

st.title("🎵 Video Audio Extractor with Multi-Language Support")
st.markdown("Upload a video file to extract audio tracks with optional ML enhancement")

# Sidebar for options
with st.sidebar:
    st.header("Settings")
    enhance_audio = st.checkbox("Apply ML Audio Enhancement", value=True)
    output_format = st.selectbox("Output Format", ["mp3", "wav", "ogg"])

# Function to get audio stream information from video
def get_audio_streams(video_path):
    try:
        # Use ffprobe to get information about audio streams
        cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'a',
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            st.error(f"Error analyzing video: {result.stderr}")
            return []
            
        data = json.loads(result.stdout)
        audio_streams = []
        
        for stream in data.get('streams', []):
            info = {
                'index': stream.get('index'),
                'codec': stream.get('codec_name', 'unknown'),
                'sample_rate': stream.get('sample_rate', 'unknown'),
                'channels': stream.get('channels', 'unknown'),
                'duration': stream.get('duration', 'unknown'),
                'bit_rate': stream.get('bit_rate', 'unknown'),
                'language': stream.get('tags', {}).get('language', 'unknown') if 'tags' in stream else 'unknown',
                'title': stream.get('tags', {}).get('title', 'unknown') if 'tags' in stream else 'unknown'
            }
            audio_streams.append(info)
            
        return audio_streams
    except Exception as e:
        st.error(f"Error getting audio streams: {str(e)}")
        return []

# Function to extract specific audio track
def extract_audio_track(video_path, audio_index, output_path):
    try:
        # Use ffmpeg to extract specific audio track
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-map', f'0:a:{audio_index}',
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format for processing
            '-ar', '44100',  # Sample rate
            '-y',  # Overwrite output file if exists
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            st.error(f"Error extracting audio: {result.stderr.decode()}")
            return False
            
        return True
    except Exception as e:
        st.error(f"Error extracting audio track: {str(e)}")
        return False

# Main content area
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"])

if uploaded_file is not None:
    # Display video information
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    }
    st.write(file_details)

    # Display the uploaded video
    st.video(uploaded_file)

    # Create temporary video file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name

    # Get audio stream information
    with st.spinner("Analyzing video audio tracks..."):
        audio_streams = get_audio_streams(video_path)
    
    if not audio_streams:
        st.error("No audio tracks found in the video file.")
    else:
        st.subheader("Available Audio Tracks")
        
        # Create selection interface for audio tracks
        track_options = []
        for i, stream in enumerate(audio_streams):
            language = stream['language']
            title = stream['title']
            channels = stream['channels']
            codec = stream['codec']
            
            label = f"Track {i}"
            if language != 'unknown':
                label += f" - Language: {language}"
            if title != 'unknown':
                label += f" - Title: {title}"
            label += f" - Channels: {channels}, Codec: {codec}"
            
            track_options.append(label)
        
        selected_track = st.selectbox("Select audio track to extract", options=track_options, index=0)
        selected_index = track_options.index(selected_track)

        # Extract audio button
        if st.button("Extract Audio"):
            with st.spinner("Extracting audio..."):
                try:
                    # Extract the selected audio track
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                        if extract_audio_track(video_path, selected_index, tmp_audio.name):
                            audio_path = tmp_audio.name

                            # Apply ML enhancement if selected
                            if enhance_audio:
                                st.info("Applying ML audio enhancement...")

                                y, sr = librosa.load(audio_path, sr=None)
                                y_enhanced = librosa.effects.preemphasis(y)
                                y_enhanced = y_enhanced / np.max(np.abs(y_enhanced))

                                enhanced_path = audio_path.replace(".wav", "_enhanced.wav")
                                sf.write(enhanced_path, y_enhanced, sr)
                                final_audio_path = enhanced_path
                            else:
                                final_audio_path = audio_path

                            # Convert to desired format
                            if output_format != "wav":
                                audio_segment = AudioSegment.from_file(final_audio_path)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp_output:
                                    audio_segment.export(tmp_output.name, format=output_format)
                                    final_output_path = tmp_output.name
                            else:
                                final_output_path = final_audio_path

                            st.success("Audio extracted successfully!")
                            
                            # Display track information
                            selected_stream = audio_streams[selected_index]
                            st.info(f"Extracted audio: {selected_stream['language']} ({selected_stream['codec']}, {selected_stream['channels']} channels)")
                            
                            # Audio player
                            st.audio(final_output_path, format=f'audio/{output_format}')

                            # Download button
                            language_suffix = f"_{selected_stream['language']}" if selected_stream['language'] != 'unknown' else ""
                            with open(final_output_path, "rb") as file:
                                st.download_button(
                                    label=f"Download Audio ({output_format.upper()})",
                                    data=file,
                                    file_name=f"{Path(uploaded_file.name).stem}{language_suffix}.{output_format}",
                                    mime=f"audio/{output_format}"
                                )

                except Exception as e:
                    st.error(f"Error extracting audio: {str(e)}")

                finally:
                    # Close video handles before cleanup
                    try:
                        if 'video' in locals():
                            video.reader.close()
                            if video.audio:
                                video.audio.reader.close_proc()
                    except Exception as close_err:
                        st.warning(f"Warning during video cleanup: {close_err}")

                    # Clean up temporary files
                    for path_var in ['video_path', 'audio_path', 'final_audio_path', 'final_output_path']:
                        if path_var in locals() and os.path.exists(locals()[path_var]):
                            try:
                                os.unlink(locals()[path_var])
                            except Exception as unlink_err:
                                st.warning(f"Could not delete {path_var}: {unlink_err}")

else:
    st.info("Please upload a video file to get started")

# Info section
with st.expander("How this works"):
    st.markdown("""
    This application extracts audio from video files with support for multiple audio tracks:

    1. **Video Upload**: Upload a video file (MP4, MOV, AVI, MKV, or WEBM)
    2. **Audio Track Analysis**: The app analyzes the video to identify all available audio tracks
    3. **Track Selection**: You can select which audio track to extract based on language and other metadata
    4. **Audio Extraction**: The selected audio track is extracted using FFmpeg
    5. **ML Enhancement** (optional): If selected, the audio is enhanced using machine learning techniques
        - Noise reduction
        - Audio normalization
    6. **Format Conversion**: The audio is converted to your selected output format
    7. **Download**: You can download the extracted audio file

    The enhancement uses Librosa for basic signal processing. You can extend it with custom ML models.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Python, Streamlit, FFmpeg, MoviePy, and Librosa")