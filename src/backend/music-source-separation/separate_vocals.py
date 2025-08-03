import soundfile as sf
import numpy as np
import os
import sys
import subprocess

def separate_vocals(input_song_path, output_dir):
    # Helper to combine stems into instrumental
    def combine_stems_to_instrumental(stem_dir, output_path):
        stems = ['bass.wav', 'drums.wav', 'other.wav']
        data = []
        samplerate = None
        found_stems = []
        for stem in stems:
            stem_path = os.path.join(stem_dir, stem)
            if os.path.exists(stem_path):
                print(f"Found stem: {stem_path}")
                audio, sr = sf.read(stem_path)
                if samplerate is None:
                    samplerate = sr
                data.append(audio)
                found_stems.append(stem)
            else:
                print(f"Missing stem: {stem_path}")
        if not data:
            raise FileNotFoundError(f"No instrumental stems found in Demucs output. Checked for: {', '.join(stems)}. Found: {', '.join(found_stems) if found_stems else 'none'}.")
        # Pad shorter arrays to match the longest
        max_len = max([len(d) for d in data])
        data = [np.pad(d, ((0, max_len - len(d)), (0, 0)), mode='constant') if d.ndim == 2 else np.pad(d, (0, max_len - len(d)), mode='constant') for d in data]
        instrumental = np.sum(data, axis=0)
        sf.write(output_path, instrumental, samplerate)
    """
    Separates vocals from a song using Demucs and saves the instrumental (no vocals) version.
    Args:
        input_song_path (str): Path to the input song file.
        output_dir (str): Directory to save the output.
    Returns:
        tuple: (instrumental_path, vocals_path) where
            instrumental_path (str): Path to the instrumental (no vocals) audio file.
            vocals_path (str): Path to the vocals audio file (for Whisper AI).
    """
    # Ensure Demucs is installed, install if missing
    try:
        import demucs
    except ImportError:
        print("Demucs not found. Installing demucs and dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'demucs'])
        try:
            import demucs
        except ImportError:
            print("Failed to install Demucs. Please check your Python environment.")
            sys.exit(1)

    # Demucs outputs to output_dir/htdemucs/song_name/{stems}
    song_name = os.path.splitext(os.path.basename(input_song_path))[0]
    demucs_output_dir = os.path.join(output_dir, 'htdemucs', song_name)
    if os.path.exists(demucs_output_dir):
        print(f"Demucs output already exists for {song_name}, skipping separation.")
    else:
        # Run Demucs to separate sources
        command = [
            sys.executable, '-m', 'demucs', input_song_path, '-o', output_dir
        ]
        subprocess.run(command, check=True)
        print(f"output directory: {output_dir}")
    print(f"Demucs output directory: {demucs_output_dir}")

    # Combine stems for instrumental and save in demucs subfolder
    instrumental_path = os.path.join(demucs_output_dir, 'instrumental.wav')
    try:
        combine_stems_to_instrumental(demucs_output_dir, instrumental_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not create {instrumental_path}: {e}")

    # Vocals
    vocals_path = os.path.join(output_dir, 'htdemucs', song_name, 'vocals.wav')
    if not os.path.exists(vocals_path):
        print(f"vocals.wav not found at {vocals_path}")
        # Print all files in the directory for debugging
        if os.path.exists(os.path.join(output_dir, 'htdemucs', song_name)):
            print("Files in vocals directory:")
            for f in os.listdir(os.path.join(output_dir, 'htdemucs', song_name)):
                print(f"  - {f}")
        raise FileNotFoundError("Vocals file not found after Demucs separation.")

    return instrumental_path, vocals_path

if __name__ == "__main__":
    # Process all songs in ../resources/songs (resources is now outside src)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    songs_dir = os.path.join(base_dir, 'resources', 'songs')
    output_folder = os.path.join(base_dir, 'resources', 'final')
    os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(songs_dir):
        print(f"Songs directory {songs_dir} does not exist. Please check the path.")
        sys.exit(1)

    for filename in os.listdir(songs_dir):
        if filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
            input_song = os.path.join(songs_dir, filename)
            print(f"Processing: {input_song}")
            try:
                instrumental, vocals = separate_vocals(input_song, output_folder)
                print(f"Instrumental saved at: {instrumental}")
                print(f"Vocals saved at: {vocals} (can be passed to Whisper AI)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
