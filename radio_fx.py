import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter


# DSP Filters
def butter_bandpass(lowcut, highcut, fs, order=4):
    """Calculates the coefficients for a Butterworth bandpass filter."""
    nyq = 0.5 * fs  # Nyquist frequency (half the sample rate)
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Applies the bandpass filter to the audio data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def process_audio_file(input_path, output_path, noise_level=0.02, drive=4.0):
    try:
        # soundfile automatically reads .ogg and .wav, returning float data (-1.0 to 1.0)
        # The return order is (data, sample_rate)
        data, fs = sf.read(input_path)

        # Convert to mono if the audio is stereo (averaging the channels)
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Find the maximum amplitude to prevent division by zero
        max_amp = np.max(np.abs(data))
        if max_amp == 0:
            print(f"[-] Skipping {input_path}: file is empty.")
            return False

        # Data is usually normalized by sf.read, but we do it safely just in case
        data_normalized = data / max_amp

        # 1. Frequency Cut (Walkie-Talkie Effect)
        # Keeps only frequencies between 300 Hz and 3000 Hz
        filtered_data = apply_bandpass_filter(data_normalized, 300, 3000, fs)

        # 2. Distortion (Microphone Overdrive / Clipping)
        driven_data = filtered_data * drive
        clipped_data = np.clip(driven_data, -1.0, 1.0)

        # 3. Add White Noise (Radio Static)
        noise = np.random.normal(0, noise_level, clipped_data.shape)
        final_audio = clipped_data + noise

        # Clip one last time to ensure the noise doesn't push the signal out of bounds
        final_audio = np.clip(final_audio, -1.0, 1.0)

        # Save the result.
        # We enforce .wav format (even if input was .ogg) because Whisper prefers PCM .wav
        sf.write(output_path, final_audio, fs, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"[-] Error processing {input_path}: {e}")
        return False


def batch_process_radio(input_dir="test_audios", output_dir="radio_audios"):
    if not os.path.exists(input_dir):
        print(f"[!] Error: Directory '{input_dir}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[*] Created directory: {output_dir}")

    # Search for valid audio formats
    valid_extensions = (".wav", ".ogg")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"[!] No .ogg or .wav files found in '{input_dir}'.")
        return

    print(f"[*] Found {len(files)} files. Starting radio FX generation...\n")

    success_count = 0
    for filename in files:
        input_path = os.path.join(input_dir, filename)

        # Build output filename (always saving as .wav for ASR stability)
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"radio_{name_without_ext}.wav"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename} -> {output_filename}")
        if process_audio_file(input_path, output_path):
            success_count += 1

    print(
        f"\n[+] Done! Successfully processed {success_count} out of {len(files)} files."
    )
    print(f"[+] All tactical radio intercepts saved to: {output_dir}")


if __name__ == "__main__":
    batch_process_radio()
