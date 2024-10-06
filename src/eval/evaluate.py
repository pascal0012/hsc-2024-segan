import argparse
import os

import jiwer
import librosa
import numpy as np
import pandas as pd
from deepspeech import Model

# This file was adapted from the provided version
# for the Helsinki Speech Challenge 2024 at:
# https://zenodo.org/records/11380835/files/evaluate.py


def process_and_transcribe(model, audio_file):
    # Load and resample the audio file to 16kHz
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    # Convert the audio signal from floating point to 16-bit PCM
    audio_int16 = (audio * np.iinfo(np.int16).max).astype(np.int16)
    # Perform speech-to-text with model
    transcription = model.stt(audio_int16)
    return transcription


def replace_dashes(text):
    # Replace dashes '-' with spaces ' ', avoids some amgiguities.
    return text.replace("-", " ")


def replace_z(text):
    # Replace z with s, avoids confusion between british and american english
    return text.replace("z", "s")


def normalize_us_spelling(text):
    # Maps some common words in the dataset from british to american
    spelling_corrections = {
        "behaviour": "behavior",
        "colour": "color",
        "favour": "favor",
        "flavour": "flavor",
        "honour": "honor",
        "humour": "humor",
        "labour": "labor",
        "neighbour": "neighbor",
        "odour": "odor",
        "savour": "savor",
        "armour": "armor",
        "clamour": "clamor",
        "enamoured": "enamored",
        "favourable": "favorable",
        "favourite": "favorite",
        "glamour": "glamor",
        "rumour": "rumor",
        "valour": "valor",
        "vigour": "vigor",
        "harbour": "harbor",
        "mould": "mold",
        "plough": "plow",
        "saviour": "savior",
        "splendour": "splendor",
        "tumour": "tumor",
        "theatre": "theater",
        "centre": "center",
        "fibre": "fiber",
        "litre": "liter",
        "metre": "meter",
        "labour": "labor",
        "labourer": "laborer",
        "kilometre": "kilometer",
    }

    for british, american in spelling_corrections.items():
        text = text.replace(british, american)
    return text


def calculate_metrics(original_text, transcribed_text, transformation):
    # Apply transformations
    transformed_original = transformation(original_text)
    transformed_transcribed = transformation(transcribed_text)

    # Ensure non-empty input for calculations. If empty input, set to max error.
    if not transformed_original.strip() or not transformed_transcribed.strip():
        print(
            f"Empty text after transformation: Original text - {original_text}, Transcribed text - {transcribed_text}"
        )
        return {"WER": 1.0, "CER": 1.0, "MER": 1.0, "WIL": 1.0, "WIP": 0.0}

    try:
        measures = jiwer.compute_measures(
            transformed_original,
            transformed_transcribed,
        )
        # Additionally for the calculation of CER, remove all spaces
        cer = jiwer.cer(
            transformed_original.replace(" ", ""),
            transformed_transcribed.replace(" ", ""),
        )

        return {
            "WER": measures["wer"],
            "CER": cer,
            "MER": measures["mer"],
            "WIL": measures["wil"],
            "WIP": measures["wip"],
        }

    # Error handling
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        print(f"Transformed original text: {transformed_original}")
        print(f"Transformed transcribed text: {transformed_transcribed}")
        return None


def evaluate(args):
    # Load the DeepSpeech model
    model = Model(args.model_path)
    model.enableExternalScorer(args.scorer_path)

    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            normalize_us_spelling,
            jiwer.ExpandCommonEnglishContractions(),
            replace_dashes,
            replace_z,
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.Strip(),
        ]
    )

    # Directory and file processing
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".wav")]
    transcriptions = []

    # List to store the result
    full_result = []
    cers = {}
    amounts = {}

    with open(args.text_file, "r") as file:
        for audio_file in sorted(audio_files):
            full_path = os.path.join(args.audio_dir, audio_file)
            if args.verbose > 0:
                print(f"Processing and transcribing {audio_file}...")

            transcribed_text = process_and_transcribe(model, full_path)
            transcriptions.append(transcribed_text)

            line = file.readline()
            parts = line.strip().split("\t")
            while parts[0] != audio_file:
                line = file.readline()
                parts = line.strip().split("\t")

            if parts[0] == audio_file:
                original_text = parts[1]
                if args.verbose > 0:
                    print(f"Transcription: {transcribed_text}")
                    print(f"True: {original_text}")
                metrics = calculate_metrics(
                    original_text, transcribed_text, transformation
                )

                # Custom logging metrics
                task = get_task_from_filepath(audio_file)
                if task not in cers:
                    cers[task] = 0
                    amounts[task] = 0

                cers[task] += metrics["CER"]
                amounts[task] += 1
                result = {
                    "Filename": audio_file,
                    "Original Text": original_text,
                    "Transcribed Text": transcribed_text,
                }
                if metrics:
                    result.update(metrics)
                    if args.verbose > 0:
                        for metric, value in metrics.items():
                            print(f"{metric}: {value:.2f}")

                full_result.append(result)

    # Save results to CSV
    df = pd.DataFrame(full_result)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")

    # Normalize CER
    for task in cers:
        cers[task] /= amounts[task]

    # print(f"Mean CER: {df['CER'].mean()}")
    # print("CER Quantiles:")
    # print(df['CER'].quantile([0.25, 0.5, 0.75]))
    return df["CER"].mean(), cers, transcriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio files and calculate metrics."
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        required=True,
        help="Path to the text file containing true text",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to output CSV file. If not provided, no CSV file will be saved.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to DeepSpeech model file"
    )
    parser.add_argument(
        "--scorer_path", type=str, required=True, help="Path to DeepSpeech scorer file"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level (0 for minimal output)"
    )

    args = parser.parse_args()
    print(f"Mean CER: {evaluate(args)}")


def get_task_from_filepath(file_path: str):
    filename = os.path.basename(file_path)

    return "_".join(filename.split("_")[:4])
