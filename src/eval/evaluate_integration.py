from src.eval.evaluate import evaluate
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

def evaluate_parameters(audio_dir: str, output_csv = None, verbose = 0):
    """
    Simple function to wrap the provided evaluate code in a function that can be called during training.
    """
    assert os.getenv('MODEL_PATH') is not None, "MODEL_PATH environment variable is not set!"
    assert os.getenv('SCORER_PATH') is not None, "SCORER_PATH environment variable is not set!"
 
    parser = argparse.ArgumentParser(description="Transcribe audio files and calculate metrics.")
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--text_file', type=str, required=True, help='Path to the text file containing true text')
    parser.add_argument('--output_csv', type=str, default=None, help='Optional path to output CSV file. If not provided, no CSV file will be saved.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to DeepSpeech model file')
    parser.add_argument('--scorer_path', type=str, required=True, help='Path to DeepSpeech scorer file')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0 for minimal output)')
    
    arg_str = f"--audio_dir {audio_dir} --text_file {os.getenv('TEXT_FILE_PATH')} --model_path {os.getenv('MODEL_PATH')} --scorer_path {os.getenv('SCORER_PATH')} --verbose {verbose}"

    if output_csv:
        arg_str += f" --output_csv {output_csv}"

    args = parser.parse_args(arg_str.split())

    return evaluate(args)