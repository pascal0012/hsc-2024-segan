# Diffusion-SEGAN: Speech Enhancement Generative Adversarial Network (trained with diffusion)

This repository contains an implementation of the [Speech Enhancement Generative Adversarial Network](https://arxiv.org/abs/1703.09452) introduced by S. Pascual et. al in 2017 combined with [Diffusion-GAN](https://arxiv.org/abs/2206.02262) (Z. Wang, 2023). It was created to participate in the [Helsinki Speech Challenge 2024](https://arxiv.org/abs/2406.04123) and is therefore designed to run on its provided dataset, however it should be easily adaptable to other tasks.

## Authors

Implemented by [Pascal Makossa](mailto:pascal.makossa@tu-dortmund.de) as a part of his bachelor thesis under the supervision of Sebastian Konietzny and Prof. Dr. Stefan Harmeling (Artifical Intelligence Group, Department of Computer Science, TU Dortmund University, Germany).

## Architecture

This implementation is mostly based on the SEGAN paper, which trains a discriminator $D$ that is trained to distinguish between generated and clean audio samples by optimizing for

$`\min_D V_\text{LSGAN}(D,G)=\frac{1}{2}\mathbb{E}_{x,x_c\sim p_{data}(x,x_c)}[(D(x,x_c)-1)^2]+\frac{1}{2}\mathbb{E}_{z\sim p_z(z),x_c\sim p_{data}(x_c)}[D(G(z,x_c),x_c)^2]`$

while the generator $G$ is trained to create samples that can not be recognised as fake. A weighted reconstruction loss term is added to ensure the generated samples represent the input file:

$`\min_G V_\text{LSGAN}(D,G)=\frac{1}{2}\mathbb{E}_{z\sim p_z(z),\tilde{x}\sim p_{data}(\tilde{x})}[(D(G(z,\tilde{x}),\tilde{x})-1)^2]+\lambda\|G(z,\tilde{x})-x\|_1`$

To improve training performance, additional noise is added to the discriminator input $x$ according to

$`y=\sqrt{\overline{a_t}}x-\sqrt{1-\overline{a_t}}\sigma\varepsilon`$ with $`\overline{a_t}=\prod_{s=1}^ta_s`$ and $`\varepsilon\sim\mathcal{N}(0,\mathcal{I})`$

while $`t`$ is sampled from an automatically adjusting range $`[0,T]`$ depending on the discriminator loss. The $`\alpha_s`$ are uniformly distributed with:

$`0.997=\alpha_0>\alpha_1>\dots>\alpha_{T-1}>\alpha_T=0.98`$

Both networks are supplied with batches of audio chunks of length 16384, which are sampled randomly from the entire file for training. For enhancement with a trained model the entire file is split into these chunks with 50% overlap, which are then recombined by averaging over the duplicate parts after passing through the generator.

In addition, the recorded and clean audio files are aligned using cross-correlation to train the networks. A pre-emphasis filter is applied which is reversed after the enhancement to ensure that no speech information is lost during the network pass.

## Installation

Python 3.9 is required for evaluation with the DeepSpeech model, for training and enhancement newer versions work aswell. It is recommended to create a **virtual environment** to install compatible versions of the following packages via `pip`:

- torch (with torchaudio)
- python-dotenv
- numpy
- soundfile

After that simply install the package itself by running `pip install .` in the root directory.

### Training Requirements

Furthermore the training loop requires some additional configuration steps.

#### Additional packages

The training requires packages to be installed mostly for calculating the CER:

- wandb
- librosa
- pandas
- jiwer
- deepspeech
- numpy < 2.0

#### Deepspeech Model

The deepspeech model and scorer need to be stored in the `model` folder. We used the v0.9.3 model that can be found [here](https://deepspeech.readthedocs.io/en/r0.9/).

#### Environment Variables

On top of that, a `.env` file needs to be created with the following values (All paths have to be relative from the `train.py` script).

|Key |Description |
|----|------|
|`MODEL_PATH`|Path to model|
|`SCORER_PATH`|Path to scorer|
|`TEXT_FILE_PATH`|Path to file with transcriptions|
|`DATA_PATH`|Path to input dataset structured in tasks|
|`CLEAN_PATH`|Path to clean data in each task folder|
|`RECORDED_PATH`|Path to noisy data in each task folder|
|`OUTPUT_PATH`|Path to folder for testing results|
|`DISABLE_WANDB`|Set to `True` to disable logging to wandb|

A partial configuration for the used setup is provided here:

```env
DATA_PATH=../data
CLEAN_PATH=Clean
RECORDED_PATH=Recorded
OUTPUT_PATH=../output
DISABLE_WANDB=True
```

#### Transcriptions

The correct transcriptions need to be stored under the provided path in the following format:

```txt
file_name.wav   Original text
```

## Running

For all scripts, navigate to the `scripts` subfolder before executing them.

### Enhancement

To enhance all audio files contained in a directory, run the `main.py` file with the following *unnamed* command line arguments:

| Argument | Description |
|----------|-------------|
|`input_dir`|Relative path to recorded files|
|`output_dir`|Relative path to store enhanced files|
|`task_id`|Task the files belong to (Format: `TXLY`)|

```python
python main.py YOUR_INPUT_DIR YOUR_OUTPUT_DIR YOUR_TASK_ID
```

### Training

To train your own models run the `train.py` file  with the following (optional) *named* command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
|`levels` | - | `Task_1`, `Task_2` or `All` |
|`epochs` | 4,000 | Training epochs |
|`batch_size`|50 | Batch size per network pass |
|`lr` |0.0001 | Learning Rate |
|`recon_mag`|100|Magnitude of reconstruction loss term|
|`diffusion`|False|Add diffusion noise to discriminator input|

```python
python train.py --levels All --epochs 4000 --batch_size 50 --lr 0.0001 --recon_mag 100 --diffusion
```

## Results

Below are some examples of enhanced audio files from the [Helsinki Speech Challenge 2024 dataset](https://zenodo.org/records/11380835). Regardless of the task, the entire enhancement architecture achieved an **Real-Time Factor (RTF) of ~0.04** on an Apple M3 chip.

### Task 1 Level 1

#### Recorded

https://github.com/user-attachments/assets/45e0be90-cbae-45bd-8cf5-ccac3efafb52

#### Enhanced

https://github.com/user-attachments/assets/6c171566-3fac-4e8e-a718-c2e578f97a18

### Task 1 Level 3

#### Recorded

https://github.com/user-attachments/assets/508b3bc4-e9f7-4782-a755-30e0d0455fe8

#### Enhanced

https://github.com/user-attachments/assets/94017083-740d-4f18-86ba-0ce46ef4b2bd

### Task 1 Level 5

#### Recorded

https://github.com/user-attachments/assets/d63611d7-5f55-4cac-8ef3-21f5e08c219c

#### Enhanced

https://github.com/user-attachments/assets/5d0fd97d-2526-4af0-8f35-7eb8f7d88b6f

### Task 1 Level 7

#### Recorded

https://github.com/user-attachments/assets/5a0e02dd-4e6d-4ec8-87c6-25fce4d809ad

#### Enhanced

https://github.com/user-attachments/assets/59f151c7-0ae3-44dd-8417-b2301934d307
