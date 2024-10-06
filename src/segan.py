from typing import Dict, List, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from src.data.dataloader import get_loader
from src.data.dataset import SpeechData
from src.data.files import write_model
from src.data.filesampler import sample_filepaths
from src.eval.testing import Tester
from src.networks.discriminator import Discriminator
from src.networks.generator import Generator
from src.util.consts import LOG_INTERVAL
from src.util.diffuser import Diffuser
from src.util.logger import Logger
from src.util.losses import segan_discriminator_loss, segan_generator_loss
from src.util.signals import load_chunks_pair_list


class SEGAN:
    def __init__(
        self,
        levels: List[str],
        hyperparameters: Dict[str, Union[int, float]],
        device: torch.device,
    ) -> None:
        self.levels = levels
        self.hyperparameters = hyperparameters
        self.device = device

        # Initialize dataset
        test_paths, val_paths = sample_filepaths(
            tasks=levels, sample_rate=0.01, split_into=2
        )
        test_chunks, val_chunks = (
            load_chunks_pair_list(test_paths),
            load_chunks_pair_list(val_paths),
        )

        self.validation = {"paths": val_paths, "chunks": val_chunks}
        self.testing = {"paths": test_paths, "chunks": test_chunks}

        dataset = SpeechData(tasks=levels, ignore_paths=test_paths + val_paths)

        self.train_loader = get_loader(
            dataset, batch_size=hyperparameters["batch_size"], device=device
        )

        # Initialize networks
        self.generator = Generator(device=device)
        reference_batch = dataset.get_reference_batch(
            batch_size=hyperparameters["batch_size"], device=device
        )

        # Add diffuser if diffusion GAN is enabled
        diffuser = (
            Diffuser(
                device=device,
                sigma=hyperparameters["diffuser_sigma"]
                if "diffuser_sigma" in hyperparameters
                else None,
                C=hyperparameters["diffuser_C"]
                if "diffuser_C" in hyperparameters
                else None,
                d_target=hyperparameters["diffuser_d_target"]
                if "diffuser_d_target" in hyperparameters
                else None,
            )
            if hyperparameters["diffusion"]
            else None
        )

        self.discriminator = Discriminator(
            reference_batch=reference_batch,
            device=device,
            diffuser=diffuser,
        )

        # Initialize optimizers
        self.generator_optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=hyperparameters["lr"]
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=hyperparameters["lr"]
        )

        self.reconstruction_loss = torch.nn.MSELoss()

        # Initialize schedulers
        self.gen_scheduler = StepLR(self.generator_optimizer, step_size=1500, gamma=0.1)
        self.disc_scheduler = StepLR(
            self.discriminator_optimizer, step_size=1500, gamma=0.1
        )

    def learn(self, num_episodes: int = 2000, sweep: bool = False):
        # Initialize logger
        self.logger = Logger(
            tasks=self.levels,
            hyperparameters=self.hyperparameters,
            tags=["segan"],
            sweep=sweep,
        )
        self.tester = Tester(run_name=self.logger.run_name)

        for episode in range(num_episodes):
            for i, (clean_files, recorded_files, _) in enumerate(self.train_loader):
                z = nn.init.normal_(
                    torch.Tensor(clean_files.shape[0], 1024, 8).to(device=self.device)
                )

                clean_files = clean_files.to(self.device, non_blocking=True)
                recorded_files = recorded_files.to(self.device, non_blocking=True)

                # Update discriminator
                self.discriminator.zero_grad(set_to_none=True)
                g_out = self.generator(recorded_files, z)
                d_out_fake = self.discriminator(g_out, recorded_files)
                d_out_real = self.discriminator(clean_files, recorded_files)
                discriminator_loss = segan_discriminator_loss(
                    d_out_real=d_out_real, d_out_fake=d_out_fake
                )
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                # Update generator
                self.generator.zero_grad(set_to_none=True)
                g_out = self.generator(recorded_files, z)
                d_out_fake = self.discriminator(g_out, recorded_files)
                generator_loss = segan_generator_loss(
                    g_out=g_out,
                    d_out=d_out_fake,
                    x_clean=clean_files,
                    l1_mag=self.hyperparameters["l1_mag"],
                )

                generator_loss.backward()
                self.generator_optimizer.step()

                # Update diffusion step list
                if i % 4 == 0 and self.hyperparameters["diffusion"]:
                    self.discriminator.update_diffuser()

                if i % LOG_INTERVAL == 0:
                    self.logger.log_metrics(
                        generator_loss=generator_loss.item(),
                        discriminator_loss=discriminator_loss.item(),
                        episode=episode,
                        iteration=i,
                        lr=self.gen_scheduler.get_last_lr()[0],
                    )

            # Update learning rates
            # self.gen_scheduler.step()
            # self.disc_scheduler.step()

            if episode % 80 == 0:
                (
                    chunk_recon_loss,
                    mean_cer,
                    cers,
                    sample_paths,
                    transcriptions,
                ) = self.tester.test(
                    generator=self.generator,
                    paths=self.validation["paths"],
                    chunks=self.validation["chunks"],
                    episode=episode,
                    device=self.device,
                    write_all=True,
                )

                self.logger.log_metrics(
                    chunk_recon_loss=chunk_recon_loss,
                    mean_cer=mean_cer,
                    cers=cers,
                    episode=episode,
                    iteration=i,
                    lr=self.gen_scheduler.get_last_lr()[0],
                    audio_paths=sample_paths,
                    transcriptions=transcriptions,
                )

        self.logger.finish()

    def test(self):
        test_chunks = load_chunks_pair_list(sampled_paths=self.test_paths)
        return self.tester.test(
            generator=self.generator,
            paths=self.test_paths,
            chunks=test_chunks,
            episode=self.hyperparameters["num_episodes"],
            device=self.device,
            write_all=True,
        )

    def write(self):
        write_model(
            model=self.generator, run_name=self.logger.run_name, model_name="generator"
        )
        write_model(
            model=self.discriminator,
            run_name=self.logger.run_name,
            model_name="discriminator",
        )
