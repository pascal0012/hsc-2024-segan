import wandb

from src.util.consts import TASK_1, TASK_2

sweep_configuration = {
    "program": "sweep_agent.py",
    "method": "random",
    "metric": {"goal": "minimize", "name": "cer_task_1_level_4"},
    "parameters": {
        "levels": {"values": [TASK_1, TASK_1 + TASK_2]},
        "batch_size": {"values": [16, 32, 64, 128]},
        "lr": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
        "l1_mag": {"values": [60, 80, 100, 120]},
        "diffusion_C": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "diffusion_sigma": {"min": 0.01, "max": 0.4},
        "diffusion_d_target": {"min": 0.1, "max": 0.9},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 1_000},
}


def main():
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="hsc-2024-sweep")

    print(f"Starting sweep with ID: {sweep_id}")


if __name__ == "__main__":
    main()
