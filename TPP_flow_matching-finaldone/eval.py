import torch

from add_thin.metrics import MMD, lengths_distribution_wasserstein_distance
from add_thin.evaluate_utils import get_task, get_run_data

# Set run id and paths
RUN_ID = "ifd5gi6i"

WANDB_DIR = "outputs/wandb/wandb"
PROJECT_ROOT = "./"  # should include data folder

def sample_model(task, tmax, n=1000):
    """
    Unconditionally draw n event sequences from Add Thin.
    """
    with torch.no_grad():
        samples = task.model.sample(n, tmax=tmax.to(task.device)).to_time_list()

    assert len(samples) == n, "not enough samples"
    return samples

# Get run data
data_name, seed, run_path = get_run_data(RUN_ID, WANDB_DIR)

# Get task and datamodule
task, datamodule = get_task(run_path, density=True, data_root=PROJECT_ROOT)

# Get test sequences
test_sequences = []
for (
    batch
) in (
    datamodule.test_dataloader()
):  # batch is set to full test set, but better be safe
    test_sequences = test_sequences + batch.to_time_list()

# Sample event sequences from trained model
samples = sample_model(task, datamodule.tmax, n=4000)

# Evaluate metrics against test dataset
mmd = MMD(
    samples,
    test_sequences,
    datamodule.tmax.detach().cpu().item(),
)[0]
wasserstein = lengths_distribution_wasserstein_distance(
    samples,
    test_sequences,
    datamodule.tmax.detach().cpu().item(),
    datamodule.n_max,
)

# Print rounded results for data and seed
print("ADD and Thin density evaluation:")
print("================================")
print(
    f"{data_name} (Seed: {seed}): MMD: {mmd:.3f}, Wasserstein: {wasserstein:.3f}"
)