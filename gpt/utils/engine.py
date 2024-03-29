import torch
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from pathlib import Path

SAVE_CHECKPOINT_INTERVAL = 10
TEST_PROMPT = "Once upon a time"


def train_step(
    model: nn.Module,
    vocab_size: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    device: torch.device,
) -> float:
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
    outputs = model(batch_inputs)

    # Calculate the loss
    loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    return loss.item()


def test_step(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> str:
    model.eval()

    prompt = TEST_PROMPT.lower()
    token_ids = []
    for char in prompt:
        token_ids.append(dataset.char_to_idx[char])

    input_ids = (
        torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    )  # Add batch dimension

    generated_text = prompt
    max_length = 100  # Maximum length of generated text

    with torch.inference_mode():
        for _ in range(max_length):
            outputs = model(input_ids)

            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[next_token_id]], dtype=torch.long).to(device),
                ],
                dim=-1,
            )

            next_token = dataset.idx_to_char[int(next_token_id)]
            generated_text += next_token

            if next_token == "<EOS>":
                break

    return generated_text


def train(
    epochs: int,
    batch_size: int,
    model: nn.Module,
    vocab_size: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    device: torch.device,
    save_path: Path | None = None,
):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_inputs, batch_targets in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            total_loss += train_step(
                model=model,
                vocab_size=vocab_size,
                criterion=criterion,
                optimizer=optimizer,
                batch_inputs=batch_inputs,
                batch_targets=batch_targets,
                device=device,
            )

            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Save the model checkpoint every 10 epochs
        if save_path and (epoch + 1) % SAVE_CHECKPOINT_INTERVAL == 0:
            file_path = save_path / f"gpt2_epoch_{epoch + 1}.pth"
            print(f"Saving model checkpoint to {file_path}")
            torch.save(model.state_dict(), file_path)

        generated_output = test_step(model=model, dataset=dataset, device=device)
        print(f"Epoch {epoch + 1}/{epochs}, Generated Output: {generated_output}")

    if save_path:
        time_stamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        file_path = save_path / f"gpt2_{time_stamp}.pth"
        print(f"Saving final model to {file_path}")
        torch.save(model.state_dict(), file_path)
