import torch

from src.helpers.bath_split import get_batches
from src.helpers.read_text import read_text
from src.model.encoders_decoder import get_encoders_decoders
from src.model.hyper_paramters import SPLIT_RATIO
from src.model.language_models import BigramLanguageMode

file_path = "training_data/shakespear/text_1.txt"
text = read_text(file_path=file_path)
encode, decode, vocabular_size = get_encoders_decoders(text=text)
"""split the training data and test data"""
data = torch.tensor(encode(text), dtype=torch.long)
split_data = {
    "train": data[: int(SPLIT_RATIO * len(data))],
    "validation": data[int(SPLIT_RATIO * len(data)) :],
}
torch.manual_seed(1337)
model = BigramLanguageMode(vocabulary_size=vocabular_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for _ in range(1000):
    training_input, training_target = get_batches(split=split_data, spit_type="train")
    logits, loss = model(training_input, training_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("loss in prediction", loss.item())
print("Prediction with auto initiation")
print(
    decode(
        model.generate(
            index=torch.zeros((1, 1), dtype=torch.long), maximum_new_tokens=100
        )[0].tolist(),
    )
)
