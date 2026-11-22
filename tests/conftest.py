import pandas as pd
import pytest
from transformers import BertConfig, BertModel, BertTokenizer


@pytest.fixture
def encoder_dir(tmp_path):
    directory = tmp_path / "encoder"
    directory.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "day", "night", "sun", "rain"]
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizer(str(directory / "vocab.txt")).save_pretrained(directory)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=3,
                         num_attention_heads=2, intermediate_size=32)).save_pretrained(directory)
    return directory


@pytest.fixture
def pairs_csv(tmp_path):
    path = tmp_path / "pairs.csv"
    pd.DataFrame({"anchor": ["good", "bad", "day", "night", "sun", "rain", "good day"],
                  "positive": ["good day sun", "bad night rain", "day sun", "night rain", "sun day",
                               "rain night", "good sun day"]}).to_csv(path, index=False)
    return path


