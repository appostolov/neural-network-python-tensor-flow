import json
import config

class Vocab:
    def __init__(self):
        with open(config.VOCAB_JSON_PATH, "r", encoding="utf-8") as file:
            arr = json.load(file)
            self.map = {char: idx for idx, char in enumerate(arr)}

    def text_to_indexes(self, text: str, max: int)->list[int]:
        text = text[:max]
        unk = self.map.get('<UNK>', 0)
        indexes = [self.map.get(char, unk) for char in text]
        if len(indexes) < max:
            pad = self.map.get('<PAD>', 1)
            indexes += [pad] * (max - len(indexes))
        return indexes

    def len(self)->int:
        return len(self.map)
