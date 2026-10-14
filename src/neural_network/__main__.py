import sys

from .network import Network
from .vocab.vocab import Vocab

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "none"

    nn = Network()
    vocab = Vocab()

    if sys.argv[1] == "train":
        nn.create(vocab_size = vocab.len())
        inputs, labels = nn.dataset()
        nn.train(emails = inputs, labels = labels)
        nn.save()
    elif sys.argv[1] == "run":
        email = sys.argv[2]
        nn.open()
        prediction = nn.run( email = email)
        print(f"{email} is {prediction * 100}% valid")
    else:
        print("Command train or run missing!")