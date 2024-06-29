from tokenizers import Tokenizer
from smart_open import open
from tqdm import tqdm
import sys


SEQ_LEN = 128 - 2
tokenizer = Tokenizer.from_file("/cluster/home/sgerstner/contextualizer_code/tokenizers/tokenizer_small.json")


documents = [[]]



for line in tqdm(open(sys.argv[1])):

    line = line.strip()

    if len(line) == 0:
        if len(documents[-1]) > 0:
            documents.append([])
        continue

    ids = tokenizer.encode(line, add_special_tokens=False).ids
    documents[-1].append(ids)

dataset_name = sys.argv[1].split("/")[-1].split(".")[0].split("_")[0]




with open(f"../data/processed_100M_dev/cached_train_{SEQ_LEN + 2}_{dataset_name}.txt", "w") as f:

    for document in tqdm(documents):
        segment = []
        for i, sentence in enumerate(document):
            segment += sentence

            if len(segment) > SEQ_LEN:
                segment = segment[:SEQ_LEN]
                subwords = [tokenizer.id_to_token(token_id) for token_id in segment]
                f.write(" ".join(subwords) + "\n")

                segment = [s for s in sentence]

        if len(segment) > 0:
            segment = segment[:SEQ_LEN]
            subwords = [tokenizer.id_to_token(token_id) for token_id in segment]
            f.write(" ".join(subwords) + "\n")
