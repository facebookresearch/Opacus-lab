#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved

import csv

import torch
import tqdm
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

"""
Script assumes in same directory as wikitext download (i.e. same directory
that ran prepare-wikitext-103.sh)
Change paths below as needed
"""
file_train = "wikitext-103/wiki.train.tokens"
file_test = "wikitext-103/wiki.test.tokens"
file_valid = "wikitext-103/wiki.valid.tokens"

for file, name in [(file_train, "train"), (file_test, "test"), (file_valid, "valid")]:
    corpus = []
    print(f"Processing wikitext {name} set...")
    with open(file, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter="\n")
        for i, row in tqdm.tqdm(enumerate(csv_reader)):
            if row != [" "]:
                next_row = torch.tensor(tokenizer.encode(row[0]))
                corpus.append(next_row)

    corpus = torch.cat(corpus).long()
    print(f"Saving wikitext {name} corpus as Pytoch tensor...")
    torch.save(corpus, f"wikitext-103-{name}-corpus.pt")
print("Done!")
