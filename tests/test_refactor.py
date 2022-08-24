#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved
import unittest

from opacus_lab.models.GPT2.refactor import refactor_transformer, test_refactor
from transformers import GPT2Config, GPT2LMHeadModel


class LoadModelTest(unittest.TestCase):
    def test_load_model(self):
        size = "S"
        configuration = GPT2Config()
        pretrained_model = GPT2LMHeadModel(configuration)
        model = refactor_transformer(
            pretrained_model,
            size=size,
        )
        self.assertTrue(
            test_refactor(pretrained_model, model, exact=False), "Refactor failed..."
        )
