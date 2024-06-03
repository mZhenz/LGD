#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys
import torch
from collections import OrderedDict

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.student."):
            continue
        old_k = k
        k = k.replace("module.student.", "")
        print(old_k, "->", k)
        newmodel[k] = v

    # res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    torch.save(OrderedDict(newmodel), sys.argv[2])