#!/usr/bin/env python3
# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from setuptools import find_packages
from setuptools import setup

setup(
    name="thumt",
    version="1.1.0",
    author="The THUMT Authors",
    author_email="thumt17@gmail.com",
    description="THUMT: An open-source toolkit for neural machine translation",
    url="http://thumt.thunlp.org",
    entry_points={
        "console_scripts": [
            "thumt-trainer = thumt.bin.trainer:cli_main",
            "thumt-translator = thumt.bin.translator:cli_main"]},
    scripts=[
        "thumt/scripts/average_checkpoint.py",
        "thumt/scripts/build_vocab.py",
        "thumt/scripts/convert_checkpoint.py",
        "thumt/scripts/shuffle_corpus.py"],
    packages=find_packages(),
    install_requires=[
        "future",
        "pillow",
        "torch>=1.1.0",
        "tensorflow>=2.0.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"])