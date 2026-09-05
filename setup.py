#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="gcbfplus",
    version="0.2.1",
    description='Jax implementation of GCBF+ (S Zhang, O So, K Garg, C Fan: "GCBF+: A Neural Graph '
                'Control Barrier Function Framework for Distributed Safe Multi-Agent Control", CoRL), '
                'extended with diffusion- and MILP-based STL planning on top of the GCBF+ controller.',
    author="Songyuan Zhang",
    author_email="szhang21@mit.edu",
    url="https://github.com/MIT-REALM/gcbfplus",
    install_requires=[],
    packages=find_packages(),
)
