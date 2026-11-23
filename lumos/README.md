# LUMOS
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>LUMOS: Language-Conditioned Imitation Learning with World Models</b>](https://arxiv.org/pdf/.pdf)

## Installation

To begin, clone this repository locally and switch to the `torch2x` branch:

```bash
git clone https://github.com/acl21/lumos.git
cd lumos
git checkout diwali
export LUMOS_ROOT=$(pwd)

```
Install requirements:
```bash
cd LUMOS_ROOT
uv venv --python 3.10
source .venv/bin/activate
sh install.sh 
```