# Python Environment Setup

## Prerequisites

Install the `venv` package (one-time system setup):

```bash
sudo apt install -y python3.12-venv
```

## Setup & Install

**1. Create the virtual environment:**

```bash
python3 -m venv .venv
```

**2. Activate it:**

```bash
source .venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** Run all commands from the `09A-backend/` directory.

## Daily Usage

Every time you open a new terminal, re-activate the venv before running scripts:

```bash
source .venv/bin/activate
```
