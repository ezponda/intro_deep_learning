# Intro to Makefiles for Deep Learning Projects

## 🧾 What is a Makefile?

A **Makefile** is a simple way to **automate repetitive tasks** in a project.  
Instead of typing long commands each time, you define them once in a `Makefile` and run them with `make <task-name>`.

## ⚙️ Why Use Makefiles in Deep Learning?

In DL projects, you often repeat steps like:
- Creating environments
- Installing dependencies
- 
- Running training scripts
- Exporting requirements

Makefiles help you organize and run these tasks with one command.

---

## 📁 Walkthrough of the `Makefile`

Let's go through the actual `Makefile` defined for this project. You can find it at `class/Makefiles/Makefile`.

### 🔹 `PROJ_DIR` and `NEW_VENV`
```make
PROJ_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
NEW_VENV := $(PROJ_DIR)/.venv
```
These lines define the path to the project directory and the location of the virtual environment (`.venv`).

---

### 🧹 `clean`

```make
clean:
	@echo "Cleaning up..."
	rm -rf $(NEW_VENV)
	rm -f final-requirements.txt
```
This removes the virtual environment and any frozen dependencies. Use it to reset your setup.

---

### 🧪 `create`

```make
create:
	@echo "Creating virtual environment..."
	@if [ ! -d "$(NEW_VENV)" ]; then \
  		echo "🔧 Creating virtual environment..."; python3 -m venv $(NEW_VENV); else \
		echo "✅ Virtual environment already exists."; fi
```
This checks if the `.venv` directory exists. If not, it creates a new virtual environment.

---

### ⚙️ `setup`

```make
setup: clean create
	@echo "Installing initial requirements..."
	@$(NEW_VENV)/bin/pip install -r initial-requirements.txt
```
This is the main setup command. It runs `clean`, creates the environment, and installs dependencies listed in `initial-requirements.txt`.

---

### ➕ `add-deps`

```make
add-deps: create
	@$(NEW_VENV)/bin/pip install matplotlib
```
Installs a new dependency (in this case, `matplotlib`) into the environment.

---

### 📦 `freeze-env`

```make
freeze-env: create
	@$(NEW_VENV)/bin/pip freeze > final-requirements.txt
```
Saves the current environment's installed packages into `final-requirements.txt`.

---

### ♻️ `update-env`

```make
update-env: create
	@$(NEW_VENV)/bin/pip install -r final-requirements.txt
```
Restores the environment using `final-requirements.txt`, useful for syncing dependencies across machines.

---

Each of these targets can be run with:

```bash
make <target-name>
```

For example:
```bash
make setup
```

---

---

## 🧪 Activate the Virtual Environment

After running `make setup`, activate the virtual environment:

```bash
source .venv/bin/activate
```

You can check that Python is now using the virtual environment by running:

```bash
which python
```

You should see a path like:

```
/your/project/path/.venv/bin/python
```

---

## 🔍 Check that `matplotlib` is Not Installed

Let's confirm that `matplotlib` is not installed yet:

```bash
python
```

Inside the Python shell, type:

```python
import matplotlib
```

You should see an error like:

```
ModuleNotFoundError: No module named 'matplotlib'
```

Exit Python by pressing `Ctrl+D` or typing:

```python
exit()
```

---

## ➕ Install `matplotlib` Using the Makefile

Now, install `matplotlib` with:

```bash
make add-deps
```

This will install `matplotlib` into your `.venv`.

To verify:

```bash
python
```

Then inside Python:

```python
import matplotlib
print(matplotlib.__version__)
```

You should see the installed version printed without any error.

---

## 🛠 Troubleshooting: Forgot to Activate?

If you see a `ModuleNotFoundError` even after installing, make sure you have activated your virtual environment:

```bash
source .venv/bin/activate
```
