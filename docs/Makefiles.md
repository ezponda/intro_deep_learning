# Intro to Makefiles for Deep Learning Projects

## 🧾 What is a Makefile?

A **Makefile** is a simple way to **automate repetitive tasks** in a project.  
Instead of typing long commands each time, you define them once in a `Makefile` and run them with `make <task-name>`.

## ⚙️ Why Use Makefiles in Deep Learning?

In DL projects, you often repeat steps like:
- Creating environments
- Installing dependenciesgst
- 
- Running training scripts
- Exporting requirements

Makefiles help you organize and run these tasks with one command.

---

## 🛠️ Basic Structure

A Makefile has **rules** that look like this:

```make
setup:
    python3 -m venv .venv
    source .venv/bin/activate && pip install -r initial-requirements.txt
```
Go and check the Makefile at [Makefile](../class/Makefiles/Makefile) for more details.

```bash
make setup
```


