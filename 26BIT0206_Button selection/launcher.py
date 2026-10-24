"""
One-click launcher for the dwell project (Tanmay Gupta, 26BIT0206).
Put this file in the same folder as dwell_gui.py, then run:  python launcher.py
(or just press the Run button in your editor). It uses the .venv Python if one exists.
"""
import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

HERE = os.path.dirname(os.path.abspath(__file__))


def find_python():
    """Prefer the project's virtual environment, else the Python running this file."""
    for rel in (os.path.join(".venv", "Scripts", "python.exe"),
                os.path.join(".venv", "bin", "python")):
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            return p
    return sys.executable


PY = find_python()


def run(args, wait=False):
    """Start a script in this folder. GUI scripts open without blocking the launcher."""
    try:
        if wait:
            result = subprocess.run([PY] + args, cwd=HERE, capture_output=True, text=True)
            show_output(" ".join(args), result.stdout + result.stderr)
        else:
            subprocess.Popen([PY] + args, cwd=HERE)
    except Exception as e:
        messagebox.showerror("Could not run", str(e))


def show_output(title, text):
    win = tk.Toplevel(root)
    win.title(title)
    box = tk.Text(win, width=90, height=28, bg="#11111b", fg="#cdd6f4", font=("Consolas", 10))
    box.pack(fill="both", expand=True)
    box.insert("1.0", text or "(no output)")


root = None


def main():
    global root
    root = tk.Tk()
    root.title("Dwell Project Launcher - 26BIT0206")
    root.configure(bg="#1e1e2e")
    root.geometry("420x420")

    tk.Label(root, text="Dwell-Based Gaze Interaction", bg="#1e1e2e", fg="#cdd6f4",
             font=("Segoe UI", 15, "bold")).pack(pady=(16, 2))
    tk.Label(root, text=f"Python: {os.path.relpath(PY, HERE) if PY.startswith(HERE) else PY}",
             bg="#1e1e2e", fg="#9399b2", font=("Segoe UI", 9)).pack()

    row = tk.Frame(root, bg="#1e1e2e")
    row.pack(pady=12)
    tk.Label(row, text="User ID:", bg="#1e1e2e", fg="#cdd6f4").pack(side="left")
    user = tk.Entry(row, width=10)
    user.insert(0, "U001")
    user.pack(side="left", padx=6)

    def btn(text, cmd):
        tk.Button(root, text=text, command=cmd, width=34, height=2, bg="#89b4fa",
                  fg="#11111b", font=("Segoe UI", 11, "bold"), relief="flat").pack(pady=5)

    btn("1. Run unit tests", lambda: run(["test_dwell.py"], wait=True))
    btn("2. Open GUI (mouse = gaze)", lambda: run(["dwell_gui.py"]))
    btn("3. Run experiment (3 s / 4 s / 5 s)",
        lambda: run(["dwell_gui.py", "--experiment", "--user", user.get().strip() or "U001"]))
    btn("4. Make results + charts", lambda: run(["analyze_results.py"], wait=True))

    root.mainloop()


if __name__ == "__main__":
    main()
