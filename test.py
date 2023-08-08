from rich import print

if __name__ == "__main__":
    import os
    import subprocess

    dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    for d in dirs:
        if d == ".git":
            continue
        os.chdir(d)
        print("Deploying directory: " + d)

        proc = subprocess.run(
            ["sieve", "deploy", "--yes"], capture_output=True, text=True
        )

        if proc.returncode != 0:
            print("[red bold]Error deploying directory: [/]" + d)
            print(proc.stderr)
            break

        os.chdir("..")
        print("[green bold]Deployed directory: [/]" + d)
