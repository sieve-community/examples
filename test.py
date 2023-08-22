from rich import print
from pathlib import Path
import os
import subprocess
import argparse
import time
import sieve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--github", action="store_true")
    args = parser.parse_args()
    ignore_dirs = [
        ".git",
        ".github",
        "talking_head_avatars",
        "text_to_speech",
        "video_lipsyncing",
        "audio_transcription",
        "auto_chapter_title",
        "segment_anything",
    ]

    dirs = [d for d in os.listdir(".") if os.path.isdir(d) and d not in ignore_dirs]
    dirs.sort()
    dir_status = {}
    if args.deploy:
        for d in dirs:
            os.chdir(d)
            print("Deploying directory: " + d)

            proc = subprocess.run(
                ["sieve", "deploy", "--yes"], capture_output=True, text=True
            )

            if proc.returncode != 0:
                print("[red bold]Error deploying directory: [/]" + d)
                print(proc.stderr)
                raise Exception("Error deploying directory: " + d)

            os.chdir("..")
            print("[green bold]Deployed directory: [/]" + d)
            dir_status[d] = {"status": "deployed"}

        print("[green bold]Deployed all examples[/]")

    if args.test:
        job_ids = set()
        for d in dirs:
            if Path(d + "/main.py").exists():
                os.chdir(d)
                print("Running example: " + d)
                proc = subprocess.run(
                    ["python3", "main.py"], capture_output=True, text=True
                )

                if proc.returncode != 0:
                    print("[red bold]Error running example: [/]" + d)
                    print(proc.stderr)
                    raise Exception("Error running example: " + d)

                # Parse the job IDs from the output
                found_job = False
                for line in proc.stdout.split("\n"):
                    if "id=" in line:
                        found_job = True
                        job_id = line.split("=")[1].split(" ")[0].strip()
                        curr_time = time.time()
                        job_ids.add((d, curr_time, job_id))

                if not found_job:
                    print("[red bold]No jobs submitted from: [/]" + d)
                    print(proc.stdout)

                os.chdir("..")
            else:
                print("[red bold]No tests found in directory: [/]" + d)

        print("[green bold]Submitted jobs for all examples: [/]")
        print("Waiting for outputs...")

        start_time = time.time()
        timeout_min = 10
        failed = False

        while len(job_ids) > 0:
            test_name, time_started, job_id = job_ids.pop()
            job = sieve.get(job_id)
            status = job["status"]
            if time.time() - start_time > timeout_min * 60:
                dir_status[d] = {
                    "status": "failed",
                    "error": "10 minute timeout reached",
                }
                print("[red bold]Timeout reached[/]")
                break
            if status == "finished":
                print(f"[green bold]Job finished:[/] {test_name} {job_id}")
                dir_status[d] = {"status": "tested", "time": time.time() - time_started}
            elif status == "error":
                print(f"[red bold]Job failed:[/] {test_name} {job_id}")
                print(job["error"])
                dir_status[d] = {"status": "failed", "error": job["error"]}
                failed = True
                # raise Exception(f"Job failed: {test_name} {job_id}")
            else:
                job_ids.add((test_name, time_started, job_id))
                time.sleep(10)

        print(dir_status)
        if len(job_ids) == 0 and not failed:
            print("[green bold]All jobs finished[/]")
        else:
            print("[red bold]Some jobs failed to finish[/]")
            print(job_ids)
            raise Exception(f"Some jobs timed out: {str(job_ids)}")

    if args.github:
        env_file = os.getenv("GITHUB_ENV")
        github_output = ""
        for dir_name, status_info in dir_status.items():
            if status_info["status"] == "failed":
                github_output += f"* {dir_name} :x:\n"
            else:
                github_output += f"* {dir_name} :white_check_mark: {round(status_info['time'], 2)}s\n"
        with open(env_file, "a") as f:
            f.write(f"job_info={github_output}")
