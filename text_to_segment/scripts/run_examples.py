import sieve

def run_examples():

    functions = [
        "sam2-focus",
        "sam2-callout",
        "sam2-color-filter",
        "sam2-blur",
        "sam2-selective-color",
        "sam2-pixelate",
    ]

    video = sieve.File(path="duckling.mp4")
    subject = "duckling"

    jobs = []
    for function in functions:
        fn = sieve.function.get("sieve-internal/" + function)
        jobs.append(fn.push(video=video, subject=subject))

        print(f"https://www.sievedata.com/jobs/{jobs[-1].job['id']}")


    exit()

if __name__ == "__main__":
    run_examples()
