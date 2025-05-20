import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to train generative models."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )
    return parser.parse_args()
