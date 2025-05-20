import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils

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

def show_images(tensor, title=None, nrow=5):
    """
    Plot a batch of images: tensor of shape (B, C, H, W)
    """
    # If tensor is on GPU, bring it to CPU
    tensor = tensor.detach().cpu()

    # Convert to grid of images
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True)

    # Convert to numpy and plot
    plt.figure(figsize=(nrow * 2, 2.5))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()
