"""Generate synthetic leaf illustrations and run the full pipeline without a dataset download."""
import argparse
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def draw_leaf(seed, spotted=False, size=128):
    rng = random.Random(seed)
    image = Image.new('RGB', (size, size), (236, 232, 211))
    draw = ImageDraw.Draw(image)
    shift = rng.randint(-8, 8)
    green = (rng.randint(35, 65), rng.randint(115, 165), rng.randint(35, 70))
    draw.ellipse((25 + shift, 12, 100 + shift, 110), fill=green, outline=(35, 90, 35), width=2)
    draw.line((63 + shift, 28, 63 + shift, 121), fill=(124, 176, 79), width=3)
    for y in range(38, 98, 14):
        draw.line((63 + shift, y + 7, 35 + shift, y - 10), fill=(95, 155, 65), width=1)
        draw.line((63 + shift, y + 7, 90 + shift, y - 10), fill=(95, 155, 65), width=1)
    if spotted:
        for _ in range(rng.randint(8, 15)):
            x, y = rng.randint(43, 83) + shift, rng.randint(30, 93)
            radius = rng.randint(3, 6)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(142, 105, 42))
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(74, 55, 25))
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description='Demo with synthetic leaf illustrations, not real disease data.')
    parser.add_argument('--output', type=Path, default=Path('runs/demo'))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Demo directory exists. Choose another --output to preserve earlier results.')
    args.output.mkdir(parents=True)
    data = args.output / 'images'
    for label, spotted in (('synthetic_plain', False), ('synthetic_spotted', True)):
        folder = data / label
        folder.mkdir(parents=True)
        for seed in range(20):
            draw_leaf(seed, spotted).save(folder / f'{seed:02d}.png')
    root = Path(__file__).resolve().parent
    print('SYNTHETIC DEMO ONLY: these illustrations do not validate real leaf-disease accuracy.', flush=True)
    command = [sys.executable, '-B', str(root / 'train.py'), '--data', str(data),
               '--output', str(args.output / 'results'), '--epochs', '2', '--search-epochs', '1',
               '--wolves', '3', '--iterations', '2', '--repeats', '1', '--image-size', '64']
    subprocess.run(command, check=True)
    sample = data / 'synthetic_spotted/00.png'
    subprocess.run([sys.executable, '-B', str(root / 'predict.py'),
                    '--checkpoint', str(args.output / 'results/gwo_42.pt'),
                    '--image', str(sample), '--gradcam', str(args.output / 'prediction_gradcam.png')], check=True)
    print(f'Input image: {sample}\nGrad-CAM: {args.output / "prediction_gradcam.png"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
