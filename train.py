"""CNN baseline vs Grey Wolf Optimization; test set used only after selection."""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import LeafDataset, split_dataset
from gradcam import save_overlay
from gwo import Hyperparameters, optimize
from training import fit, metrics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def plot_matrix(matrix, classes, output):
    figure, axis = plt.subplots(figsize=(max(6, len(classes) * 0.3), max(5, len(classes) * 0.3)))
    picture = axis.imshow(matrix, cmap='Blues')
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)),
             xticklabels=classes, yticklabels=classes, xlabel='Predicted', ylabel='True')
    plt.setp(axis.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    figure.colorbar(picture, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plant leaf CNN classification with Grey Wolf Optimization.')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/experiment'))
    parser.add_argument('--architecture', choices=('cnn', 'resnet18'), default='cnn')
    parser.add_argument('--no-pretrained', action='store_true', help='ResNet18 without downloading ImageNet weights.')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--search-epochs', type=int, default=3)
    parser.add_argument('--wolves', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--complexity-weight', type=float, default=0.02)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.epochs, args.search_epochs, args.iterations, args.repeats, args.threads) < 1 or args.wolves < 3 or args.image_size < 32:
        parser.error('Positive epochs/iterations/repeats/threads, >=3 wolves and image-size >=32 are required.')
    if not math.isfinite(args.complexity_weight) or args.complexity_weight < 0:
        parser.error('complexity-weight must be finite and nonnegative.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu.')
    if args.output.exists():
        parser.error('Output directory already exists. Choose a new --output to preserve earlier runs.')
    torch.set_num_threads(args.threads)
    try:
        manifest = split_dataset(args.data, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True)
    write_json(args.output / 'split.json', manifest)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings['torch_version'] = str(torch.__version__)
    write_json(args.output / 'settings.json', settings)
    test = LeafDataset(manifest, 'test', args.image_size)
    results = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        trials = []
        def train(config, epochs):
            return fit(manifest, config, epochs=epochs, seed=seed, size=args.image_size,
                       device=args.device, architecture=args.architecture, pretrained=not args.no_pretrained)
        def objective(config):
            print(f'Search repeat {repeat + 1}: {config}', flush=True)
            model, report = train(config, args.search_epochs)
            penalty = args.complexity_weight * report['parameters']['trainable'] / report['parameters']['total']
            fitness = 1 - report['best_val_macro_f1'] + penalty
            trials.append({'config': asdict(config), 'fitness': fitness, **report})
            # Persist expensive search progress after every evaluation.
            write_json(args.output / f'trials_{seed}.json', trials)
            del model
            return fitness
        search_start = time.perf_counter()
        max_frozen = 4 if args.architecture == 'resnet18' and not args.no_pretrained else 0
        best, search = optimize(objective, wolves=args.wolves, iterations=args.iterations,
                                seed=seed, max_frozen=max_frozen)
        search_seconds = time.perf_counter() - search_start
        write_json(args.output / f'gwo_{seed}.json', search)
        plt.figure()
        plt.plot([row['iteration'] for row in search], [row['best_fitness'] for row in search], marker='o')
        plt.xlabel('GWO iteration'); plt.ylabel('Best validation fitness (lower is better)')
        plt.tight_layout(); plt.savefig(args.output / f'gwo_{seed}.png'); plt.close()
        for method, config in [('baseline', Hyperparameters()), ('gwo', best)]:
            print(f'Final training: {method}, repeat {repeat + 1}', flush=True)
            model, report = train(config, args.epochs)
            started = time.perf_counter()
            test_metrics = metrics(model, DataLoader(test, batch_size=config.batch_size),
                                   args.device, len(manifest['classes']))
            row = {'method': method, 'seed': seed, 'config': asdict(config), **report,
                   'test': test_metrics, 'test_seconds': time.perf_counter() - started,
                   'search_seconds': search_seconds if method == 'gwo' else 0}
            results.append(row)
            write_json(args.output / 'results.json', results)
            prefix = f'{method}_{seed}'
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': asdict(config), 'architecture': args.architecture,
                        'classes': manifest['classes'], 'image_size': args.image_size},
                       args.output / f'{prefix}.pt')
            plot_matrix(test_metrics['confusion_matrix'], manifest['classes'], args.output / f'{prefix}_confusion.png')
            image, _ = test[0]
            with Image.open(Path(manifest['root']) / test.rows[0]['path']) as original:
                save_overlay(model, image.unsqueeze(0).to(args.device), original,
                             args.output / f'{prefix}_gradcam.png')
            figure, axes = plt.subplots(1, 2, figsize=(9, 3))
            axes[0].plot([h['train_loss'] for h in report['history']]); axes[0].set_title('Training loss')
            axes[1].plot([h['val_macro_f1'] for h in report['history']]); axes[1].set_title('Validation macro-F1')
            figure.tight_layout(); figure.savefig(args.output / f'{prefix}_learning.png'); plt.close(figure)
            del model
    summary = {}
    for method in ('baseline', 'gwo'):
        rows = [row for row in results if row['method'] == method]
        summary[method] = {'repeats': len(rows)}
        for metric in ('accuracy', 'macro_f1'):
            values = [row['test'][metric] for row in rows]
            summary[method][metric] = {'mean': float(np.mean(values)), 'std_population': float(np.std(values))}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
