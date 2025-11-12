<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;

$options = getopt('', [
    'dataset:',
    'model:',
    'train-ratio::',
    'ewc::',
]);
$datasetPath = $options['dataset'] ?? __DIR__ . '/data/dataset.json';
$modelPath = $options['model'] ?? __DIR__ . '/storage/model.json';
$trainRatio = isset($options['train-ratio']) ? (float) $options['train-ratio'] : 0.75;
$ewc = isset($options['ewc']) ? (float) $options['ewc'] : 0.2;

if (!is_file($datasetPath)) {
    fwrite(STDERR, "Dataset not found: {$datasetPath}\n");
    exit(1);
}

$raw = file_get_contents($datasetPath);
if ($raw === false) {
    fwrite(STDERR, "Unable to read dataset.\n");
    exit(1);
}

$data = json_decode($raw, true, flags: JSON_THROW_ON_ERROR);
$samples = $data['samples'] ?? [];
if ($samples === []) {
    fwrite(STDERR, "Dataset contains no samples.\n");
    exit(1);
}

$model = new NSCTXModel(new Storage($modelPath));
$metrics = $model->train($samples, $trainRatio, $ewc);

fwrite(STDOUT, "Training complete!\n");
fwrite(STDOUT, sprintf("Samples: %d\n", $metrics['sample_count']));
fwrite(STDOUT, sprintf("Train accuracy: %.2f%%\n", $metrics['train_accuracy'] * 100));
fwrite(STDOUT, sprintf("Test accuracy: %.2f%%\n", $metrics['test_accuracy'] * 100));
fwrite(STDOUT, sprintf("Timestamp: %s\n", $metrics['trained_at']));

$alpha = $model->getAlpha();
if ($alpha !== []) {
    fwrite(STDOUT, "Modal weights:\n");
    foreach ($alpha as $name => $weight) {
        fwrite(STDOUT, sprintf("  - %s: %.3f\n", $name, $weight));
    }
}
