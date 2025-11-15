<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;

$options = getopt('', [
    'model::',
    'base::',
    'adapt::',
    'carryover::',
    'train-ratio::',
    'ewc::',
]);

$modelPath = array_key_exists('model', $options) && $options['model'] !== false
    ? (string) $options['model']
    : __DIR__ . '/storage/model.json';
$basePath = array_key_exists('base', $options) && $options['base'] !== false
    ? (string) $options['base']
    : __DIR__ . '/data/dataset.json';
$adaptPath = array_key_exists('adapt', $options) && $options['adapt'] !== false
    ? (string) $options['adapt']
    : __DIR__ . '/data/adaptation.json';
$carryover = isset($options['carryover']) ? (float) $options['carryover'] : 0.35;
$trainRatio = isset($options['train-ratio']) ? (float) $options['train-ratio'] : 0.8;
$ewc = isset($options['ewc']) ? (float) $options['ewc'] : 0.2;

/**
 * @return array<int, array{label: string, modalities: array<string, mixed>}>
 */
function loadDatasetSamples(string $path): array
{
    if (!is_file($path)) {
        throw new InvalidArgumentException(sprintf('Dataset not found: %s', $path));
    }
    $raw = file_get_contents($path);
    if ($raw === false) {
        throw new RuntimeException(sprintf('Unable to read dataset: %s', $path));
    }
    $data = json_decode($raw, true, flags: JSON_THROW_ON_ERROR);
    return $data['samples'] ?? [];
}

try {
    $baseSamples = loadDatasetSamples($basePath);
    $adaptSamples = loadDatasetSamples($adaptPath);
} catch (Throwable $exception) {
    fwrite(STDERR, $exception->getMessage() . "\n");
    exit(1);
}

$model = new NSCTXModel(new Storage($modelPath));
$metrics = $model->transferLearn($baseSamples, $adaptSamples, $carryover, $trainRatio, $ewc);

fwrite(STDOUT, "Transfer learning complete!\n");
foreach ([
    'train_accuracy' => 'Train accuracy',
    'test_accuracy' => 'Test accuracy',
    'base_retention' => 'Base retention',
    'adapt_performance' => 'Adaptation performance',
] as $key => $label) {
    $value = $metrics[$key] ?? null;
    if ($value === null) {
        fwrite(STDOUT, sprintf("%s: n/a\n", $label));
    } else {
        fwrite(STDOUT, sprintf("%s: %.2f%%\n", $label, $value * 100));
    }
}

fwrite(STDOUT, sprintf("Carryover samples: %d\n", $metrics['carryover_samples'] ?? 0));
fwrite(STDOUT, sprintf("Samples trained: %d\n", $metrics['sample_count'] ?? 0));
fwrite(STDOUT, sprintf("Timestamp: %s\n", $metrics['trained_at'] ?? 'unknown'));
