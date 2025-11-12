<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;

$options = getopt('', [
    'model:',
    'text::',
    'image::',
    'audio::',
]);

$modelPath = $options['model'] ?? __DIR__ . '/storage/model.json';
$text = $options['text'] ?? '';
$imageRaw = $options['image'] ?? '';
$audioRaw = $options['audio'] ?? '';

if (!is_file($modelPath)) {
    fwrite(STDERR, "Model not found. Train the model first.\n");
    exit(1);
}

$model = new NSCTXModel(new Storage($modelPath));
$modalities = [
    'text' => (string) $text,
    'image' => $imageRaw === '' ? [] : array_map('floatval', explode(',', (string) $imageRaw)),
    'audio' => $audioRaw === '' ? [] : array_map('floatval', explode(',', (string) $audioRaw)),
];

$result = $model->predict($modalities);

fwrite(STDOUT, sprintf("Prediction: %s\n", $result['prediction']));
fwrite(STDOUT, "Probabilities:\n");
foreach ($result['probabilities'] as $label => $probability) {
    fwrite(STDOUT, sprintf("  - %s: %.2f%%\n", $label, $probability * 100));
}

fwrite(STDOUT, "Modal weights:\n");
foreach ($result['modal_weights'] as $name => $weight) {
    fwrite(STDOUT, sprintf("  - %s: %.3f\n", $name, $weight));
}

$intermediates = $result['intermediates'];
fwrite(STDOUT, "Reasoning steps:\n");
foreach ($intermediates['reasoning_steps'] as $index => $step) {
    $preview = array_slice($step, 0, 4);
    fwrite(STDOUT, sprintf("  Step %d: [%s]\n", $index + 1, implode(', ', array_map(fn ($v) => sprintf('%.3f', $v), $preview))));
}
