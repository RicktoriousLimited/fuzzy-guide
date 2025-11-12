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
    'turn:',
]);

$modelPath = $options['model'] ?? __DIR__ . '/storage/model.json';
$text = $options['text'] ?? '';
$imageRaw = $options['image'] ?? '';
$audioRaw = $options['audio'] ?? '';
$turnOptions = $options['turn'] ?? [];
if (!is_array($turnOptions)) {
    $turnOptions = [$turnOptions];
}

/**
 * @return array{role: string, content: string}|null
 */
function parseTurnLine(string $line): ?array
{
    $line = trim($line);
    if ($line === '') {
        return null;
    }
    if (strpos($line, ':') === false) {
        return [
            'role' => 'narrator',
            'content' => $line,
        ];
    }
    [$role, $content] = array_map('trim', explode(':', $line, 2));
    if ($content === '') {
        return null;
    }
    return [
        'role' => $role === '' ? 'unknown' : strtolower($role),
        'content' => $content,
    ];
}

if (!is_file($modelPath)) {
    fwrite(STDERR, "Model not found. Train the model first.\n");
    exit(1);
}

$model = new NSCTXModel(new Storage($modelPath));
$turns = [];
foreach ($turnOptions as $turnLine) {
    $parsed = parseTurnLine((string) $turnLine);
    if ($parsed !== null) {
        $turns[] = $parsed;
    }
}

if ($turns === [] && $text !== '') {
    $lines = preg_split("/\r?\n/", (string) $text) ?: [];
    foreach ($lines as $line) {
        $parsed = parseTurnLine($line);
        if ($parsed !== null) {
            $turns[] = $parsed;
        }
    }
}

$textPayload = $turns === [] ? (string) $text : $turns;
$imageValues = $imageRaw === '' ? [] : array_map('floatval', explode(',', (string) $imageRaw));
$audioValues = $audioRaw === '' ? [] : array_map('floatval', explode(',', (string) $audioRaw));

if ($textPayload === '' && $turns === []) {
    fwrite(STDERR, "Provide either --turn options or --text content for prediction.\n");
    exit(1);
}

$modalities = [
    'text' => $textPayload,
    'image' => $imageValues,
    'audio' => $audioValues,
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
fwrite(STDOUT, "Conversation summary:\n");
if (($intermediates['conversation_turns'] ?? []) !== []) {
    foreach ($intermediates['conversation_turns'] as $turn) {
        fwrite(STDOUT, sprintf("  - %s: %s\n", $turn['role'], $turn['content']));
    }
} elseif (($intermediates['conversation_summary'] ?? '') !== '') {
    fwrite(STDOUT, sprintf("  %s\n", $intermediates['conversation_summary']));
}

fwrite(STDOUT, "Reasoning steps:\n");
foreach ($intermediates['reasoning_steps'] as $index => $step) {
    $preview = array_slice($step, 0, 4);
    fwrite(STDOUT, sprintf(
        "  Step %d: [%s]\n",
        $index + 1,
        implode(', ', array_map(fn ($v) => sprintf('%.3f', $v), $preview))
    ));
}
