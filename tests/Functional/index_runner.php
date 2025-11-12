<?php

declare(strict_types=1);

if ($argc < 2) {
    fwrite(STDERR, "Usage: php index_runner.php <config.json>\n");
    exit(1);
}

$configPath = $argv[1];
if (!is_file($configPath)) {
    fwrite(STDERR, "Config file not found: {$configPath}\n");
    exit(1);
}

$config = json_decode(file_get_contents($configPath) ?: 'null', true);
if (!is_array($config)) {
    fwrite(STDERR, "Invalid config payload.\n");
    exit(1);
}

$_SERVER['REQUEST_METHOD'] = $config['method'] ?? 'GET';
$_POST = $config['post'] ?? [];
$_GET = $config['get'] ?? [];

ob_start();
require __DIR__ . '/../../php-app/index.php';
$output = ob_get_clean();

if (($config['output'] ?? '') !== '') {
    file_put_contents($config['output'], $output);
    exit(0);
}

echo $output;
