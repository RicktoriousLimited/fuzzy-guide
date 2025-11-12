<?php

declare(strict_types=1);

spl_autoload_register(
    static function (string $class): void {
        if (!str_starts_with($class, 'NSCTX\\')) {
            return;
        }
        $relative = str_replace('\\', '/', $class);
        $path = __DIR__ . '/src/' . $relative . '.php';
        if (is_file($path)) {
            require $path;
        }
    }
);
