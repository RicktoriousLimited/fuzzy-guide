<?php

declare(strict_types=1);

require_once __DIR__ . '/../php-app/bootstrap.php';

spl_autoload_register(
    static function (string $class): void {
        $prefix = 'PHPUnit\\Framework\\';
        if (str_starts_with($class, $prefix)) {
            $relative = substr($class, strlen($prefix));
            $path = __DIR__ . '/phpunit/framework/' . $relative . '.php';
            if (is_file($path)) {
                require $path;
            }
        }
        if ($class === 'PHPUnit\\TextUI\\TestRunner') {
            require __DIR__ . '/phpunit/framework/TestRunner.php';
        }
    }
);

return true;
