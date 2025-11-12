<?php

declare(strict_types=1);

namespace PHPUnit\TextUI;

use PHPUnit\Framework\AssertionFailedError;
use PHPUnit\Framework\TestCase;
use ReflectionClass;
use Throwable;
use function array_diff;
use function array_map;
use function array_values;
use function file_exists;
use function glob;
use function is_dir;
use function iterator_to_array;
use function preg_match;
use function realpath;
use function simplexml_load_file;
use function sprintf;

final class TestRunner
{
    /** @var list<string> */
    private array $testFiles = [];

    /** @var list<string> */
    private array $filters;

    public function __construct(array $filters = [])
    {
        $this->filters = $filters;
    }

    public function discoverFromConfig(string $configPath): void
    {
        if (!file_exists($configPath)) {
            throw new \RuntimeException(sprintf('Configuration file not found: %s', $configPath));
        }
        $xml = simplexml_load_file($configPath);
        if ($xml === false) {
            throw new \RuntimeException(sprintf('Unable to parse configuration file: %s', $configPath));
        }
        $directories = [];
        if (isset($xml->testsuites)) {
            foreach ($xml->testsuites->testsuite as $suite) {
                foreach ($suite->directory as $directory) {
                    $path = realpath((string) $directory);
                    if ($path !== false && is_dir($path)) {
                        $directories[] = $path;
                    }
                }
            }
        }
        if ($directories === []) {
            $default = realpath('tests');
            if ($default !== false) {
                $directories[] = $default;
            }
        }
        foreach ($directories as $directory) {
            $this->discoverFromDirectory($directory);
        }
    }

    public function discoverFromDirectory(string $directory): void
    {
        $pattern = rtrim($directory, '/\\') . DIRECTORY_SEPARATOR . '*Test.php';
        $files = glob($pattern);
        if ($files !== false) {
            foreach ($files as $file) {
                $this->testFiles[] = $file;
            }
        }
        $subdirectories = glob(rtrim($directory, '/\\') . DIRECTORY_SEPARATOR . '*', GLOB_ONLYDIR);
        if ($subdirectories !== false) {
            foreach ($subdirectories as $subdirectory) {
                $this->discoverFromDirectory($subdirectory);
            }
        }
    }

    public function run(): int
    {
        $total = 0;
        $failures = 0;
        $errors = 0;
        foreach ($this->testFiles as $file) {
            $before = get_declared_classes();
            require_once $file;
            $after = get_declared_classes();
            $newClasses = array_values(array_diff($after, $before));
            foreach ($newClasses as $className) {
                $reflection = new ReflectionClass($className);
                if (!$reflection->isSubclassOf(TestCase::class) || $reflection->isAbstract()) {
                    continue;
                }
                $methods = array_map(
                    static fn ($method) => $method->getName(),
                    iterator_to_array($reflection->getMethods())
                );
                foreach ($methods as $method) {
                    if (!preg_match('/^test[A-Z0-9_]/', $method)) {
                        continue;
                    }
                    if ($this->filters !== []) {
                        $matchesFilter = false;
                        foreach ($this->filters as $filter) {
                            if (preg_match('/' . $filter . '/i', $method)) {
                                $matchesFilter = true;
                                break;
                            }
                        }
                        if (!$matchesFilter) {
                            continue;
                        }
                    }
                    $total++;
                    $instance = $reflection->newInstance();
                    try {
                        $reflection->getMethod('setUp')->invoke($instance);
                        $reflection->getMethod($method)->invoke($instance);
                        echo '.';
                    } catch (AssertionFailedError $assertionFailedError) {
                        $failures++;
                        echo 'F';
                        $this->printFailure($className, $method, $assertionFailedError);
                    } catch (Throwable $throwable) {
                        $errors++;
                        echo 'E';
                        $this->printFailure($className, $method, $throwable);
                    } finally {
                        $reflection->getMethod('tearDown')->invoke($instance);
                    }
                }
            }
        }
        if ($total === 0) {
            echo "No tests executed.\n";
            return 1;
        }
        echo "\n";
        echo sprintf("Tests: %d, Failures: %d, Errors: %d\n", $total, $failures, $errors);
        if ($failures === 0 && $errors === 0) {
            echo "OK\n";
            return 0;
        }
        echo "FAILURES!\n";
        return 1;
    }

    private function printFailure(string $className, string $method, Throwable $throwable): void
    {
        echo sprintf("\n%s::%s\n%s\n", $className, $method, $throwable->getMessage());
    }
}
