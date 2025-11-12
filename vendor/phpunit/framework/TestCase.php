<?php

declare(strict_types=1);

namespace PHPUnit\Framework;

use Countable;
use function abs;
use function array_key_exists;
use function count;
use function is_array;
use function is_file;
use function sprintf;
use function str_contains;
use function var_export;

abstract class TestCase
{
    /** @var list<callable> */
    private array $cleanupCallbacks = [];

    protected function setUp(): void
    {
        // Override in subclasses
    }

    protected function tearDown(): void
    {
        while ($this->cleanupCallbacks !== []) {
            $callback = array_pop($this->cleanupCallbacks);
            $callback();
        }
    }

    /**
     * @param callable(): void $callback
     */
    protected function addTearDownCallback(callable $callback): void
    {
        $this->cleanupCallbacks[] = $callback;
    }

    /**
     * @param mixed $haystack
     */
    protected function assertCount(int $expected, $haystack, string $message = ''): void
    {
        $actual = null;
        if (is_array($haystack) || $haystack instanceof Countable) {
            $actual = count($haystack);
        }
        if ($actual === null) {
            $this->fail($message !== '' ? $message : 'Failed asserting that value is countable.');
        }
        if ($actual !== $expected) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that actual size %d matches expected %d.', $actual, $expected));
        }
    }

    protected function assertSame($expected, $actual, string $message = ''): void
    {
        if ($expected !== $actual) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that %s is identical to %s.', var_export($actual, true), var_export($expected, true)));
        }
    }

    protected function assertNotSame($expected, $actual, string $message = ''): void
    {
        if ($expected === $actual) {
            $this->fail($message !== '' ? $message : 'Failed asserting that values are not identical.');
        }
    }

    protected function assertGreaterThan(float $expected, float $actual, string $message = ''): void
    {
        if (!($actual > $expected)) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that %.5f is greater than %.5f.', $actual, $expected));
        }
    }

    protected function assertGreaterThanOrEqual(float $expected, float $actual, string $message = ''): void
    {
        if (!($actual >= $expected)) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that %.5f is greater than or equal to %.5f.', $actual, $expected));
        }
    }

    protected function assertLessThan(float $expected, float $actual, string $message = ''): void
    {
        if (!($actual < $expected)) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that %.5f is less than %.5f.', $actual, $expected));
        }
    }

    protected function assertNotEmpty($value, string $message = ''): void
    {
        if (empty($value)) {
            $this->fail($message !== '' ? $message : 'Failed asserting that value is not empty.');
        }
    }

    protected function assertEqualsWithDelta($expected, $actual, float $delta, string $message = ''): void
    {
        if (is_array($expected) && is_array($actual)) {
            $this->assertCount(count($expected), $actual, $message);
            foreach ($expected as $index => $value) {
                $this->assertEqualsWithDelta($value, $actual[$index], $delta, $message);
            }
            return;
        }

        if (abs($expected - $actual) > $delta) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that %.5f matches %.5f within %.5f.', $actual, $expected, $delta));
        }
    }

    protected function assertFileExists(string $path, string $message = ''): void
    {
        if (!is_file($path)) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that file exists: %s', $path));
        }
    }

    protected function assertStringContainsString(string $needle, string $haystack, string $message = ''): void
    {
        if (!str_contains($haystack, $needle)) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that "%s" contains "%s".', $haystack, $needle));
        }
    }

    /**
     * @param array-key $key
     * @param array<mixed> $array
     */
    protected function assertArrayHasKey($key, array $array, string $message = ''): void
    {
        if (!array_key_exists($key, $array)) {
            $this->fail($message !== '' ? $message : sprintf('Failed asserting that key %s exists in array.', var_export($key, true)));
        }
    }

    protected function assertTrue(bool $condition, string $message = ''): void
    {
        if (!$condition) {
            $this->fail($message !== '' ? $message : 'Failed asserting that condition is true.');
        }
    }

    protected function assertFalse(bool $condition, string $message = ''): void
    {
        if ($condition) {
            $this->fail($message !== '' ? $message : 'Failed asserting that condition is false.');
        }
    }

    protected function fail(string $message): void
    {
        throw new AssertionFailedError($message);
    }
}
