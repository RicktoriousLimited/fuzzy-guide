<?php

declare(strict_types=1);

use NSCTX\Model\Storage;
use PHPUnit\Framework\TestCase;

final class StorageTest extends TestCase
{
    private string $storagePath;

    protected function setUp(): void
    {
        $this->storagePath = sys_get_temp_dir() . '/nsctx-storage-' . bin2hex(random_bytes(4)) . '/model.json';
    }

    protected function tearDown(): void
    {
        $directory = dirname($this->storagePath);
        if (is_dir($directory)) {
            array_map('unlink', glob($directory . '/*') ?: []);
            rmdir($directory);
        }
    }

    public function testLoadReturnsEmptyArrayWhenFileMissing(): void
    {
        $storage = new Storage($this->storagePath);
        $this->assertSame([], $storage->load());
    }

    public function testSavePersistsStateToDisk(): void
    {
        $storage = new Storage($this->storagePath);
        $state = ['trained_at' => '2025-01-01T00:00:00Z'];
        $storage->save($state);

        $this->assertFileExists($this->storagePath);
        $this->assertSame($state, $storage->load());
    }
}
