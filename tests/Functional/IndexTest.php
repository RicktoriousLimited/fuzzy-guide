<?php

declare(strict_types=1);

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;
use PHPUnit\Framework\TestCase;

final class IndexTest extends TestCase
{
    private string $storagePath;

    protected function setUp(): void
    {
        $this->storagePath = __DIR__ . '/../../php-app/storage/model.json';
        if (is_file($this->storagePath)) {
            unlink($this->storagePath);
        }
        if (!is_dir(dirname($this->storagePath))) {
            mkdir(dirname($this->storagePath), 0775, true);
        }
        $this->addTearDownCallback(function (): void {
            if (is_file($this->storagePath)) {
                unlink($this->storagePath);
            }
        });
    }

    public function testTeachActionPersistsMemory(): void
    {
        $config = [
            'method' => 'POST',
            'post' => [
                'action' => 'teach',
                'passage' => 'Research logs describe the coral reef survey in detail.',
            ],
            'output' => $this->createTempFilePath(),
        ];

        $this->runIndex($config);
        $output = file_get_contents($config['output']);
        $this->assertStringContainsString('New passage stored in memory.', $output ?: '');
        $this->assertStringContainsString('Memory bank', $output ?: '');
    }

    public function testChatActionDisplaysLatestReplyPanel(): void
    {
        // teach the chatbot first so memory exists
        $teachConfig = [
            'method' => 'POST',
            'post' => [
                'action' => 'teach',
                'passage' => 'The observatory detected a comet near Jupiter.',
            ],
            'output' => $this->createTempFilePath(),
        ];
        $this->runIndex($teachConfig);

        $chatConfig = [
            'method' => 'POST',
            'post' => [
                'action' => 'chat',
                'message' => 'Remind me what the observatory saw.',
            ],
            'output' => $this->createTempFilePath(),
        ];

        $this->runIndex($chatConfig);
        $output = file_get_contents($chatConfig['output']);
        $this->assertStringContainsString('Assistant replied.', $output ?: '');
        $this->assertStringContainsString('Latest reply', $output ?: '');
        $this->assertStringContainsString('Matched memory', $output ?: '');
    }

    public function testResetActionShowsConfirmation(): void
    {
        $config = [
            'method' => 'POST',
            'post' => [
                'action' => 'reset_chat',
            ],
            'output' => $this->createTempFilePath(),
        ];

        $this->runIndex($config);
        $output = file_get_contents($config['output']);
        $this->assertStringContainsString('Conversation reset.', $output ?: '');
        $this->assertStringContainsString('You are the NSCTX memory assistant', $output ?: '');
    }

    /**
     * @param array<string, mixed> $config
     */
    private function runIndex(array $config): void
    {
        $tempConfigPath = $this->createTempFilePath();
        file_put_contents($tempConfigPath, json_encode($config, JSON_PRETTY_PRINT));
        $this->addTearDownCallback(static function () use ($tempConfigPath): void {
            if (is_file($tempConfigPath)) {
                unlink($tempConfigPath);
            }
        });
        $command = sprintf(
            'php %s %s',
            escapeshellarg(__DIR__ . '/index_runner.php'),
            escapeshellarg($tempConfigPath)
        );
        exec($command, $output, $exitCode);
        if ($exitCode !== 0) {
            $this->fail('Index runner failed: ' . implode("\n", $output));
        }
    }

    private function createTempFilePath(): string
    {
        $path = sys_get_temp_dir() . '/nsctx-index-' . bin2hex(random_bytes(4));
        $this->addTearDownCallback(static function () use ($path): void {
            if (is_file($path)) {
                unlink($path);
            }
        });
        return $path;
    }
}
