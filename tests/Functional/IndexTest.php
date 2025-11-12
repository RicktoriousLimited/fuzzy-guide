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

    public function testTrainingFlowDisplaysSuccessMessage(): void
    {
        $config = [
            'method' => 'POST',
            'post' => [
                'action' => 'train',
                'train_ratio' => '0.5',
                'ewc' => '0.1',
            ],
            'output' => $this->createTempFilePath(),
        ];

        $this->runIndex($config);

        $output = file_get_contents($config['output']);
        $this->assertStringContainsString('Training complete', $output ?: '');
    }

    public function testPredictionFlowDisplaysConversationSummary(): void
    {
        $this->trainModelFixture();

        $conversation = implode("\n", [
            'User: Hello assistant',
            'Assistant: Hello user, how can I help?',
        ]);

        $config = [
            'method' => 'POST',
            'post' => [
                'action' => 'predict',
                'text' => $conversation,
                'image' => '0.1,0.2,0.3',
                'audio' => '0.05,0.15,0.2',
            ],
            'output' => $this->createTempFilePath(),
        ];

        $this->runIndex($config);

        $output = file_get_contents($config['output']);
        $this->assertStringContainsString('Prediction generated.', $output ?: '');
        $this->assertStringContainsString('Conversation summary', $output ?: '');
    }

    private function trainModelFixture(): void
    {
        $storage = new Storage($this->storagePath);
        $model = new NSCTXModel($storage);
        $datasetPath = __DIR__ . '/../../php-app/data/dataset.json';
        $payload = json_decode(file_get_contents($datasetPath) ?: 'null', true);
        $samples = $payload['samples'] ?? [];
        $model->train($samples, 0.6, 0.1);
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
