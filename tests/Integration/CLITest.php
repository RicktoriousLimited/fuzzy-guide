<?php

declare(strict_types=1);

use PHPUnit\Framework\TestCase;

final class CLITest extends TestCase
{
    private string $projectRoot;

    protected function setUp(): void
    {
        $this->projectRoot = dirname(__DIR__, 1);
    }

    public function testTrainCommandProducesModelAndMetrics(): void
    {
        $dataset = $this->createTempDataset();
        $modelPath = $this->createTempModelPath();

        $result = $this->runPhpScript('php-app/train.php', [
            '--dataset=' . $dataset,
            '--model=' . $modelPath,
            '--train-ratio=0.6',
            '--ewc=0.1',
        ]);

        $this->assertSame(0, $result['exitCode']);
        $this->assertStringContainsString('Training complete!', $result['stdout']);
        $this->assertStringContainsString('Samples: 3', $result['stdout']);
        $this->assertFileExists($modelPath);
    }

    public function testPredictCommandProducesConversationAwareOutput(): void
    {
        $dataset = $this->createTempDataset();
        $modelPath = $this->createTempModelPath();

        $train = $this->runPhpScript('php-app/train.php', [
            '--dataset=' . $dataset,
            '--model=' . $modelPath,
        ]);
        $this->assertSame(0, $train['exitCode']);

        $result = $this->runPhpScript('php-app/predict.php', [
            '--model=' . $modelPath,
            '--turn=user: hello there',
            '--turn=assistant: hi again',
        ]);

        $this->assertSame(0, $result['exitCode']);
        $this->assertStringContainsString('Prediction:', $result['stdout']);
        $this->assertStringContainsString('Conversation summary', $result['stdout']);
        $this->assertStringContainsString('Reasoning steps', $result['stdout']);
    }

    private function createTempDataset(): string
    {
        $source = __DIR__ . '/../Fixtures/cli-dataset.json';
        $path = sys_get_temp_dir() . '/nsctx-dataset-' . bin2hex(random_bytes(4)) . '.json';
        copy($source, $path);
        $this->addTearDownCallback(static function () use ($path): void {
            if (is_file($path)) {
                unlink($path);
            }
        });
        return $path;
    }

    private function createTempModelPath(): string
    {
        $directory = sys_get_temp_dir() . '/nsctx-model-' . bin2hex(random_bytes(4));
        $path = $directory . '/model.json';
        $this->addTearDownCallback(static function () use ($directory): void {
            if (is_dir($directory)) {
                array_map('unlink', glob($directory . '/*') ?: []);
                rmdir($directory);
            }
        });
        return $path;
    }

    /**
     * @param list<string> $arguments
     * @return array{stdout: string, stderr: string, exitCode: int}
     */
    private function runPhpScript(string $script, array $arguments): array
    {
        $command = ['php', $this->projectRoot . '/../' . $script];
        foreach ($arguments as $argument) {
            $command[] = $argument;
        }
        $descriptor = [
            0 => ['pipe', 'r'],
            1 => ['pipe', 'w'],
            2 => ['pipe', 'w'],
        ];
        $process = proc_open($this->escapeCommand($command), $descriptor, $pipes, $this->projectRoot . '/..');
        if (!is_resource($process)) {
            return ['stdout' => '', 'stderr' => 'Unable to execute command.', 'exitCode' => 1];
        }
        fclose($pipes[0]);
        $stdout = stream_get_contents($pipes[1]);
        $stderr = stream_get_contents($pipes[2]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        $exitCode = proc_close($process);

        return [
            'stdout' => $stdout === false ? '' : $stdout,
            'stderr' => $stderr === false ? '' : $stderr,
            'exitCode' => $exitCode,
        ];
    }

    /**
     * @param list<string> $command
     */
    private function escapeCommand(array $command): string
    {
        return implode(' ', array_map('escapeshellarg', $command));
    }
}
