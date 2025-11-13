<?php

declare(strict_types=1);

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;
use PHPUnit\Framework\TestCase;

final class ChatbotLearningTest extends TestCase
{
    private string $storagePath;

    protected function setUp(): void
    {
        $this->storagePath = sys_get_temp_dir() . '/nsctx-chat-' . bin2hex(random_bytes(4)) . '/model.json';
        $this->addTearDownCallback(function (): void {
            $directory = dirname($this->storagePath);
            if (is_dir($directory)) {
                array_map('unlink', glob($directory . '/*') ?: []);
                rmdir($directory);
            }
        });
    }

    public function testModelLearnsPassagesWithoutLabels(): void
    {
        $model = new NSCTXModel(new Storage($this->storagePath));
        $entry = $model->learnFromPassage('Mission logs confirm the rover found hydrated minerals on Mars.');

        $this->assertArrayHasKey('summary', $entry);
        $this->assertArrayHasKey('vector', $entry);
        $this->assertNotEmpty($model->getMemoryBank());
    }

    public function testChatUsesLearnedPassages(): void
    {
        $model = new NSCTXModel(new Storage($this->storagePath));
        $model->learnFromPassage('The observatory detected a bright comet passing near Jupiter.');

        $chat = $model->chat([
            ['role' => 'user', 'content' => 'Remind me what the observatory spotted.'],
        ]);

        $this->assertStringContainsString('prior knowledge', strtolower($chat['response']));
        $this->assertGreaterThan(0.0, $chat['match_score']);
        $this->assertNotSame(null, $chat['memory']);
    }
}
