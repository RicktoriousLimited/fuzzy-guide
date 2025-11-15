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

        $response = strtolower($chat['response']);
        $this->assertStringContainsString('you are asking', $response);
        $this->assertStringContainsString('archived', $response);
        $this->assertStringContainsString('confidence', $response);
        $this->assertGreaterThan(0.0, $chat['match_score']);
        $this->assertNotSame(null, $chat['memory']);
    }

    public function testChatProvidesInterpretationWithoutMemory(): void
    {
        $model = new NSCTXModel(new Storage($this->storagePath));

        $chat = $model->chat([
            ['role' => 'user', 'content' => 'Can you help me plan my next rover checkup?'],
        ]);

        $response = strtolower($chat['response']);
        $this->assertStringContainsString('do not have a saved note', $response);
        $this->assertStringContainsString('tell me more', $response);
        $this->assertSame(0.0, $chat['match_score']);
        $this->assertSame(null, $chat['memory']);
    }
}
