<?php

declare(strict_types=1);

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;
use PHPUnit\Framework\TestCase;

final class ModelConversationTest extends TestCase
{
    private string $storagePath;

    protected function setUp(): void
    {
        $this->storagePath = sys_get_temp_dir() . '/nsctx-model-' . bin2hex(random_bytes(4)) . '/model.json';
        $this->addTearDownCallback(function (): void {
            $directory = dirname($this->storagePath);
            if (is_dir($directory)) {
                array_map('unlink', glob($directory . '/*') ?: []);
                rmdir($directory);
            }
        });
    }

    public function testModelRetainsConversationAwareSignals(): void
    {
        $storage = new Storage($this->storagePath);
        $model = new NSCTXModel($storage, 8);
        $dataset = [
            [
                'label' => 'greeting',
                'modalities' => [
                    'text' => [
                        ['role' => 'User', 'content' => 'Hello assistant'],
                        ['role' => 'Assistant', 'content' => 'Hello user'],
                    ],
                    'image' => [0.1, 0.2, 0.3, 0.4],
                    'audio' => [0.05, 0.1, 0.15, 0.2],
                ],
            ],
            [
                'label' => 'question',
                'modalities' => [
                    'text' => [
                        ['role' => 'User', 'content' => 'Can you help?'],
                        ['role' => 'Assistant', 'content' => 'Sure thing'],
                    ],
                    'image' => [0.15, 0.1, 0.2, 0.25],
                    'audio' => [0.08, 0.12, 0.1, 0.18],
                ],
            ],
        ];

        $metrics = $model->train($dataset, 0.5, 0.2);
        $this->assertArrayHasKey('train_accuracy', $metrics);
        $this->assertNotEmpty($model->getAlpha());

        $prediction = $model->predict([
            'text' => [
                ['role' => 'User', 'content' => 'Hello assistant'],
                ['role' => 'Assistant', 'content' => 'How can I assist?'],
            ],
            'image' => [0.12, 0.18, 0.24, 0.3],
            'audio' => [0.07, 0.09, 0.11, 0.2],
        ]);

        $intermediates = $prediction['intermediates'];
        $this->assertStringContainsString('user:', strtolower($intermediates['conversation_summary']));
        $this->assertCount(2, $intermediates['conversation_turns']);
        $this->assertSame('user', $intermediates['conversation_turns'][0]['role']);
        $this->assertArrayHasKey('text', $prediction['modal_weights']);
        $this->assertArrayHasKey('image', $prediction['modal_weights']);
        $this->assertArrayHasKey('audio', $prediction['modal_weights']);
    }
}
