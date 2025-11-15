<?php

declare(strict_types=1);

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;
use PHPUnit\Framework\TestCase;

final class TransferLearningTest extends TestCase
{
    private string $storagePath;

    protected function setUp(): void
    {
        $this->storagePath = sys_get_temp_dir() . '/nsctx-transfer-' . bin2hex(random_bytes(4)) . '/model.json';
        $this->addTearDownCallback(function (): void {
            $directory = dirname($this->storagePath);
            if (is_dir($directory)) {
                array_map('unlink', glob($directory . '/*') ?: []);
                rmdir($directory);
            }
        });
    }

    public function testTransferLearningBlendsDatasets(): void
    {
        $model = new NSCTXModel(new Storage($this->storagePath), 8);
        $base = [
            [
                'label' => 'science',
                'modalities' => [
                    'text' => 'The rover documented hydrated minerals near the ridge.',
                    'image' => [0.4, 0.5, 0.3],
                    'audio' => [0.2, 0.2, 0.3],
                ],
            ],
            [
                'label' => 'science',
                'modalities' => [
                    'text' => 'Astronomers confirmed the comet tail was water-rich.',
                    'image' => [0.3, 0.2, 0.6],
                    'audio' => [0.1, 0.3, 0.2],
                ],
            ],
        ];
        $adapt = [
            [
                'label' => 'rescue',
                'modalities' => [
                    'text' => 'Medics triaged a pilot after an engine flare.',
                    'image' => [0.7, 0.5, 0.2],
                    'audio' => [0.3, 0.4, 0.2],
                ],
            ],
            [
                'label' => 'celebration',
                'modalities' => [
                    'text' => 'Crews celebrated the successful array restart.',
                    'image' => [0.2, 0.6, 0.7],
                    'audio' => [0.4, 0.5, 0.3],
                ],
            ],
        ];

        $metrics = $model->transferLearn($base, $adapt, 0.5, 0.75, 0.2);

        $this->assertSame('transfer', $metrics['mode']);
        $this->assertGreaterThan(0, $metrics['carryover_samples']);
        $this->assertArrayHasKey('base_retention', $metrics);
        $this->assertArrayHasKey('adapt_performance', $metrics);
        $this->assertNotSame(null, $metrics['base_retention']);
        $this->assertNotSame(null, $metrics['adapt_performance']);
    }
}
