<?php

declare(strict_types=1);

use NSCTX\Fusion\CrossModalFusion;
use PHPUnit\Framework\TestCase;

final class FusionTest extends TestCase
{
    public function testFusionRespectsProvidedWeights(): void
    {
        $fusion = new CrossModalFusion();
        $modalities = [
            'text' => [0.2, 0.1, 0.3],
            'image' => [0.5, 0.4, 0.1],
            'audio' => [0.1, 0.3, 0.2],
        ];
        $weights = [
            'text' => 3.0,
            'image' => 1.0,
            'audio' => 0.2,
        ];

        $result = $fusion->fuse($modalities, $weights);

        $this->assertCount(3, $result['fused']);
        $this->assertArrayHasKey('text', $result['weights']);
        $this->assertArrayHasKey('image', $result['weights']);
        $this->assertArrayHasKey('audio', $result['weights']);
        $this->assertGreaterThan($result['weights']['audio'], $result['weights']['text']);
    }
}
