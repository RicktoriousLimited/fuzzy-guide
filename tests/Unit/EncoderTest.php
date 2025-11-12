<?php

declare(strict_types=1);

use NSCTX\Encoder\NumericEncoder;
use NSCTX\Encoder\TextEncoder;
use NSCTX\Support\Math;
use PHPUnit\Framework\TestCase;

final class EncoderTest extends TestCase
{
    public function testTextEncoderProducesEmbeddingsAndContextualVector(): void
    {
        $encoder = new TextEncoder(6);
        $result = $encoder->encode('Alpha Beta Gamma', ['user', 'assistant', 'system']);

        $this->assertCount(3, $result['embeddings']);
        foreach ($result['embeddings'] as $embedding) {
            $this->assertCount(6, $embedding);
        }

        $this->assertCount(6, $result['contextual']);
        $this->assertGreaterThan(0.0, Math::norm($result['contextual']));
    }

    public function testTextEncoderHandlesEmptyText(): void
    {
        $encoder = new TextEncoder(4);
        $result = $encoder->encode('', []);

        $this->assertSame([], $result['embeddings']);
        $this->assertCount(4, $result['contextual']);
        $this->assertLessThan(0.01, Math::norm($result['contextual']));
    }

    public function testNumericEncoderProjectsValuesIntoEmbeddingSpace(): void
    {
        $encoder = new NumericEncoder(5, 'image');
        $vector = $encoder->encode([0.5, 0.25, 0.75]);

        $this->assertCount(5, $vector);
        $this->assertGreaterThan(0.0, Math::norm($vector));
    }
}
