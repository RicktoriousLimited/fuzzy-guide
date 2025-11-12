<?php

declare(strict_types=1);

use NSCTX\Reasoning\ReasoningEngine;
use PHPUnit\Framework\TestCase;

final class ReasoningEngineTest extends TestCase
{
    public function testReasoningProducesHistoryAndFinalState(): void
    {
        $engine = new ReasoningEngine();
        $nodes = [
            ['representation' => [0.1, 0.2, 0.3]],
            ['representation' => [0.4, 0.5, 0.6]],
        ];

        $result = $engine->run($nodes, 3);

        $this->assertCount(3, $result['steps']);
        $this->assertCount(3, $result['state']);
        $this->assertEqualsWithDelta(end($result['steps']), $result['state'], 1e-9);
    }
}
