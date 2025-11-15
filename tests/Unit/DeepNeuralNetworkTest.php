<?php

declare(strict_types=1);

use NSCTX\Model\DeepNeuralNetwork;
use PHPUnit\Framework\TestCase;

final class DeepNeuralNetworkTest extends TestCase
{
    public function testLearnsSimpleDecisionBoundary(): void
    {
        $network = new DeepNeuralNetwork();
        $network->initialize(2, 2, 4, 2, true);

        $features = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        $targets = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ];

        $network->train($features, $targets, 200);

        $lowProb = $network->predict([0.0, 0.0]);
        $highProb = $network->predict([1.0, 1.0]);

        $this->assertGreaterThan(0.6, $lowProb[0]);
        $this->assertGreaterThan(0.6, $highProb[1]);
    }
}
