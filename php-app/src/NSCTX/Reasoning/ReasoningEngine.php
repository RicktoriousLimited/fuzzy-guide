<?php

declare(strict_types=1);

namespace NSCTX\Reasoning;

use NSCTX\Support\Hasher;
use NSCTX\Support\Math;
use NSCTX\Support\Vector;

final class ReasoningEngine
{
    /**
     * @param array<int, array{representation: array<int, float>}> $nodes
     * @return array{state: array<int, float>, steps: array<int, array<int, float>>}
     */
    public function run(array $nodes, int $steps = 2): array
    {
        if ($nodes === []) {
            return ['state' => [], 'steps' => []];
        }
        $state = Vector::average(array_map(
            static fn (array $node): array => $node['representation'],
            $nodes
        ));
        $history = [];
        for ($step = 0; $step < $steps; $step++) {
            $message = Hasher::project($state, count($state), 'reason:msg:' . $step);
            $state = Math::layerNorm(Math::gelu(Vector::add($state, $message)));
            $history[] = $state;
        }
        return [
            'state' => $state,
            'steps' => $history,
        ];
    }
}
