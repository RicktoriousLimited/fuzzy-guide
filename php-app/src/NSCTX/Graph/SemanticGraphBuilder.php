<?php

declare(strict_types=1);

namespace NSCTX\Graph;

use NSCTX\Support\Hasher;
use NSCTX\Support\Math;

final class SemanticGraphBuilder
{
    /**
     * @param array<int, float> $contextVector
     * @return array{
     *     nodes: array<int, array{span: array<int, float>, representation: array<int, float>}>,
     *     edges: array<int, array{source: int, target: int, weight: float}>
     * }
     */
    public function build(array $contextVector): array
    {
        if ($contextVector === []) {
            return ['nodes' => [], 'edges' => []];
        }
        $segmentSize = max(4, (int) floor(count($contextVector) / 4));
        $nodes = [];
        for ($offset = 0; $offset < count($contextVector); $offset += $segmentSize) {
            $span = array_slice($contextVector, $offset, $segmentSize);
            if ($span === []) {
                continue;
            }
            $projection = Hasher::project($span, $segmentSize, 'graph:node:' . $offset);
            $nodes[] = [
                'span' => $span,
                'representation' => Math::layerNorm($projection),
            ];
        }

        $edges = [];
        $nodeCount = count($nodes);
        for ($i = 0; $i < $nodeCount; $i++) {
            for ($j = 0; $j < $nodeCount; $j++) {
                if ($i === $j) {
                    continue;
                }
                $similarity = Math::cosine($nodes[$i]['representation'], $nodes[$j]['representation']);
                if ($similarity <= 0.0) {
                    continue;
                }
                $edges[] = [
                    'source' => $i,
                    'target' => $j,
                    'weight' => $similarity,
                ];
            }
        }

        return [
            'nodes' => $nodes,
            'edges' => $edges,
        ];
    }
}
