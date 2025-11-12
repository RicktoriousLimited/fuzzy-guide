<?php

declare(strict_types=1);

namespace NSCTX\Decoder;

use NSCTX\Support\Math;
use NSCTX\Support\Vector;

final class HubDecoder
{
    /**
     * @param array<int, float> $hubState
     * @param array<string, array<int, float>> $prototypes
     * @return array{
     *     prediction: string,
     *     probabilities: array<string, float>
     * }
     */
    public function decode(array $hubState, array $prototypes): array
    {
        if ($hubState === [] || $prototypes === []) {
            return [
                'prediction' => 'unknown',
                'probabilities' => [],
            ];
        }
        $scores = [];
        foreach ($prototypes as $label => $prototype) {
            $scores[$label] = Math::cosine($hubState, $prototype);
        }
        $labels = array_keys($scores);
        $values = array_values($scores);
        $probabilities = Math::softmax($values);
        $probabilityMap = [];
        foreach ($labels as $index => $label) {
            $probabilityMap[$label] = $probabilities[$index] ?? 0.0;
        }
        arsort($probabilityMap);
        $prediction = array_key_first($probabilityMap) ?? 'unknown';
        return [
            'prediction' => $prediction,
            'probabilities' => $probabilityMap,
        ];
    }

    /**
     * @param array<int, float> $hubState
     * @param array<int, array<int, float>> $relayStates
     */
    public function relay(array $hubState, array $relayStates): array
    {
        $updated = $hubState;
        foreach ($relayStates as $index => $state) {
            $scaled = Vector::scale($state, 1.0 / (1 + $index));
            $updated = Vector::add($updated, $scaled);
        }
        return Math::layerNorm($updated);
    }
}
