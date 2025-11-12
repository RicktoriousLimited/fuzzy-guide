<?php

declare(strict_types=1);

namespace NSCTX\Fusion;

use NSCTX\Support\Math;
use NSCTX\Support\Vector;

final class CrossModalFusion
{
    /**
     * @param array<string, array<int, float>> $modalities
     * @param array<string, float> $weights
     * @return array{fused: array<int, float>, weights: array<string, float>}
     */
    public function fuse(array $modalities, array $weights = []): array
    {
        if ($modalities === []) {
            return [
                'fused' => [],
                'weights' => [],
            ];
        }

        $energy = [];
        foreach ($modalities as $name => $vector) {
            $energy[$name] = Math::norm($vector);
        }

        $weightValues = [];
        $names = array_keys($modalities);
        if ($weights !== []) {
            foreach ($names as $name) {
                $weightValues[] = $weights[$name] ?? 1.0;
            }
        } else {
            foreach ($names as $name) {
                $weightValues[] = $energy[$name] ?? 1.0;
            }
        }
        $normalizedWeights = Math::softmax($weightValues);

        $fused = array_fill(0, count(reset($modalities)), 0.0);
        foreach ($names as $index => $name) {
            $vector = $modalities[$name];
            $weight = $normalizedWeights[$index] ?? (1.0 / max(count($modalities), 1));
            $scaled = Vector::scale($vector, $weight);
            $fused = Vector::add($fused, $scaled);
        }

        return [
            'fused' => Math::layerNorm(Math::gelu($fused)),
            'weights' => array_combine($names, $normalizedWeights),
        ];
    }
}
