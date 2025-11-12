<?php

declare(strict_types=1);

namespace NSCTX\Support;

final class Vector
{
    /**
     * @param array<int, float> $values
     */
    public static function zeros(int $dimension): array
    {
        return array_fill(0, $dimension, 0.0);
    }

    /**
     * @param array<int, float> $a
     * @param array<int, float> $b
     */
    public static function add(array $a, array $b): array
    {
        $count = min(count($a), count($b));
        $result = [];
        for ($i = 0; $i < $count; $i++) {
            $result[$i] = $a[$i] + $b[$i];
        }
        return $result;
    }

    /**
     * @param array<int, float> $a
     * @param array<int, float> $b
     */
    public static function subtract(array $a, array $b): array
    {
        $count = min(count($a), count($b));
        $result = [];
        for ($i = 0; $i < $count; $i++) {
            $result[$i] = $a[$i] - $b[$i];
        }
        return $result;
    }

    /**
     * @param array<int, float> $vector
     */
    public static function scale(array $vector, float $factor): array
    {
        return array_map(
            static fn (float $value) => $value * $factor,
            $vector
        );
    }

    /**
     * @param array<int, array<int, float>> $vectors
     */
    public static function average(array $vectors): array
    {
        if ($vectors === []) {
            return [];
        }
        $dimension = count($vectors[0]);
        $sum = array_fill(0, $dimension, 0.0);
        foreach ($vectors as $vector) {
            for ($i = 0; $i < $dimension; $i++) {
                $sum[$i] += $vector[$i];
            }
        }
        $count = count($vectors);
        for ($i = 0; $i < $dimension; $i++) {
            $sum[$i] /= $count;
        }
        return $sum;
    }
}
