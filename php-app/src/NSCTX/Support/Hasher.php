<?php

declare(strict_types=1);

namespace NSCTX\Support;

final class Hasher
{
    public static function float(string $seed, float $scale = 1.0): float
    {
        $hash = hash('sha256', $seed, true);
        $bytes = substr($hash, 0, 8);
        $value = unpack('J', $bytes)[1];
        $normalized = ($value / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0;
        return $normalized * $scale;
    }

    /**
     * @param array<int, float> $vector
     */
    public static function project(array $vector, int $dimension, string $context): array
    {
        $output = array_fill(0, $dimension, 0.0);
        $count = count($vector);
        for ($i = 0; $i < $dimension; $i++) {
            $sum = 0.0;
            for ($j = 0; $j < $count; $j++) {
                $weight = self::float($context . ':' . $i . ':' . $j);
                $sum += $vector[$j] * $weight;
            }
            $output[$i] = $sum;
        }
        return $output;
    }
}
