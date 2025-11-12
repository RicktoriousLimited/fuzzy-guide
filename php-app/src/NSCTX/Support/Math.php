<?php

declare(strict_types=1);

namespace NSCTX\Support;

final class Math
{
    private const EPSILON = 1e-6;

    /**
     * @param array<int, float> $a
     * @param array<int, float> $b
     */
    public static function dot(array $a, array $b): float
    {
        $sum = 0.0;
        $count = min(count($a), count($b));
        for ($i = 0; $i < $count; $i++) {
            $sum += $a[$i] * $b[$i];
        }
        return $sum;
    }

    /**
     * @param array<int, float> $vector
     */
    public static function norm(array $vector): float
    {
        return sqrt(max(self::dot($vector, $vector), self::EPSILON));
    }

    /**
     * @param array<int, float> $vector
     */
    public static function layerNorm(array $vector): array
    {
        $count = count($vector);
        if ($count === 0) {
            return [];
        }
        $mean = array_sum($vector) / $count;
        $variance = 0.0;
        foreach ($vector as $value) {
            $variance += ($value - $mean) ** 2;
        }
        $variance /= $count;
        $denominator = sqrt($variance + self::EPSILON);
        return array_map(
            static fn (float $value): float => ($value - $mean) / $denominator,
            $vector
        );
    }

    /**
     * @param array<int, float> $vector
     */
    public static function gelu(array $vector): array
    {
        return array_map(
            static fn (float $value): float => 0.5 * $value * (1.0 + tanh(
                sqrt(2.0 / M_PI) * ($value + 0.044715 * ($value ** 3))
            )),
            $vector
        );
    }

    /**
     * @param array<int, float> $vector
     */
    public static function softmax(array $vector): array
    {
        if ($vector === []) {
            return [];
        }
        $max = max($vector);
        $exp = array_map(
            static fn (float $value): float => exp($value - $max),
            $vector
        );
        $sum = array_sum($exp);
        if ($sum <= self::EPSILON) {
            $count = count($exp);
            return array_fill(0, $count, 1.0 / $count);
        }
        return array_map(
            static fn (float $value) => $value / $sum,
            $exp
        );
    }

    /**
     * @param array<int, float> $vector
     */
    public static function normalize(array $vector): array
    {
        $norm = self::norm($vector);
        if ($norm <= self::EPSILON) {
            return array_fill(0, count($vector), 0.0);
        }
        return array_map(
            static fn (float $value) => $value / $norm,
            $vector
        );
    }

    /**
     * @param array<int, float> $a
     * @param array<int, float> $b
     */
    public static function cosine(array $a, array $b): float
    {
        $normProduct = self::norm($a) * self::norm($b);
        if ($normProduct <= self::EPSILON) {
            return 0.0;
        }
        return self::dot($a, $b) / $normProduct;
    }

    /**
     * @param array<int, float> $vector
     */
    public static function clamp(array $vector, float $limit = 5.0): array
    {
        return array_map(
            static fn (float $value) => max(min($value, $limit), -$limit),
            $vector
        );
    }
}
