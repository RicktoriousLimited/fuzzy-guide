<?php

declare(strict_types=1);

namespace NSCTX\Encoder;

use NSCTX\Support\Hasher;
use NSCTX\Support\Math;

final class NumericEncoder
{
    private int $dimension;
    private string $modality;

    public function __construct(int $dimension, string $modality)
    {
        $this->dimension = $dimension;
        $this->modality = $modality;
    }

    public function getDimension(): int
    {
        return $this->dimension;
    }

    /**
     * @param array<int, float> $values
     * @return array<int, float>
     */
    public function encode(array $values): array
    {
        $projected = Hasher::project($values, $this->dimension, 'numeric:' . $this->modality);
        return Math::layerNorm(Math::gelu($projected));
    }
}
