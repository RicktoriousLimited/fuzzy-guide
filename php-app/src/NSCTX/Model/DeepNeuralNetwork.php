<?php

declare(strict_types=1);

namespace NSCTX\Model;

use NSCTX\Support\Math;

/**
 * Lightweight multi-layer perceptron trained with vanilla SGD.
 */
final class DeepNeuralNetwork
{
    private float $learningRate;

    /**
     * @var array<int, array{weights: array<int, array<int, float>>, bias: array<int, float>}>
     */
    private array $layers;

    /**
     * @param array<string, mixed>|null $state
     */
    public function __construct(?array $state = null, float $learningRate = 0.05)
    {
        $this->learningRate = $state['learning_rate'] ?? $learningRate;
        $this->layers = $state['layers'] ?? [];
    }

    public function isReady(): bool
    {
        return $this->layers !== [];
    }

    public function reset(): void
    {
        $this->layers = [];
    }

    /**
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'learning_rate' => $this->learningRate,
            'layers' => $this->layers,
        ];
    }

    public function initialize(
        int $inputDim,
        int $outputDim,
        int $hiddenDim = 64,
        int $hiddenLayers = 2,
        bool $force = false
    ): void {
        if ($this->isReady() && !$force) {
            return;
        }

        $structure = [$inputDim];
        for ($i = 0; $i < $hiddenLayers; $i++) {
            $structure[] = $hiddenDim;
        }
        $structure[] = $outputDim;

        $layers = [];
        $layerCount = count($structure) - 1;
        for ($layer = 0; $layer < $layerCount; $layer++) {
            $inDim = $structure[$layer];
            $outDim = $structure[$layer + 1];
            $layers[] = [
                'weights' => $this->randomMatrix($outDim, $inDim),
                'bias' => array_fill(0, $outDim, 0.0),
            ];
        }

        $this->layers = $layers;
    }

    /**
     * @param array<int, array<int, float>> $features
     * @param array<int, array<int, float>> $targets
     */
    public function train(array $features, array $targets, int $epochs = 60): void
    {
        if ($features === [] || $targets === [] || $this->layers === []) {
            return;
        }

        $count = min(count($features), count($targets));
        if ($count === 0) {
            return;
        }

        $indices = range(0, $count - 1);
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            shuffle($indices);
            foreach ($indices as $index) {
                $feature = $features[$index];
                $target = $targets[$index];
                $this->trainSample($feature, $target);
            }
        }
    }

    /**
     * @return array<int, float>
     */
    public function predict(array $input): array
    {
        if ($this->layers === []) {
            return [];
        }

        $forward = $this->forwardDetailed($input);
        return $forward['probabilities'];
    }

    /**
     * @return array<int, array<int, float>>
     */
    private function randomMatrix(int $rows, int $cols): array
    {
        $matrix = [];
        $limit = sqrt(6.0 / max(1, $rows + $cols));
        for ($i = 0; $i < $rows; $i++) {
            $row = [];
            for ($j = 0; $j < $cols; $j++) {
                $row[$j] = $this->randomFloat($limit);
            }
            $matrix[$i] = $row;
        }
        return $matrix;
    }

    private function randomFloat(float $limit): float
    {
        $value = mt_rand() / max(mt_getrandmax(), 1);
        return ($value * 2.0 - 1.0) * $limit;
    }

    /**
     * @param array<int, float> $feature
     * @param array<int, float> $target
     */
    private function trainSample(array $feature, array $target): void
    {
        $forward = $this->forwardDetailed($feature);
        $activations = $forward['activations'];
        $zs = $forward['zs'];
        $probabilities = $forward['probabilities'];
        $layerCount = count($this->layers);
        if ($layerCount === 0) {
            return;
        }

        $deltas = [];
        $outputDelta = [];
        foreach ($probabilities as $index => $prob) {
            $expected = $target[$index] ?? 0.0;
            $outputDelta[$index] = $prob - $expected;
        }
        $deltas[$layerCount - 1] = $outputDelta;

        for ($layerIndex = $layerCount - 2; $layerIndex >= 0; $layerIndex--) {
            $nextDelta = $deltas[$layerIndex + 1];
            $nextWeights = $this->layers[$layerIndex + 1]['weights'];
            $currentZ = $zs[$layerIndex];
            $derivative = $this->tanhDerivative($currentZ);
            $delta = [];
            $nodeCount = count($currentZ);
            for ($node = 0; $node < $nodeCount; $node++) {
                $sum = 0.0;
                foreach ($nextDelta as $targetIndex => $deltaValue) {
                    $sum += ($nextWeights[$targetIndex][$node] ?? 0.0) * $deltaValue;
                }
                $delta[$node] = $sum * ($derivative[$node] ?? 1.0);
            }
            $deltas[$layerIndex] = $delta;
        }

        foreach ($this->layers as $layerIndex => &$layer) {
            $delta = $deltas[$layerIndex] ?? [];
            $inputActivation = $activations[$layerIndex] ?? [];
            foreach ($layer['weights'] as $nodeIndex => &$weights) {
                $gradient = $delta[$nodeIndex] ?? 0.0;
                foreach ($weights as $weightIndex => &$weight) {
                    $weight -= $this->learningRate * $gradient * ($inputActivation[$weightIndex] ?? 0.0);
                }
                unset($weight);
                $layer['bias'][$nodeIndex] -= $this->learningRate * $gradient;
            }
            unset($weights);
        }
        unset($layer);
    }

    /**
     * @return array{activations: array<int, array<int, float>>, zs: array<int, array<int, float>>, probabilities: array<int, float>}
     */
    private function forwardDetailed(array $input): array
    {
        $activations = [array_values($input)];
        $zs = [];
        $current = array_values($input);
        $lastLayerIndex = count($this->layers) - 1;
        foreach ($this->layers as $index => $layer) {
            $z = $this->applyLayer($layer['weights'], $layer['bias'], $current);
            $zs[] = $z;
            if ($index === $lastLayerIndex) {
                $current = $z;
            } else {
                $current = $this->tanh($z);
            }
            $activations[] = $current;
        }

        $probabilities = Math::softmax($activations[count($activations) - 1]);
        return [
            'activations' => $activations,
            'zs' => $zs,
            'probabilities' => $probabilities,
        ];
    }

    /**
     * @param array<int, array<int, float>> $weights
     * @param array<int, float> $bias
     * @param array<int, float> $input
     * @return array<int, float>
     */
    private function applyLayer(array $weights, array $bias, array $input): array
    {
        $output = [];
        foreach ($weights as $rowIndex => $row) {
            $sum = $bias[$rowIndex] ?? 0.0;
            foreach ($row as $colIndex => $weight) {
                $sum += $weight * ($input[$colIndex] ?? 0.0);
            }
            $output[$rowIndex] = $sum;
        }
        return $output;
    }

    /**
     * @param array<int, float> $values
     * @return array<int, float>
     */
    private function tanh(array $values): array
    {
        return array_map(static fn (float $value): float => tanh($value), $values);
    }

    /**
     * @param array<int, float> $values
     * @return array<int, float>
     */
    private function tanhDerivative(array $values): array
    {
        return array_map(
            static function (float $value): float {
                $tanh = tanh($value);
                return 1.0 - ($tanh * $tanh);
            },
            $values
        );
    }
}
