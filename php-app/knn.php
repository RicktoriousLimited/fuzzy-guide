<?php

declare(strict_types=1);

/**
 * Lightweight K-Nearest Neighbors implementation backed by JSON storage.
 */
final class KNNModel
{
    private string $modelPath;
    private string $defaultDatasetPath;

    /**
     * @var array<int, array{features: array<int, float>, label: string}>
     */
    private array $samples = [];
    private int $k = 3;
    private ?string $trainedAt = null;

    public function __construct(string $modelPath, string $defaultDatasetPath)
    {
        $this->modelPath = $modelPath;
        $this->defaultDatasetPath = $defaultDatasetPath;
        $this->load();
    }

    public function getK(): int
    {
        return $this->k;
    }

    public function getTrainedAt(): ?string
    {
        return $this->trainedAt;
    }

    /**
     * @return array<int, array{features: array<int, float>, label: string}>
     */
    public function getSamples(): array
    {
        return $this->samples;
    }

    public function setK(int $k): void
    {
        if ($k < 1) {
            throw new InvalidArgumentException('k must be at least 1.');
        }
        $this->k = $k;
        $this->save();
    }

    /**
     * @param array<int, float> $features
     */
    public function addSample(array $features, string $label): void
    {
        if ($label === '') {
            throw new InvalidArgumentException('Label cannot be empty.');
        }

        if ($this->samples !== [] && count($features) !== $this->getFeatureCount()) {
            throw new InvalidArgumentException(sprintf(
                'Expected %d features, received %d.',
                $this->getFeatureCount(),
                count($features)
            ));
        }

        $this->samples[] = [
            'features' => array_values($features),
            'label' => $label,
        ];
        $this->trainedAt = null;
        $this->save();
    }

    public function clearSamples(): void
    {
        $this->samples = [];
        $this->trainedAt = null;
        $this->save();
    }

    public function resetToDefault(): void
    {
        $this->loadDefaultDataset();
        $this->trainedAt = null;
        $this->save();
    }

    public function train(): string
    {
        if ($this->samples === []) {
            throw new RuntimeException('Cannot train on an empty dataset. Add samples first.');
        }

        $this->trainedAt = (new DateTimeImmutable('now', new DateTimeZone('UTC')))->format(DateTimeInterface::ATOM);
        $this->save();
        return $this->trainedAt;
    }

    /**
     * @param array<int, float> $features
     * @return array{
     *     prediction: string,
     *     probabilities: array<string, float>,
     *     neighbors: array<int, array{distance: float, label: string}>
     * }
     */
    public function predict(array $features): array
    {
        if ($this->samples === []) {
            throw new RuntimeException('No samples available. Train the model before predicting.');
        }

        if (count($features) !== $this->getFeatureCount()) {
            throw new InvalidArgumentException(sprintf(
                'Expected %d features, received %d.',
                $this->getFeatureCount(),
                count($features)
            ));
        }

        $neighbors = [];
        foreach ($this->samples as $sample) {
            $neighbors[] = [
                'distance' => $this->euclideanDistance($features, $sample['features']),
                'label' => $sample['label'],
            ];
        }

        usort($neighbors, static fn (array $a, array $b): int => $a['distance'] <=> $b['distance']);
        $k = min($this->k, count($neighbors));

        $votes = [];
        for ($i = 0; $i < $k; $i++) {
            $label = $neighbors[$i]['label'];
            $votes[$label] = ($votes[$label] ?? 0) + 1;
        }

        arsort($votes);

        $probabilities = [];
        foreach ($votes as $label => $count) {
            $probabilities[$label] = $count / $k;
        }

        return [
            'prediction' => (string) array_key_first($votes),
            'probabilities' => $probabilities,
            'neighbors' => array_slice($neighbors, 0, $k),
        ];
    }

    private function load(): void
    {
        if (is_file($this->modelPath)) {
            $raw = file_get_contents($this->modelPath);
            if ($raw === false) {
                throw new RuntimeException('Failed to read model storage.');
            }
            $decoded = json_decode($raw, true, flags: JSON_THROW_ON_ERROR);
        } else {
            $decoded = $this->loadDefaultDataset();
            $this->save();
        }

        $this->k = isset($decoded['k']) ? (int) $decoded['k'] : 3;
        $this->samples = array_map(
            static fn (array $item): array => [
                'features' => array_map('floatval', $item['features']),
                'label' => (string) $item['label'],
            ],
            $decoded['samples'] ?? []
        );
        $this->trainedAt = isset($decoded['trained_at']) ? (string) $decoded['trained_at'] : null;
    }

    /**
     * @return array{k: int, samples: array<int, array{features: array<int, float>, label: string}>, trained_at?: string|null}
     */
    private function loadDefaultDataset(): array
    {
        $raw = file_get_contents($this->defaultDatasetPath);
        if ($raw === false) {
            throw new RuntimeException('Failed to read default dataset.');
        }
        $decoded = json_decode($raw, true, flags: JSON_THROW_ON_ERROR);
        $this->k = isset($decoded['k']) ? (int) $decoded['k'] : $this->k;
        $this->samples = array_map(
            static fn (array $item): array => [
                'features' => array_map('floatval', $item['features']),
                'label' => (string) $item['label'],
            ],
            $decoded['samples'] ?? []
        );
        return [
            'k' => $this->k,
            'samples' => $this->samples,
            'trained_at' => $this->trainedAt,
        ];
    }

    private function save(): void
    {
        $this->ensureStorageDirectory();
        $payload = [
            'k' => $this->k,
            'samples' => $this->samples,
            'trained_at' => $this->trainedAt,
        ];
        $encoded = json_encode($payload, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR);
        if (file_put_contents($this->modelPath, $encoded) === false) {
            throw new RuntimeException('Failed to write model storage.');
        }
    }

    private function ensureStorageDirectory(): void
    {
        $directory = dirname($this->modelPath);
        if (!is_dir($directory) && !mkdir($directory, 0775, true) && !is_dir($directory)) {
            throw new RuntimeException(sprintf('Unable to create storage directory: %s', $directory));
        }
    }

    /**
     * @param array<int, float> $a
     * @param array<int, float> $b
     */
    private function euclideanDistance(array $a, array $b): float
    {
        $distance = 0.0;
        $count = count($a);
        for ($i = 0; $i < $count; $i++) {
            $diff = $a[$i] - $b[$i];
            $distance += $diff * $diff;
        }
        return sqrt($distance);
    }

    private function getFeatureCount(): int
    {
        return $this->samples === [] ? 0 : count($this->samples[0]['features']);
    }
}
