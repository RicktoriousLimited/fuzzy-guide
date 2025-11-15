<?php

declare(strict_types=1);

namespace NSCTX\Encoder;

use NSCTX\Support\Hasher;
use NSCTX\Support\Math;
use NSCTX\Support\Vector;

final class TextEncoder
{
    private int $embeddingDim;

    /**
     * @var array<string, int>
     */
    private array $vocabulary;

    public function __construct(int $embeddingDim, array $vocabulary = [])
    {
        $this->embeddingDim = $embeddingDim;
        $this->vocabulary = $vocabulary;
    }

    public function getDimension(): int
    {
        return $this->embeddingDim;
    }

    /**
     * @param array<int, string> $texts
     */
    public function fit(array $texts): void
    {
        foreach ($texts as $text) {
            $tokens = $this->tokenize($text);
            foreach ($tokens as $token) {
                $this->vocabulary[$token] = ($this->vocabulary[$token] ?? 0) + 1;
            }
        }
    }

    /**
     * @return array<string, int>
     */
    public function getVocabulary(): array
    {
        return $this->vocabulary;
    }

    /**
     * @param array<int, string> $speakerHints
     * @return array{
     *     embeddings: array<int, array<int, float>>,
     *     contextual: array<int, float>
     * }
     */
    public function encode(string $text, array $speakerHints = []): array
    {
        $tokens = $this->tokenize($text);
        if ($tokens === []) {
            return [
                'embeddings' => [],
                'contextual' => array_fill(0, $this->embeddingDim, 0.0),
            ];
        }

        $tokenEmbeddings = [];
        foreach ($tokens as $index => $token) {
            $base = $this->tokenEmbedding($token);
            $position = $this->positionEmbedding($index);
            $role = $this->roleEmbedding($token, $index, $speakerHints[$index] ?? null);
            $embedding = Vector::add(Vector::add($base, $position), $role);
            $tokenEmbeddings[] = $embedding;
        }

        $contextual = $this->selfAttention($tokenEmbeddings);
        $contextual = Math::layerNorm(Math::gelu($contextual));

        return [
            'embeddings' => $tokenEmbeddings,
            'contextual' => $contextual,
        ];
    }

    /**
     * @return array<int, string>
     */
    private function tokenize(string $text): array
    {
        $normalized = strtolower(trim($text));
        $normalized = preg_replace('/[^a-z0-9\s]/', '', $normalized) ?? $normalized;
        $tokens = preg_split('/\s+/', $normalized) ?: [];
        return array_values(array_filter($tokens, static fn (string $token): bool => $token !== ''));
    }

    /**
     * @return array<int, float>
     */
    private function tokenEmbedding(string $token): array
    {
        $vector = [];
        for ($i = 0; $i < $this->embeddingDim; $i++) {
            $vector[$i] = Hasher::float('token:' . $token . ':' . $i, 0.7);
        }
        return $vector;
    }

    /**
     * @return array<int, float>
     */
    private function positionEmbedding(int $position): array
    {
        $vector = [];
        for ($i = 0; $i < $this->embeddingDim; $i++) {
            $vector[$i] = Hasher::float('position:' . $position . ':' . $i, 0.3);
        }
        return $vector;
    }

    /**
     * @return array<int, float>
     */
    private function roleEmbedding(string $token, int $index, ?string $speakerHint = null): array
    {
        $role = $this->inferRole($token, $index);
        $vector = [];
        for ($i = 0; $i < $this->embeddingDim; $i++) {
            $vector[$i] = Hasher::float('role:' . $role . ':' . $i, 0.5);
        }
        if ($speakerHint !== null) {
            $speakerVector = [];
            for ($i = 0; $i < $this->embeddingDim; $i++) {
                $speakerVector[$i] = Hasher::float('speaker:' . $speakerHint . ':' . $i, 0.45);
            }
            $vector = Vector::add($vector, $speakerVector);
        }
        return $vector;
    }

    private function inferRole(string $token, int $index): string
    {
        if ($index === 0) {
            return 'subject';
        }
        if (str_ends_with($token, 'ed') || str_ends_with($token, 'ing')) {
            return 'predicate';
        }
        if (in_array($token, ['to', 'with', 'for', 'into'], true)) {
            return 'relation';
        }
        if (isset($this->vocabulary[$token]) && $this->vocabulary[$token] > 2) {
            return 'entity';
        }
        return 'unk';
    }

    /**
     * @param array<int, array<int, float>> $tokenEmbeddings
     * @return array<int, float>
     */
    private function selfAttention(array $tokenEmbeddings): array
    {
        $heads = 2;
        $dimension = $this->embeddingDim;
        $headDim = (int) ($dimension / $heads);
        $outputs = array_fill(0, $heads, array_fill(0, $headDim, 0.0));
        foreach ($tokenEmbeddings as $tokenIndex => $embedding) {
            for ($head = 0; $head < $heads; $head++) {
                $slice = array_slice($embedding, $head * $headDim, $headDim);
                $projected = Hasher::project($slice, $headDim, 'attn:' . $head . ':' . $tokenIndex);
                for ($i = 0; $i < $headDim; $i++) {
                    $outputs[$head][$i] += $projected[$i];
                }
            }
        }
        $combined = [];
        for ($head = 0; $head < $heads; $head++) {
            $normalized = Math::normalize($outputs[$head]);
            $combined = array_merge($combined, $normalized);
        }
        return $combined;
    }
}
