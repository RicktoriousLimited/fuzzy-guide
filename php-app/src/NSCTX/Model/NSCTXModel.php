<?php

declare(strict_types=1);

namespace NSCTX\Model;

use DateTimeImmutable;
use DateTimeInterface;
use DateTimeZone;
use NSCTX\Decoder\HubDecoder;
use NSCTX\Encoder\NumericEncoder;
use NSCTX\Encoder\TextEncoder;
use NSCTX\Fusion\CrossModalFusion;
use NSCTX\Graph\SemanticGraphBuilder;
use NSCTX\Reasoning\ReasoningEngine;
use NSCTX\Support\Math;
use NSCTX\Support\Vector;

final class NSCTXModel
{
    private const MEMORY_LIMIT = 32;

    private Storage $storage;
    private TextEncoder $textEncoder;
    private NumericEncoder $imageEncoder;
    private NumericEncoder $audioEncoder;
    private CrossModalFusion $fusion;
    private SemanticGraphBuilder $graphBuilder;
    private ReasoningEngine $reasoner;
    private HubDecoder $decoder;

    /**
     * @var array<string, mixed>
     */
    private array $state;

    public function __construct(Storage $storage, int $embeddingDim = 16)
    {
        $this->storage = $storage;
        $saved = $storage->load();
        $vocabulary = $saved['vocabulary'] ?? [];
        $this->textEncoder = new TextEncoder($embeddingDim, $vocabulary);
        $this->imageEncoder = new NumericEncoder($embeddingDim, 'image');
        $this->audioEncoder = new NumericEncoder($embeddingDim, 'audio');
        $this->fusion = new CrossModalFusion();
        $this->graphBuilder = new SemanticGraphBuilder();
        $this->reasoner = new ReasoningEngine();
        $this->decoder = new HubDecoder();
        $this->state = $saved + [
            'trained_at' => null,
            'prototypes' => [],
            'alpha' => [],
            'metrics' => null,
            'vocabulary' => $vocabulary,
            'speaker_profiles' => $saved['speaker_profiles'] ?? [],
            'memory_bank' => $saved['memory_bank'] ?? [],
        ];
    }

    /**
     * Store an unlabeled passage inside the chatbot memory bank.
     *
     * @return array<string, mixed>
     */
    public function learnFromPassage(string $passage): array
    {
        $prepared = $this->prepareConversationInput($passage);
        if ($prepared['summary'] === '') {
            throw new \InvalidArgumentException('Provide a non-empty passage for learning.');
        }

        $this->textEncoder->fit([$prepared['summary']]);
        $this->state['vocabulary'] = $this->textEncoder->getVocabulary();

        $encoded = $this->textEncoder->encode($prepared['summary'], $prepared['hints']);
        $entry = [
            'id' => substr(hash('sha1', $prepared['summary'] . microtime(true)), 0, 12),
            'summary' => $prepared['summary'],
            'vector' => $encoded['contextual'],
            'turns' => $prepared['turns'],
            'speakers' => $prepared['speakers'],
            'timestamp' => (new DateTimeImmutable('now', new DateTimeZone('UTC')))->format(DateTimeInterface::ATOM),
        ];

        $memory = $this->state['memory_bank'] ?? [];
        $memory[] = $entry;
        if (count($memory) > self::MEMORY_LIMIT) {
            $memory = array_slice($memory, -self::MEMORY_LIMIT);
        }
        $this->state['memory_bank'] = $memory;
        $this->storage->save($this->state);

        return $entry;
    }

    /**
     * @param array<int, array{role: string, content: string}>|string $conversation
     * @return array{
     *     response: string,
     *     match_score: float,
     *     memory: array<string, mixed>|null,
     *     conversation: array<string, mixed>
     * }
     */
    public function chat($conversation): array
    {
        $prepared = $this->prepareConversationInput($conversation);
        if ($prepared['summary'] === '') {
            throw new \InvalidArgumentException('Conversation content missing.');
        }

        $encoded = $this->textEncoder->encode($prepared['summary'], $prepared['hints']);
        $memoryBank = $this->state['memory_bank'] ?? [];

        $best = null;
        $bestScore = 0.0;
        foreach ($memoryBank as $entry) {
            $score = Math::cosine($encoded['contextual'], $entry['vector']);
            if ($best === null || $score > $bestScore) {
                $best = $entry;
                $bestScore = $score;
            }
        }

        if ($best === null || $bestScore < 0.05) {
            return [
                'response' => 'I am still learning, but here is my interpretation: ' . $prepared['summary'],
                'match_score' => 0.0,
                'memory' => null,
                'conversation' => $prepared,
            ];
        }

        $response = sprintf(
            'Drawing from prior knowledge (score %.2f): %s',
            $bestScore,
            $best['summary']
        );

        return [
            'response' => $response,
            'match_score' => $bestScore,
            'memory' => $best,
            'conversation' => $prepared,
        ];
    }

    /**
     * @param array<int, array{label: string, modalities: array<string, mixed>}> $dataset
     * @return array{train_accuracy: float, test_accuracy: float, trained_at: string, sample_count: int}
     */
    public function train(array $dataset, float $trainRatio = 0.8, float $ewcLambda = 0.2): array
    {
        if ($dataset === []) {
            throw new \InvalidArgumentException('Training dataset is empty.');
        }

        $sampleCount = count($dataset);
        $trainCount = max(1, (int) floor($sampleCount * $trainRatio));
        $trainSet = array_slice($dataset, 0, $trainCount);
        $testSet = array_slice($dataset, $trainCount);

        $preparedTexts = array_map(function (array $item): array {
            $modalities = $item['modalities'] ?? [];
            return $this->prepareConversationInput($modalities['text'] ?? '');
        }, $trainSet);

        $this->textEncoder->fit(array_map(
            static fn (array $prepared): string => $prepared['summary'],
            $preparedTexts
        ));
        $this->state['vocabulary'] = $this->textEncoder->getVocabulary();

        $speakerProfiles = $this->state['speaker_profiles'] ?? [];
        foreach ($preparedTexts as $prepared) {
            foreach ($prepared['speakers'] as $role => $count) {
                $speakerProfiles[$role] = ($speakerProfiles[$role] ?? 0) + $count;
            }
        }
        $this->state['speaker_profiles'] = $speakerProfiles;

        $prototypes = [];
        $alphaAccumulator = [];
        foreach ($trainSet as $sample) {
            $encoded = $this->encodeModalities($sample['modalities']);
            $label = $sample['label'];
            $prototypes[$label][] = $encoded['hub_state'];
            foreach ($encoded['modal_weights'] as $name => $weight) {
                $alphaAccumulator[$name] = ($alphaAccumulator[$name] ?? 0.0) + $weight;
            }
        }

        $alpha = [];
        $modalities = array_keys($alphaAccumulator);
        if ($modalities !== []) {
            $counts = [];
            foreach ($modalities as $name) {
                $counts[] = $alphaAccumulator[$name];
            }
            $normalized = Math::softmax($counts);
            foreach ($modalities as $index => $name) {
                $alpha[$name] = $normalized[$index] ?? (1.0 / max(count($modalities), 1));
            }
        }

        $oldPrototypes = $this->state['prototypes'] ?? [];
        $finalPrototypes = [];
        foreach ($prototypes as $label => $vectors) {
            $mean = Vector::average($vectors);
            if (isset($oldPrototypes[$label])) {
                $finalPrototypes[$label] = $this->elasticCombine($oldPrototypes[$label], $mean, $ewcLambda);
            } else {
                $finalPrototypes[$label] = $mean;
            }
        }

        $timestamp = (new DateTimeImmutable('now', new DateTimeZone('UTC')))->format(DateTimeInterface::ATOM);
        $this->state['trained_at'] = $timestamp;
        $this->state['prototypes'] = $finalPrototypes;
        $this->state['alpha'] = $alpha;

        $trainAccuracy = $this->evaluate($trainSet);
        $testAccuracy = $testSet === [] ? $trainAccuracy : $this->evaluate($testSet);

        $metrics = [
            'train_accuracy' => $trainAccuracy,
            'test_accuracy' => $testAccuracy,
            'trained_at' => $timestamp,
            'sample_count' => $sampleCount,
        ];
        $this->state['metrics'] = $metrics;

        $this->storage->save($this->state);

        return $metrics;
    }

    /**
     * @param array<string, mixed> $modalities
     * @return array{
     *     prediction: string,
     *     probabilities: array<string, float>,
     *     intermediates: array<string, mixed>,
     *     modal_weights: array<string, float>
     * }
     */
    public function predict(array $modalities): array
    {
        $encoded = $this->encodeModalities($modalities, $this->state['alpha'] ?? []);
        $decoded = $this->decoder->decode($encoded['hub_state'], $this->state['prototypes'] ?? []);
        return [
            'prediction' => $decoded['prediction'],
            'probabilities' => $decoded['probabilities'],
            'intermediates' => $encoded['intermediates'],
            'modal_weights' => $encoded['modal_weights'],
        ];
    }

    /**
     * @return array<string, mixed>|null
     */
    public function getLastMetrics(): ?array
    {
        return $this->state['metrics'] ?? null;
    }

    public function getTrainedAt(): ?string
    {
        return $this->state['trained_at'] ?? null;
    }

    /**
     * @return array<string, array<int, float>>
     */
    public function getPrototypes(): array
    {
        return $this->state['prototypes'] ?? [];
    }

    /**
     * @return array<string, float>
     */
    public function getAlpha(): array
    {
        return $this->state['alpha'] ?? [];
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    public function getMemoryBank(): array
    {
        return $this->state['memory_bank'] ?? [];
    }

    /**
     * @param array<int, array{label: string, modalities: array<string, mixed>}> $dataset
     */
    private function evaluate(array $dataset): float
    {
        if ($dataset === []) {
            return 0.0;
        }
        $correct = 0;
        foreach ($dataset as $sample) {
            $prediction = $this->predict($sample['modalities']);
            if ($prediction['prediction'] === $sample['label']) {
                $correct++;
            }
        }
        return $correct / count($dataset);
    }

    /**
     * @param array<string, mixed> $modalities
     * @param array<string, float> $alpha
     * @return array{
     *     hub_state: array<int, float>,
     *     intermediates: array<string, mixed>,
     *     modal_weights: array<string, float>
     * }
     */
    private function encodeModalities(array $modalities, array $alpha = []): array
    {
        $textData = $this->prepareConversationInput($modalities['text'] ?? '');
        $text = $textData['summary'];
        $image = array_map('floatval', $modalities['image'] ?? []);
        $audio = array_map('floatval', $modalities['audio'] ?? []);

        $textEncoded = $this->textEncoder->encode($text, $textData['hints']);
        $imageEncoded = $image === [] ? Vector::zeros($this->getEmbeddingDim()) : $this->imageEncoder->encode($image);
        $audioEncoded = $audio === [] ? Vector::zeros($this->getEmbeddingDim()) : $this->audioEncoder->encode($audio);

        $modalVectors = [
            'text' => $textEncoded['contextual'],
            'image' => $imageEncoded,
            'audio' => $audioEncoded,
        ];

        $fusion = $this->fusion->fuse($modalVectors, $alpha);
        $graph = $this->graphBuilder->build($fusion['fused']);
        $reason = $this->reasoner->run($graph['nodes']);
        $hubInput = Vector::add($fusion['fused'], $reason['state']);
        $hubState = $this->decoder->relay($hubInput, [$fusion['fused'], $reason['state']]);

        return [
            'hub_state' => $hubState,
            'modal_weights' => $fusion['weights'],
            'intermediates' => [
                'text_tokens' => $textEncoded['embeddings'],
                'conversation_summary' => $textData['summary'],
                'conversation_turns' => $textData['turns'],
                'fused' => $fusion['fused'],
                'graph' => $graph,
                'reasoning_steps' => $reason['steps'],
                'hub' => $hubState,
            ],
        ];
    }

    /**
     * @param mixed $textInput
     * @return array{
     *     summary: string,
     *     hints: array<int, string>,
     *     turns: array<int, array{role: string, content: string}>,
     *     speakers: array<string, int>
     * }
     */
    private function prepareConversationInput($textInput): array
    {
        $turns = [];
        if (is_array($textInput)) {
            foreach ($textInput as $item) {
                if (!is_array($item)) {
                    continue;
                }
                $role = strtolower(trim((string) ($item['role'] ?? 'unknown')));
                $content = trim((string) ($item['content'] ?? ''));
                if ($content === '') {
                    continue;
                }
                $turns[] = [
                    'role' => $role === '' ? 'unknown' : $role,
                    'content' => $content,
                ];
            }
        }

        if ($turns === []) {
            $summary = trim((string) $textInput);
            if ($summary === '') {
                return [
                    'summary' => '',
                    'hints' => [],
                    'turns' => [],
                    'speakers' => [],
                ];
            }
            return [
                'summary' => $summary,
                'hints' => [],
                'turns' => [[
                    'role' => 'narrator',
                    'content' => $summary,
                ]],
                'speakers' => ['narrator' => 1],
            ];
        }

        $segments = [];
        $hints = [];
        $speakers = [];
        $tokenIndex = 0;
        foreach ($turns as $turn) {
            $role = $turn['role'];
            $content = $turn['content'];
            $segments[] = sprintf('%s: %s', $role, $content);
            $speakers[$role] = ($speakers[$role] ?? 0) + 1;
            $tokens = $this->basicTokenize($role . ' ' . $content);
            foreach ($tokens as $_) {
                $hints[$tokenIndex] = $role;
                $tokenIndex++;
            }
        }

        return [
            'summary' => implode("\n", $segments),
            'hints' => $hints,
            'turns' => $turns,
            'speakers' => $speakers,
        ];
    }

    /**
     * @return array<int, string>
     */
    private function basicTokenize(string $text): array
    {
        $normalized = strtolower(trim($text));
        $normalized = preg_replace('/[^a-z0-9\s]/', '', $normalized) ?? $normalized;
        $tokens = preg_split('/\s+/', $normalized) ?: [];
        return array_values(array_filter($tokens, static fn (string $token): bool => $token !== ''));
    }

    private function getEmbeddingDim(): int
    {
        return count($this->state['prototypes'][array_key_first($this->state['prototypes'])] ?? []) ?: 16;
    }

    /**
     * @param array<int, float> $old
     * @param array<int, float> $new
     */
    private function elasticCombine(array $old, array $new, float $lambda): array
    {
        if ($lambda <= 0.0) {
            return $new;
        }
        $scaledNew = Vector::scale($new, 1.0 / (1.0 + $lambda));
        $scaledOld = Vector::scale($old, $lambda / (1.0 + $lambda));
        return Vector::add($scaledNew, $scaledOld);
    }
}
