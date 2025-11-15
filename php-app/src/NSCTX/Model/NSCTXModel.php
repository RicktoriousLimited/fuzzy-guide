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
    private const WORD_REPLACEMENTS = [
        '/\bdetected\b/i' => 'documented',
        '/\bfound\b/i' => 'identified',
        '/\bpassing\b/i' => 'sweeping past',
        '/\bnear\b/i' => 'nearby',
        '/\bbright\b/i' => 'brilliant',
        '/hydrat/i' => 'water-rich',
        '/\bcomet\b/i' => 'cometary body',
        '/\bminerals\b/i' => 'mineral deposits',
        '/\bobservatory\b/i' => 'the observatory team',
        '/\brover\b/i' => 'the rover unit',
        '/\bconfirm\w*/i' => 'verify',
        '/\bjupiter\b/i' => 'Jupiter',
        '/\bmars\b/i' => 'Mars',
    ];

    private Storage $storage;
    private TextEncoder $textEncoder;
    private NumericEncoder $imageEncoder;
    private NumericEncoder $audioEncoder;
    private CrossModalFusion $fusion;
    private SemanticGraphBuilder $graphBuilder;
    private ReasoningEngine $reasoner;
    private HubDecoder $decoder;
    private DeepNeuralNetwork $network;

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
        $networkState = is_array($saved['neural_network'] ?? null) ? $saved['neural_network'] : null;
        $this->network = new DeepNeuralNetwork($networkState);
        $this->state = $saved + [
            'trained_at' => null,
            'prototypes' => [],
            'alpha' => [],
            'metrics' => null,
            'vocabulary' => $vocabulary,
            'speaker_profiles' => $saved['speaker_profiles'] ?? [],
            'memory_bank' => $saved['memory_bank'] ?? [],
            'neural_network' => $networkState ?? $this->network->toArray(),
            'label_index' => $saved['label_index'] ?? [],
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

        $latestUserIntent = $this->extractLatestUserIntent($prepared['turns'], $prepared['summary']);
        $memory = ($best !== null && $bestScore >= 0.05) ? $best : null;
        $score = $memory !== null ? $bestScore : 0.0;

        $response = $this->composeConversationalResponse(
            $latestUserIntent,
            $memory,
            $score,
            $prepared
        );

        return [
            'response' => $response,
            'match_score' => $score,
            'memory' => $memory,
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
        $featureVectors = [];
        $labelSequence = [];
        foreach ($trainSet as $sample) {
            $encoded = $this->encodeModalities($sample['modalities']);
            $label = $sample['label'];
            $prototypes[$label][] = $encoded['hub_state'];
            $featureVectors[] = $encoded['hub_state'];
            $labelSequence[] = $label;
            foreach ($encoded['modal_weights'] as $name => $weight) {
                $alphaAccumulator[$name] = ($alphaAccumulator[$name] ?? 0.0) + $weight;
            }
        }

        $labelIndex = $this->buildLabelIndex($labelSequence);
        $this->state['label_index'] = $labelIndex;

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
        $this->trainDeepNetwork($featureVectors, $labelSequence, $labelIndex, $sampleCount);

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
        $probabilityMap = $this->classifyHubState($encoded['hub_state']);
        $prediction = array_key_first($probabilityMap) ?? 'unknown';
        return [
            'prediction' => $prediction,
            'probabilities' => $probabilityMap,
            'intermediates' => $encoded['intermediates'],
            'modal_weights' => $encoded['modal_weights'],
        ];
    }

    /**
     * @return array<string, float>
     */
    private function classifyHubState(array $hubState): array
    {
        if ($hubState === []) {
            return ['unknown' => 1.0];
        }

        $labelIndex = $this->state['label_index'] ?? [];
        if ($this->network->isReady() && $labelIndex !== []) {
            $probabilities = $this->network->predict($hubState);
            $map = [];
            foreach ($labelIndex as $label => $index) {
                $map[$label] = $probabilities[$index] ?? 0.0;
            }
            arsort($map);
            return $map;
        }

        $decoded = $this->decoder->decode($hubState, $this->state['prototypes'] ?? []);
        $probabilities = $decoded['probabilities'];
        if ($probabilities === []) {
            return ['unknown' => 1.0];
        }

        return $probabilities;
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
        $imageEncoded = $image === [] ? Vector::zeros($this->imageEncoder->getDimension()) : $this->imageEncoder->encode($image);
        $audioEncoded = $audio === [] ? Vector::zeros($this->audioEncoder->getDimension()) : $this->audioEncoder->encode($audio);

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
     * @param array<int, array<int, float>> $features
     * @param array<int, string> $labels
     * @param array<string, int> $labelIndex
     */
    private function trainDeepNetwork(array $features, array $labels, array $labelIndex, int $sampleCount): void
    {
        $featureDim = count($features[0] ?? []);
        if ($featureDim === 0 || $labelIndex === []) {
            return;
        }

        $hiddenDim = max(8, (int) ceil($featureDim * 1.5));
        $targets = $this->buildOneHotTargets($labels, $labelIndex);
        $this->network->initialize($featureDim, count($labelIndex), $hiddenDim, 2, true);
        $epochs = max(30, $sampleCount * 4);
        $this->network->train($features, $targets, $epochs);
        $this->state['neural_network'] = $this->network->toArray();
    }

    /**
     * @param array<int, string> $labels
     * @return array<string, int>
     */
    private function buildLabelIndex(array $labels): array
    {
        if ($labels === []) {
            return $this->state['label_index'] ?? [];
        }

        $unique = array_values(array_unique($labels));
        sort($unique, SORT_STRING);
        $index = [];
        foreach ($unique as $position => $label) {
            $index[$label] = $position;
        }

        return $index;
    }

    /**
     * @param array<int, string> $labels
     * @param array<string, int> $labelIndex
     * @return array<int, array<int, float>>
     */
    private function buildOneHotTargets(array $labels, array $labelIndex): array
    {
        $count = count($labelIndex);
        if ($count === 0) {
            return [];
        }

        $targets = [];
        foreach ($labels as $label) {
            $vector = array_fill(0, $count, 0.0);
            if (isset($labelIndex[$label])) {
                $vector[$labelIndex[$label]] = 1.0;
            }
            $targets[] = $vector;
        }

        return $targets;
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
        return $this->textEncoder->getDimension();
    }

    /**
     * @param array<int, array{role: string, content: string}> $turns
     */
    private function extractLatestUserIntent(array $turns, string $fallback): string
    {
        for ($index = count($turns) - 1; $index >= 0; $index--) {
            $turn = $turns[$index];
            if (strtolower($turn['role']) === 'user' && trim($turn['content']) !== '') {
                return $this->truncateIntent($turn['content']);
            }
        }

        if ($turns !== []) {
            $lastTurn = $turns[count($turns) - 1];
            if (trim($lastTurn['content']) !== '') {
                return $this->truncateIntent($lastTurn['content']);
            }
        }

        return $this->truncateIntent($fallback);
    }

    /**
     * @param array<string, mixed> $conversation
     * @param array<string, mixed>|null $memory
     */
    private function composeConversationalResponse(
        string $latestUserIntent,
        ?array $memory,
        float $score,
        array $conversation
    ): string {
        $acknowledgement = $this->craftIntentAcknowledgement($latestUserIntent);

        if ($memory === null) {
            $interpretation = $this->craftInterpretationWithoutMemory($conversation['summary']);

            return trim($acknowledgement . ' ' . $interpretation);
        }

        $narrative = $this->craftMemorySupportedNarrative($memory['summary']);
        $referenceLine = $this->buildMemoryReferenceLine($memory, $score);

        return trim($acknowledgement . ' ' . $narrative . ' ' . $referenceLine);
    }

    /**
     * @return array<int, string>
     */
    private function extractInsightsFromSummary(string $summary): array
    {
        $lines = preg_split('/\R+/', $summary) ?: [];
        $insights = [];

        foreach ($lines as $line) {
            $stripped = preg_replace('/^[a-z0-9_-]+:\s*/i', '', trim($line)) ?? '';
            if ($stripped === '') {
                continue;
            }

            $sentences = preg_split('/(?<=[.!?])\s+/', $stripped) ?: [$stripped];
            foreach ($sentences as $sentence) {
                $clean = trim($sentence);
                if ($clean === '') {
                    continue;
                }

                $insights[] = $this->ensureSentence($clean);
                if (count($insights) >= 3) {
                    break 2;
                }
            }
        }

        return $insights;
    }

    /**
     * @param array<int, string> $insights
     */
    private function buildInsightText(array $insights): string
    {
        if ($insights === []) {
            return '';
        }

        if (count($insights) === 1) {
            return $insights[0];
        }

        $last = array_pop($insights);
        $body = implode(' ', array_map('trim', $insights));

        return trim($body) . ' Also, ' . $last;
    }

    private function ensureSentence(string $text): string
    {
        $trimmed = trim($text);
        if ($trimmed === '') {
            return '';
        }

        $trimmed = rtrim($trimmed, '.!?');
        if (function_exists('mb_substr')) {
            $firstChar = mb_substr($trimmed, 0, 1);
            $rest = mb_substr($trimmed, 1);
        } else {
            $firstChar = substr($trimmed, 0, 1);
            $rest = substr($trimmed, 1);
        }

        $capitalized = strtoupper($firstChar) . $rest;

        return rtrim($capitalized) . '.';
    }

    private function craftIntentAcknowledgement(string $latestUserIntent): string
    {
        $clean = trim($latestUserIntent);
        if ($clean === '') {
            return 'I am fully focused on your request.';
        }

        if (preg_match('/\?$/', $clean) === 1) {
            return sprintf('You are asking: %s', rtrim($clean));
        }

        return 'You are asking me to: ' . $this->ensureSentence($clean);
    }

    private function craftInterpretationWithoutMemory(string $conversationSummary): string
    {
        $interpretation = $this->paraphraseSummary($conversationSummary);
        if ($interpretation === '') {
            $interpretation = 'I will reason through the idea with you.';
        }

        return $interpretation . ' Tell me more if you would like me to capture additional detail for my archive, as I do not have a saved note on this yet.';
    }

    private function craftMemorySupportedNarrative(string $memorySummary): string
    {
        $narrative = $this->paraphraseSummary($memorySummary);
        if ($narrative === '') {
            $narrative = 'I can revisit that archived topic with you in detail.';
        }

        return $narrative;
    }

    /**
     * @param array<string, mixed> $memory
     */
    private function buildMemoryReferenceLine(array $memory, float $score): string
    {
        $timestamp = $memory['timestamp'] ?? null;
        $when = 'earlier';
        if (is_string($timestamp)) {
            try {
                $date = new DateTimeImmutable($timestamp);
                $when = 'on ' . $date->format('F j, Y H:i T');
            } catch (\Exception $exception) {
                $when = 'earlier';
            }
        }

        $speakerCounts = $memory['speakers'] ?? [];
        $totalSpeakers = 0;
        foreach ($speakerCounts as $count) {
            $totalSpeakers += (int) $count;
        }

        if ($totalSpeakers <= 0) {
            $speakerText = 'a note';
        } elseif ($totalSpeakers === 1) {
            $speakerText = 'a single-speaker note';
        } else {
            $speakerText = sprintf('a %d-speaker note', $totalSpeakers);
        }

        return sprintf(
            'That perspective comes from %s I archived %s (confidence %.2f).',
            $speakerText,
            $when,
            $score
        );
    }

    private function paraphraseSummary(string $summary): string
    {
        $insights = $this->extractInsightsFromSummary($summary);
        if ($insights === []) {
            return '';
        }

        $lines = [];
        foreach ($insights as $insight) {
            $line = $this->paraphraseSentence($insight);
            if ($line !== '') {
                $lines[] = $line;
            }
            if (count($lines) >= 2) {
                break;
            }
        }

        return trim(implode(' ', $lines));
    }

    private function paraphraseSentence(string $sentence): string
    {
        $clean = trim($sentence);
        if ($clean === '') {
            return '';
        }

        $lowered = strtolower($clean);
        $lowered = $this->applyWordReplacements($lowered);
        $lowered = preg_replace('/\s+/', ' ', $lowered) ?? $lowered;

        return $this->ensureSentence($lowered);
    }

    private function applyWordReplacements(string $text): string
    {
        $result = $text;
        foreach (self::WORD_REPLACEMENTS as $pattern => $replacement) {
            $result = preg_replace($pattern, $replacement, $result) ?? $result;
        }

        return $result;
    }

    private function truncateIntent(string $text): string
    {
        $text = trim($text);
        if ($text === '') {
            return '';
        }

        $length = function_exists('mb_strlen') ? mb_strlen($text) : strlen($text);
        if ($length <= 200) {
            return $text;
        }

        $snippet = function_exists('mb_substr') ? mb_substr($text, 0, 197) : substr($text, 0, 197);
        return rtrim($snippet) . '...';
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
