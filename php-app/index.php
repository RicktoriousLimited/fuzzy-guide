<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;

$model = new NSCTXModel(new Storage(__DIR__ . '/storage/model.json'));

$datasetPath = __DIR__ . '/data/dataset.json';
$dataset = ['samples' => []];
if (is_file($datasetPath)) {
    $raw = file_get_contents($datasetPath);
    if ($raw !== false) {
        $dataset = json_decode($raw, true, flags: JSON_THROW_ON_ERROR);
    }
}
$samples = $dataset['samples'] ?? [];

$message = null;
$error = null;
$prediction = null;
$intermediates = null;
$modalWeights = null;
$conversationInput = (string) ($_POST['text'] ?? '');
$chatInput = (string) ($_POST['chat_text'] ?? '');
$passageInput = (string) ($_POST['passage'] ?? '');
$chatResult = null;
$learnedMemory = null;

/**
 * @return array<int, array{role: string, content: string}>
 */
function parseConversationTurns(string $raw): array
{
    $lines = preg_split('/\r?\n/', trim($raw)) ?: [];
    $turns = [];
    foreach ($lines as $line) {
        $line = trim($line);
        if ($line === '') {
            continue;
        }
        if (strpos($line, ':') === false) {
            $turns[] = [
                'role' => 'narrator',
                'content' => $line,
            ];
            continue;
        }
        [$role, $content] = array_map('trim', explode(':', $line, 2));
        if ($content === '') {
            continue;
        }
        $turns[] = [
            'role' => $role === '' ? 'unknown' : strtolower($role),
            'content' => $content,
        ];
    }

    return $turns;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';
    try {
        if ($action === 'train') {
            if ($samples === []) {
                throw new RuntimeException('Dataset is empty.');
            }
            $trainRatio = isset($_POST['train_ratio']) ? (float) $_POST['train_ratio'] : 0.75;
            $ewc = isset($_POST['ewc']) ? (float) $_POST['ewc'] : 0.2;
            $metrics = $model->train($samples, $trainRatio, $ewc);
            $message = sprintf(
                'Training complete &mdash; train accuracy %.1f%%, test accuracy %.1f%%.',
                $metrics['train_accuracy'] * 100,
                $metrics['test_accuracy'] * 100
            );
        } elseif ($action === 'predict') {
            $conversationInput = trim((string) ($_POST['text'] ?? ''));
            $imageRaw = trim((string) ($_POST['image'] ?? ''));
            $audioRaw = trim((string) ($_POST['audio'] ?? ''));
            if ($conversationInput === '') {
                throw new InvalidArgumentException('Provide conversation turns for prediction.');
            }
            $turns = parseConversationTurns($conversationInput);
            if ($turns === []) {
                throw new InvalidArgumentException('Conversation format invalid. Use "role: message" per line.');
            }
            $image = $imageRaw === '' ? [] : array_map('floatval', array_filter(array_map('trim', explode(',', $imageRaw)), static fn ($value) => $value !== ''));
            $audio = $audioRaw === '' ? [] : array_map('floatval', array_filter(array_map('trim', explode(',', $audioRaw)), static fn ($value) => $value !== ''));
            $result = $model->predict([
                'text' => $turns,
                'image' => $image,
                'audio' => $audio,
            ]);
            $prediction = $result['prediction'];
            $intermediates = $result['intermediates'];
            $modalWeights = $result['modal_weights'];
            $message = 'Prediction generated.';
        } elseif ($action === 'teach') {
            $passageInput = trim((string) ($_POST['passage'] ?? ''));
            if ($passageInput === '') {
                throw new InvalidArgumentException('Provide a passage so the chatbot can learn.');
            }
            $learnedMemory = $model->learnFromPassage($passageInput);
            $message = 'Passage learned for unsupervised memory.';
        } elseif ($action === 'chat') {
            $chatInput = trim((string) ($_POST['chat_text'] ?? ''));
            if ($chatInput === '') {
                throw new InvalidArgumentException('Provide a conversation to generate a reply.');
            }
            $chatTurns = parseConversationTurns($chatInput);
            $chatPayload = $chatTurns === [] ? $chatInput : $chatTurns;
            $chatResult = $model->chat($chatPayload);
            $message = 'Chat response generated.';
        } else {
            $error = 'Unknown action.';
        }
    } catch (Throwable $exception) {
        $error = $exception->getMessage();
    }
}

$metrics = $model->getLastMetrics();
$prototypes = $model->getPrototypes();
$alpha = $model->getAlpha();
$memoryBank = $model->getMemoryBank();
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>NSCTX PHP Playground</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            color-scheme: light dark;
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background-color: #0f172a;
            color: #e2e8f0;
        }
        body {
            margin: 0;
            padding: 2rem;
            background: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.25), transparent 50%),
                        radial-gradient(circle at 90% 10%, rgba(16, 185, 129, 0.15), transparent 40%),
                        #0f172a;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            background-color: rgba(15, 23, 42, 0.85);
            border-radius: 1.5rem;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(12px);
        }
        h1, h2, h3 {
            color: #f8fafc;
            margin-top: 0;
        }
        p {
            color: #cbd5f5;
        }
        form {
            margin-bottom: 2rem;
            padding: 1.5rem;
            border-radius: 1rem;
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        input, textarea {
            width: 100%;
            border-radius: 0.75rem;
            border: 1px solid rgba(148, 163, 184, 0.4);
            padding: 0.85rem;
            font-size: 1rem;
            background-color: rgba(15, 23, 42, 0.7);
            color: #f1f5f9;
            box-sizing: border-box;
        }
        textarea {
            min-height: 6rem;
        }
        button {
            padding: 0.75rem 1.5rem;
            border-radius: 0.75rem;
            border: none;
            cursor: pointer;
            font-weight: 600;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            color: #0f172a;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 30px rgba(99, 102, 241, 0.35);
        }
        .grid {
            display: grid;
            gap: 1.5rem;
        }
        .grid-2 {
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        }
        .alert {
            padding: 1rem 1.25rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            font-weight: 600;
        }
        .alert-success {
            background: rgba(16, 185, 129, 0.18);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #bbf7d0;
        }
        .alert-error {
            background: rgba(239, 68, 68, 0.18);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #fecaca;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            color: #e2e8f0;
        }
        th, td {
            padding: 0.75rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.25);
            text-align: left;
        }
        code {
            font-family: 'Fira Code', monospace;
            background: rgba(15, 23, 42, 0.65);
            padding: 0.15rem 0.35rem;
            border-radius: 0.35rem;
        }
        .card {
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
    </style>
</head>
<body>
<div class="container">
    <h1>NSCTX Conversation Playground</h1>
    <p>Explore a conversational variant of the NSCTX reference model. Fine-tune on the bundled multi-turn dataset, understand how each modality contributes, and inspect the reasoning trace that produces the final label.</p>

    <div class="card" style="margin-bottom: 2rem;">
        <h3>How conversation predictions work</h3>
        <ul>
            <li>Provide turns as <code>role: message</code> pairs, one per line.</li>
            <li>Optionally enrich predictions with comma-separated embeddings for the image and audio modalities.</li>
            <li>Run a prediction to see the conversation summary, modal attention weights, and graph reasoning statistics.</li>
        </ul>
        <p style="margin-top: 1rem; font-size: 0.95rem; color: #94a3b8;">Tip: use consistent role names such as <em>system</em>, <em>user</em>, and <em>assistant</em> to match the dataset conventions.</p>
    </div>

    <?php if ($message !== null && $error === null): ?>
        <div class="alert alert-success"><?php echo $message; ?></div>
    <?php endif; ?>
    <?php if ($error !== null): ?>
        <div class="alert alert-error"><?php echo htmlspecialchars($error, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></div>
    <?php endif; ?>

    <section>
        <h2>Training</h2>
        <form method="post" class="grid grid-2">
            <input type="hidden" name="action" value="train">
            <div>
                <label for="train_ratio">Train split (0-1)</label>
                <input id="train_ratio" type="number" min="0.5" max="0.95" step="0.05" name="train_ratio" value="0.75">
            </div>
            <div>
                <label for="ewc">EWC λ (stability)</label>
                <input id="ewc" type="number" min="0" max="1" step="0.05" name="ewc" value="0.2">
            </div>
            <div style="grid-column: 1 / -1;">
                <button type="submit">Train model</button>
            </div>
        </form>
        <?php if ($metrics !== null): ?>
            <div class="card">
                <h3>Last training run</h3>
                <p>Samples: <strong><?php echo $metrics['sample_count']; ?></strong></p>
                <p>Train accuracy: <strong><?php echo number_format($metrics['train_accuracy'] * 100, 2); ?>%</strong></p>
                <p>Test accuracy: <strong><?php echo number_format($metrics['test_accuracy'] * 100, 2); ?>%</strong></p>
                <p>Timestamp: <strong><?php echo htmlspecialchars($metrics['trained_at'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></strong></p>
            </div>
        <?php endif; ?>
    </section>

    <section>
        <h2>Dataset overview</h2>
        <p><?php echo count($samples); ?> annotated conversations across <?php echo count(array_unique(array_map(static fn ($item) => $item['label'], $samples))); ?> intent labels.</p>
        <?php if ($samples !== []): ?>
            <table>
                <thead>
                <tr>
                    <th>#</th>
                    <th>Label</th>
                    <th>Text / Conversation</th>
                </tr>
                </thead>
                <tbody>
                <?php foreach ($samples as $index => $sample): ?>
                    <tr>
                        <td><?php echo $index + 1; ?></td>
                        <td><?php echo htmlspecialchars($sample['label'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></td>
                        <td>
                            <?php $textModality = $sample['modalities']['text'] ?? ''; ?>
                            <?php if (is_array($textModality) && $textModality !== []): ?>
                                <?php $firstTurn = $textModality[0]; ?>
                                <?php $preview = sprintf('%s: %s', $firstTurn['role'], $firstTurn['content']); ?>
                                <?php echo htmlspecialchars(substr($preview, 0, 80), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?><?php echo strlen($preview) > 80 ? '…' : ''; ?>
                            <?php else: ?>
                                <?php $textPreview = (string) $textModality; ?>
                                <?php echo htmlspecialchars(substr($textPreview, 0, 80), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?><?php echo strlen($textPreview) > 80 ? '…' : ''; ?>
                            <?php endif; ?>
                        </td>
                    </tr>
                <?php endforeach; ?>
                </tbody>
            </table>
        <?php endif; ?>
    </section>

    <section>
        <h2>Teach the chatbot</h2>
        <p>Paste any passage or transcript and NSCTX will store it in its unsupervised memory bank.</p>
        <form method="post" class="grid">
            <input type="hidden" name="action" value="teach">
            <label for="passage">Passage</label>
            <textarea id="passage" name="passage" placeholder="A log entry, article paragraph, or conversation snippet" required><?php echo htmlspecialchars($passageInput, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></textarea>
            <div>
                <button type="submit">Learn passage</button>
            </div>
        </form>
        <?php if ($learnedMemory !== null): ?>
            <div class="card">
                <h3>Latest memory</h3>
                <p><strong>Timestamp:</strong> <?php echo htmlspecialchars($learnedMemory['timestamp'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></p>
                <pre style="white-space: pre-wrap; background: rgba(15, 23, 42, 0.55); padding: 0.75rem; border-radius: 0.75rem;"><?php echo htmlspecialchars($learnedMemory['summary'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></pre>
            </div>
        <?php endif; ?>
    </section>

    <section>
        <h2>Chat with NSCTX</h2>
        <p>Supply multi-turn dialogue for a retrieval-style reply grounded in the learned passages.</p>
        <form method="post" class="grid">
            <input type="hidden" name="action" value="chat">
            <label for="chat_text">Conversation turns</label>
            <textarea id="chat_text" name="chat_text" placeholder="user: remind me what the observatory discovered&#10;assistant: ..." required><?php echo htmlspecialchars($chatInput, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></textarea>
            <div>
                <button type="submit">Generate reply</button>
            </div>
        </form>
        <?php if ($chatResult !== null): ?>
            <div class="card">
                <h3>Chat response</h3>
                <p><?php echo htmlspecialchars($chatResult['response'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></p>
                <p><strong>Match score:</strong> <?php echo number_format($chatResult['match_score'], 3); ?></p>
                <?php if ($chatResult['memory'] !== null): ?>
                    <p><strong>Matched memory:</strong></p>
                    <pre style="white-space: pre-wrap; background: rgba(15, 23, 42, 0.55); padding: 0.75rem; border-radius: 0.75rem;"><?php echo htmlspecialchars($chatResult['memory']['summary'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></pre>
                <?php endif; ?>
            </div>
        <?php endif; ?>
    </section>

    <section>
        <h2>Predict</h2>
        <form method="post" class="grid">
            <input type="hidden" name="action" value="predict">
            <label for="text">Conversation turns (one per line as <code>role: message</code>)</label>
            <textarea id="text" name="text" placeholder="system: You are the NSCTX assistant.&#10;user: describe your observation&#10;assistant: acknowledgement" required><?php echo htmlspecialchars($conversationInput, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></textarea>
            <label for="image">Image embedding (comma-separated floats)</label>
            <input id="image" type="text" name="image" placeholder="0.5, 0.2, 0.1, 0.3" value="<?php echo htmlspecialchars((string) ($_POST['image'] ?? ''), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?>">
            <label for="audio">Audio embedding (comma-separated floats)</label>
            <input id="audio" type="text" name="audio" placeholder="0.2, 0.4, 0.3" value="<?php echo htmlspecialchars((string) ($_POST['audio'] ?? ''), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?>">
            <div>
                <button type="submit">Run prediction</button>
            </div>
        </form>

        <?php if ($prediction !== null && $intermediates !== null): ?>
            <div class="card">
                <h3>Prediction result</h3>
                <p><strong>Label:</strong> <?php echo htmlspecialchars($prediction, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></p>
                <h4>Conversation summary</h4>
                <pre style="white-space: pre-wrap; background: rgba(15, 23, 42, 0.55); padding: 0.75rem; border-radius: 0.75rem;"><?php echo htmlspecialchars((string) ($intermediates['conversation_summary'] ?? ''), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></pre>
                <h4>Modal weights</h4>
                <ul>
                    <?php foreach ($modalWeights as $name => $weight): ?>
                        <li><?php echo htmlspecialchars($name, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?>: <?php echo number_format($weight, 3); ?></li>
                    <?php endforeach; ?>
                </ul>
                <h4>Reasoning trace</h4>
                <p><strong>Graph nodes:</strong> <?php echo count($intermediates['graph']['nodes']); ?> &middot; <strong>Edges:</strong> <?php echo count($intermediates['graph']['edges']); ?></p>
                <p><strong>Final hub preview:</strong> [<?php echo implode(', ', array_map(static fn ($value) => number_format($value, 3), array_slice($intermediates['hub'], 0, 5))); ?>]</p>
            </div>
        <?php endif; ?>
    </section>

    <section>
        <h2>Model introspection</h2>
        <div class="grid grid-2">
            <div class="card">
                <h3>Prototype vectors</h3>
                <?php if ($prototypes === []): ?>
                    <p>No prototypes yet. Train the model.</p>
                <?php else: ?>
                    <ul>
                        <?php foreach ($prototypes as $label => $vector): ?>
                            <li>
                                <strong><?php echo htmlspecialchars($label, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></strong>
                                &mdash; [<?php echo implode(', ', array_map(static fn ($value) => number_format($value, 3), array_slice($vector, 0, 4))); ?>]
                            </li>
                        <?php endforeach; ?>
                    </ul>
                <?php endif; ?>
            </div>
            <div class="card">
                <h3>Modal attention weights</h3>
                <?php if ($alpha === []): ?>
                    <p>Weights will appear after training.</p>
                <?php else: ?>
                    <ul>
                        <?php foreach ($alpha as $name => $weight): ?>
                            <li><?php echo htmlspecialchars($name, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?>: <?php echo number_format($weight, 3); ?></li>
                        <?php endforeach; ?>
                    </ul>
                <?php endif; ?>
            </div>
            <div class="card" style="grid-column: 1 / -1;">
                <h3>Memory bank</h3>
                <?php if ($memoryBank === []): ?>
                    <p>The chatbot has not learned any passages yet.</p>
                <?php else: ?>
                    <ul>
                        <?php foreach (array_reverse($memoryBank) as $entry): ?>
                            <li>
                                <strong><?php echo htmlspecialchars($entry['timestamp'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></strong>
                                &mdash; <?php echo htmlspecialchars(substr($entry['summary'], 0, 90), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?><?php echo strlen($entry['summary']) > 90 ? '…' : ''; ?>
                            </li>
                        <?php endforeach; ?>
                    </ul>
                <?php endif; ?>
            </div>
        </div>
    </section>
</div>
</body>
</html>
