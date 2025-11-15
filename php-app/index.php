<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use NSCTX\Model\NSCTXModel;
use NSCTX\Model\Storage;

session_start();

/**
 * @return array<int, array{role: string, content: string}>
 */
function nsctxDefaultChatHistory(): array
{
    return [[
        'role' => 'system',
        'content' => 'You are the NSCTX memory assistant. Reference stored passages and stay concise.',
    ]];
}

/**
 * @param string $role
 */
function nsctxDisplayRole(string $role): string
{
    $role = trim($role);
    if ($role === '') {
        return 'Narrator';
    }

    return ucfirst($role);
}

/**
 * @return array<int, array{label: string, modalities: array<string, mixed>}>
 */
function nsctxLoadDatasetSamples(string $path): array
{
    if (!is_file($path)) {
        return [];
    }

    $raw = file_get_contents($path);
    if ($raw === false) {
        return [];
    }

    try {
        $decoded = json_decode($raw, true, flags: JSON_THROW_ON_ERROR);
    } catch (Throwable $exception) {
        return [];
    }

    $samples = $decoded['samples'] ?? [];
    return is_array($samples) ? $samples : [];
}

$storage = new Storage(__DIR__ . '/storage/model.json');
$model = new NSCTXModel($storage);

if (!isset($_SESSION['chat_history']) || !is_array($_SESSION['chat_history'])) {
    $_SESSION['chat_history'] = nsctxDefaultChatHistory();
}

$chatHistory = $_SESSION['chat_history'];
$message = null;
$error = null;
$chatResult = null;
$latestMemory = null;
$chatInput = '';
$trainingMetrics = $model->getLastMetrics();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';
    try {
        if ($action === 'teach') {
            $passage = trim((string) ($_POST['passage'] ?? ''));
            if ($passage === '') {
                throw new InvalidArgumentException('Please provide a passage to learn.');
            }
            $latestMemory = $model->learnFromPassage($passage);
            $message = 'New passage stored in memory.';
        } elseif ($action === 'chat') {
            $chatInput = trim((string) ($_POST['message'] ?? ''));
            if ($chatInput === '') {
                throw new InvalidArgumentException('Enter a message before sending.');
            }
            $chatHistory[] = [
                'role' => 'user',
                'content' => $chatInput,
            ];
            $chatResult = $model->chat($chatHistory);
            $chatHistory[] = [
                'role' => 'assistant',
                'content' => $chatResult['response'],
            ];
            $_SESSION['chat_history'] = $chatHistory;
            $message = 'Assistant replied.';
            $chatInput = '';
        } elseif ($action === 'reset_chat') {
            $chatHistory = nsctxDefaultChatHistory();
            $_SESSION['chat_history'] = $chatHistory;
            $message = 'Conversation reset.';
        } elseif ($action === 'self_learn') {
            $baseSamples = nsctxLoadDatasetSamples(__DIR__ . '/data/dataset.json');
            $adaptSamples = nsctxLoadDatasetSamples(__DIR__ . '/data/adaptation.json');
            if ($baseSamples === [] && $adaptSamples === []) {
                throw new InvalidArgumentException('No dataset samples available for transfer learning.');
            }
            $trainingMetrics = $model->transferLearn($baseSamples, $adaptSamples, 0.35);
            $message = 'Self-learning transfer complete.';
        } else {
            $error = 'Unknown action.';
        }
    } catch (Throwable $exception) {
        $error = $exception->getMessage();
    }

    $trainingMetrics = $model->getLastMetrics();
}

$memoryBank = $model->getMemoryBank();
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>NSCTX mini ChatGPT</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            color-scheme: light dark;
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: #020617;
            color: #e2e8f0;
        }
        body {
            margin: 0;
            padding: 1.5rem;
            min-height: 100vh;
            background: radial-gradient(circle at 10% 20%, rgba(59,130,246,0.25), transparent 50%),
                        radial-gradient(circle at 80% 0%, rgba(16,185,129,0.2), transparent 35%),
                        #020617;
        }
        .shell {
            max-width: 960px;
            margin: 0 auto;
            background: rgba(2, 6, 23, 0.75);
            border-radius: 1.5rem;
            padding: 2rem;
            box-shadow: 0 35px 60px rgba(2, 6, 23, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        h1 {
            margin-top: 0;
            font-size: 2rem;
            color: #f8fafc;
        }
        p.description {
            margin-bottom: 2rem;
            color: #94a3b8;
        }
        .alert {
            border-radius: 0.75rem;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .alert-success {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #bbf7d0;
        }
        .alert-error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #fecaca;
        }
        .chat-panel {
            background: rgba(8, 25, 48, 0.75);
            border-radius: 1.25rem;
            padding: 1.5rem;
            border: 1px solid rgba(51, 65, 85, 0.6);
            margin-bottom: 2rem;
        }
        .chat-log {
            max-height: 420px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding: 1rem;
            background: rgba(2, 6, 23, 0.55);
            border-radius: 1rem;
            border: 1px solid rgba(51, 65, 85, 0.5);
        }
        .chat-bubble {
            padding: 1rem;
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            line-height: 1.5;
        }
        .chat-bubble.system {
            background: rgba(59, 130, 246, 0.1);
            align-self: center;
            font-style: italic;
        }
        .chat-bubble.user {
            background: rgba(99, 102, 241, 0.2);
            align-self: flex-end;
        }
        .chat-bubble.assistant {
            background: rgba(30, 64, 175, 0.35);
            align-self: flex-start;
        }
        .chat-role {
            font-weight: 600;
            margin-bottom: 0.35rem;
            color: #cbd5f5;
        }
        .chat-form {
            margin-top: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        textarea, input[type="text"] {
            width: 100%;
            border-radius: 0.85rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(2, 6, 23, 0.65);
            color: #e2e8f0;
            padding: 0.85rem;
            font-size: 1rem;
            box-sizing: border-box;
        }
        textarea {
            min-height: 110px;
            resize: vertical;
        }
        button {
            border: none;
            border-radius: 0.85rem;
            padding: 0.85rem 1.25rem;
            font-weight: 600;
            cursor: pointer;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            color: #020617;
            box-shadow: 0 15px 30px rgba(99, 102, 241, 0.35);
        }
        button.secondary {
            background: rgba(148, 163, 184, 0.2);
            color: #f1f5f9;
            box-shadow: none;
            border: 1px solid rgba(148, 163, 184, 0.4);
        }
        .flex {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        .panel {
            border-radius: 1.25rem;
            padding: 1.5rem;
            background: rgba(8, 25, 48, 0.8);
            border: 1px solid rgba(51, 65, 85, 0.6);
        }
        .memory-list {
            list-style: none;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .memory-list li {
            padding: 1rem;
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(2, 6, 23, 0.5);
        }
        @media (min-width: 900px) {
            .flex {
                flex-direction: row;
            }
            .flex > .panel {
                flex: 1;
            }
        }
    </style>
</head>
<body>
<div class="shell">
    <h1>NSCTX mini ChatGPT</h1>
    <p class="description">A minimal chat surface powered by the NSCTX reasoning demo. Teach new passages, then chat to see how the assistant grounds its replies in stored memories.</p>

    <?php if ($message !== null && $error === null): ?>
        <div class="alert alert-success"><?php echo htmlspecialchars($message, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></div>
    <?php endif; ?>
    <?php if ($error !== null): ?>
        <div class="alert alert-error"><?php echo htmlspecialchars($error, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></div>
    <?php endif; ?>

    <section class="chat-panel">
        <div class="chat-log">
            <?php foreach ($chatHistory as $turn): ?>
                <?php $role = $turn['role']; ?>
                <div class="chat-bubble <?php echo htmlspecialchars($role, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?>">
                    <div class="chat-role"><?php echo htmlspecialchars(nsctxDisplayRole($role), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></div>
                    <div class="chat-content"><?php echo nl2br(htmlspecialchars($turn['content'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8')); ?></div>
                </div>
            <?php endforeach; ?>
        </div>
        <form method="post" class="chat-form">
            <input type="hidden" name="action" value="chat">
            <label for="message">Message</label>
            <textarea id="message" name="message" placeholder="Ask about the passages you taught..." required><?php echo htmlspecialchars($chatInput, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></textarea>
            <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
                <button type="submit">Send</button>
            </div>
        </form>
        <div style="margin-top: 0.75rem;">
            <form method="post">
                <input type="hidden" name="action" value="reset_chat">
                <button type="submit" class="secondary">Reset conversation</button>
            </form>
        </div>
        <?php if ($chatResult !== null): ?>
            <div class="panel" style="margin-top: 1.5rem;">
                <h3>Latest reply</h3>
                <p><strong>Response:</strong> <?php echo htmlspecialchars($chatResult['response'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></p>
                <p><strong>Match score:</strong> <?php echo number_format($chatResult['match_score'], 3); ?></p>
                <?php if ($chatResult['memory'] !== null): ?>
                    <p><strong>Matched memory:</strong></p>
                    <pre style="white-space: pre-wrap; background: rgba(2, 6, 23, 0.45); padding: 0.85rem; border-radius: 0.85rem; border: 1px solid rgba(148, 163, 184, 0.2);">
<?php echo htmlspecialchars($chatResult['memory']['summary'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></pre>
                <?php else: ?>
                    <p>The assistant answered without a confident memory match.</p>
                <?php endif; ?>
            </div>
        <?php endif; ?>
    </section>

    <div class="flex">
        <section class="panel">
            <h2>Teach the assistant</h2>
            <p>Paste any passage, log entry, or short transcript. The summary embedding fuels retrieval for future chats.</p>
            <form method="post">
                <input type="hidden" name="action" value="teach">
                <label for="passage">Passage</label>
                <textarea id="passage" name="passage" placeholder="The observatory detected a comet..." required></textarea>
                <div style="margin-top: 1rem;">
                    <button type="submit">Store passage</button>
                </div>
            </form>
            <?php if ($latestMemory !== null): ?>
                <div style="margin-top: 1rem;">
                    <h3>Latest memory</h3>
                    <p><strong>Timestamp:</strong> <?php echo htmlspecialchars($latestMemory['timestamp'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></p>
                    <pre style="white-space: pre-wrap; background: rgba(2, 6, 23, 0.45); padding: 0.85rem; border-radius: 0.85rem; border: 1px solid rgba(148, 163, 184, 0.2);">
<?php echo htmlspecialchars($latestMemory['summary'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></pre>
                </div>
            <?php endif; ?>
        </section>

        <section class="panel">
            <h2>Memory bank</h2>
            <?php if ($memoryBank === []): ?>
                <p>No memories stored yet. Use the teaching form to add context.</p>
            <?php else: ?>
                <ul class="memory-list">
                    <?php foreach (array_reverse($memoryBank) as $entry): ?>
                        <li>
                            <strong><?php echo htmlspecialchars($entry['timestamp'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></strong>
                            <div><?php echo htmlspecialchars(substr($entry['summary'], 0, 160), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?><?php echo strlen($entry['summary']) > 160 ? '…' : ''; ?></div>
                        </li>
                    <?php endforeach; ?>
                </ul>
            <?php endif; ?>
        </section>
    </div>

    <section class="panel" style="margin-top: 1.5rem;">
        <h2>Self-learning transfer</h2>
        <p>The NSCTX model can rehearse the base mission dataset while adapting to a new scenario. Triggering self-learning
            mixes the original samples with fresh adaptation data, runs continual training, and reports how well prior skills
            were retained.</p>
        <?php if ($trainingMetrics !== null): ?>
            <ul style="list-style: none; padding: 0; margin: 1rem 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem;">
                <li><strong>Mode:</strong> <?php echo htmlspecialchars($trainingMetrics['mode'] ?? 'standard', ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></li>
                <li><strong>Samples:</strong> <?php echo (int) ($trainingMetrics['sample_count'] ?? 0); ?></li>
                <li><strong>Train accuracy:</strong> <?php echo number_format(($trainingMetrics['train_accuracy'] ?? 0.0) * 100, 2); ?>%</li>
                <li><strong>Test accuracy:</strong> <?php echo number_format(($trainingMetrics['test_accuracy'] ?? 0.0) * 100, 2); ?>%</li>
                <?php if (isset($trainingMetrics['base_retention'])): ?>
                    <li><strong>Base retention:</strong> <?php echo $trainingMetrics['base_retention'] === null ? 'n/a' : number_format($trainingMetrics['base_retention'] * 100, 2) . '%'; ?></li>
                <?php endif; ?>
                <?php if (isset($trainingMetrics['adapt_performance'])): ?>
                    <li><strong>Adaptation:</strong> <?php echo $trainingMetrics['adapt_performance'] === null ? 'n/a' : number_format($trainingMetrics['adapt_performance'] * 100, 2) . '%'; ?></li>
                <?php endif; ?>
                <?php if (isset($trainingMetrics['carryover_samples'])): ?>
                    <li><strong>Carryover samples:</strong> <?php echo (int) $trainingMetrics['carryover_samples']; ?></li>
                <?php endif; ?>
                <li><strong>Updated:</strong> <?php echo htmlspecialchars((string) ($trainingMetrics['trained_at'] ?? 'n/a'), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></li>
            </ul>
        <?php else: ?>
            <p>No training run recorded yet. Use the button below to perform the first self-learning pass.</p>
        <?php endif; ?>
        <form method="post" style="margin-top: 1rem;">
            <input type="hidden" name="action" value="self_learn">
            <button type="submit">Run self-learning transfer</button>
        </form>
    </section>
</div>
</body>
</html>
