<?php

declare(strict_types=1);

require __DIR__ . '/knn.php';

$model = new KNNModel(
    __DIR__ . '/storage/model.json',
    __DIR__ . '/sample_dataset.json'
);

$message = null;
$error = null;
$prediction = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';
    try {
        switch ($action) {
            case 'add-sample':
                $rawFeatures = trim((string) ($_POST['features'] ?? ''));
                $label = trim((string) ($_POST['label'] ?? ''));
                if ($rawFeatures === '') {
                    throw new InvalidArgumentException('Provide at least one feature.');
                }
                $features = array_values(array_filter(array_map('trim', explode(',', $rawFeatures)), static fn ($value) => $value !== ''));
                $numericFeatures = array_map(
                    static function (string $value): float {
                        if (!is_numeric($value)) {
                            throw new InvalidArgumentException('All features must be numeric.');
                        }
                        return (float) $value;
                    },
                    $features
                );
                $model->addSample($numericFeatures, $label);
                $message = 'Sample added successfully. Remember to retrain to update the timestamp.';
                break;
            case 'update-k':
                $k = (int) ($_POST['k'] ?? 3);
                $model->setK($k);
                $message = sprintf('Updated k to %d.', $k);
                break;
            case 'clear-dataset':
                $model->clearSamples();
                $message = 'Dataset cleared.';
                break;
            case 'reset-defaults':
                $model->resetToDefault();
                $message = 'Dataset restored to bundled defaults.';
                break;
            case 'train-model':
                $timestamp = $model->train();
                $message = sprintf('Model trained at %s.', $timestamp);
                break;
            case 'predict':
                $rawFeatures = trim((string) ($_POST['predict-features'] ?? ''));
                if ($rawFeatures === '') {
                    throw new InvalidArgumentException('Enter features to run a prediction.');
                }
                $features = array_values(array_filter(array_map('trim', explode(',', $rawFeatures)), static fn ($value) => $value !== ''));
                $numericFeatures = array_map(
                    static function (string $value): float {
                        if (!is_numeric($value)) {
                            throw new InvalidArgumentException('All features must be numeric.');
                        }
                        return (float) $value;
                    },
                    $features
                );
                $prediction = $model->predict($numericFeatures);
                $message = 'Prediction complete.';
                break;
            default:
                $error = 'Unknown action requested.';
        }
    } catch (Throwable $exception) {
        $error = $exception->getMessage();
    }
}

$samples = $model->getSamples();
$featureCount = $samples === [] ? 0 : count($samples[0]['features']);
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>PHP KNN Playground</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            color-scheme: light dark;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
        }
        body {
            margin: 0;
            padding: 2rem;
            background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%);
        }
        .container {
            max-width: 960px;
            margin: 0 auto;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.12);
        }
        h1, h2 {
            color: #1e293b;
            margin-top: 0;
        }
        form {
            margin-bottom: 2rem;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            background-color: #fff;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid #cbd5f5;
            font-size: 1rem;
            box-sizing: border-box;
        }
        button {
            padding: 0.65rem 1.25rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
            font-weight: 600;
            background-color: #2563eb;
            color: #fff;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
        }
        .button-secondary {
            background-color: #64748b;
        }
        .stack {
            display: grid;
            gap: 1rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.95rem;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        .alert {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            font-weight: 600;
        }
        .alert-success {
            background-color: #dcfce7;
            color: #14532d;
        }
        .alert-error {
            background-color: #fee2e2;
            color: #7f1d1d;
        }
        code {
            font-family: "Fira Code", "Source Code Pro", monospace;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>PHP KNN Playground</h1>
    <p>Manage a lightweight k-Nearest Neighbors classifier entirely in PHP. Add training samples, configure the value of <code>k</code>, train to timestamp your configuration, and run predictions directly from this page.</p>

    <?php if ($message !== null): ?>
        <div class="alert alert-success"><?php echo htmlspecialchars($message, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></div>
    <?php endif; ?>
    <?php if ($error !== null): ?>
        <div class="alert alert-error"><?php echo htmlspecialchars($error, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></div>
    <?php endif; ?>

    <section>
        <h2>Model Configuration</h2>
        <form method="post" class="stack">
            <input type="hidden" name="action" value="update-k">
            <label for="k">Number of neighbors (k)</label>
            <input id="k" type="number" name="k" min="1" required value="<?php echo htmlspecialchars((string) $model->getK(), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?>">
            <button type="submit">Update k</button>
        </form>
        <form method="post" class="stack" style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
            <input type="hidden" name="action" value="train-model">
            <button type="submit">Train model</button>
            <span>Last trained: <strong><?php echo htmlspecialchars($model->getTrainedAt() ?? 'Not trained yet', ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></strong></span>
        </form>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
            <form method="post">
                <input type="hidden" name="action" value="clear-dataset">
                <button type="submit" class="button-secondary">Clear dataset</button>
            </form>
            <form method="post">
                <input type="hidden" name="action" value="reset-defaults">
                <button type="submit" class="button-secondary">Restore defaults</button>
            </form>
        </div>
    </section>

    <section>
        <h2>Add Training Sample</h2>
        <form method="post" class="stack">
            <input type="hidden" name="action" value="add-sample">
            <label for="features">Features (comma-separated numeric values)</label>
            <input id="features" type="text" name="features" placeholder="e.g. 5.1, 3.5, 1.4, 0.2" required>
            <label for="label">Label</label>
            <input id="label" type="text" name="label" placeholder="setosa" required>
            <button type="submit">Add sample</button>
        </form>
    </section>

    <section>
        <h2>Current Dataset</h2>
        <p><?php echo count($samples); ?> samples &middot; <?php echo $featureCount; ?> features per sample</p>
        <?php if ($samples === []): ?>
            <p>No data yet. Add samples or restore the defaults.</p>
        <?php else: ?>
            <table>
                <thead>
                <tr>
                    <th>#</th>
                    <th>Features</th>
                    <th>Label</th>
                </tr>
                </thead>
                <tbody>
                <?php foreach ($samples as $index => $sample): ?>
                    <tr>
                        <td><?php echo $index + 1; ?></td>
                        <td><?php echo htmlspecialchars(implode(', ', array_map(static fn ($value) => number_format((float) $value, 3, '.', ''), $sample['features'])), ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></td>
                        <td><?php echo htmlspecialchars($sample['label'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></td>
                    </tr>
                <?php endforeach; ?>
                </tbody>
            </table>
        <?php endif; ?>
    </section>

    <section>
        <h2>Run Prediction</h2>
        <form method="post" class="stack">
            <input type="hidden" name="action" value="predict">
            <label for="predict-features">Features (comma-separated numeric values)</label>
            <input id="predict-features" type="text" name="predict-features" placeholder="e.g. 5.0, 3.4, 1.5, 0.2" required>
            <button type="submit">Predict</button>
        </form>

        <?php if ($prediction !== null): ?>
            <div style="margin-top: 1.5rem;">
                <h3>Prediction Result</h3>
                <p><strong>Label:</strong> <?php echo htmlspecialchars($prediction['prediction'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></p>
                <h4>Class Probabilities</h4>
                <ul>
                    <?php foreach ($prediction['probabilities'] as $label => $probability): ?>
                        <li><?php echo htmlspecialchars($label, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?> &mdash; <?php echo number_format($probability * 100, 2); ?>%</li>
                    <?php endforeach; ?>
                </ul>
                <h4>Nearest Neighbors</h4>
                <table>
                    <thead>
                    <tr>
                        <th>#</th>
                        <th>Label</th>
                        <th>Distance</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($prediction['neighbors'] as $index => $neighbor): ?>
                        <tr>
                            <td><?php echo $index + 1; ?></td>
                            <td><?php echo htmlspecialchars($neighbor['label'], ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8'); ?></td>
                            <td><?php echo number_format($neighbor['distance'], 4); ?></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        <?php endif; ?>
    </section>
</div>
</body>
</html>
