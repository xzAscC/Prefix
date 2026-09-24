from prefix.paper_data import prepare_task, BASELINE_PROMPTS


def loader(dataset, **kwargs):
    if dataset == 'stanfordnlp/sst2':
        split = kwargs['split']
        return [{'sentence': f'{split}-{i}', 'label': i % 2} for i in range(20)]
    if dataset == 'Intel/polite-guard':
        split = kwargs['split']
        labels = ['polite', 'impolite', 'neutral', 'somewhat polite']
        return [{'text': f'{split}-{i}', 'label': labels[i % 4]} for i in range(40)]
    raise AssertionError(dataset)


def test_sentiment_tuning_does_not_consume_evaluation_or_direction_examples():
    data = prepare_task('sentiment', n_direction=3, n_tune=4, loader=loader)
    assert len(data['positive']) == len(data['negative']) == 3
    assert len(data['tune']) == 4 and len(data['test']) == 10
    used = {item['prompt'] for item in data['tune']}
    assert not used.intersection(data['positive'] + data['negative'])
    assert all(item['prompt'].startswith('validation-') for item in data['test'])


def test_politeness_retains_only_binary_labels_and_evaluates_impolite_inputs():
    data = prepare_task('politeness', n_direction=3, n_tune=4, loader=loader)
    assert all(int(text.split('-')[-1]) % 4 == 0 for text in data['positive'])
    assert all(int(text.split('-')[-1]) % 4 == 1 for text in data['negative'])
    assert all(item['prompt'].startswith('test-') and int(item['prompt'].split('-')[-1]) % 4 == 1 for item in data['test'])


def test_each_math_direction_compares_its_instruction_against_neutral(monkeypatch):
    import prefix.paper_data as module
    monkeypatch.setattr(module, 'load_math500', lambda: [dict(problem=f'p{i}', answer=str(i)) for i in range(500)])
    boxed = prepare_task('boxed')
    plain = prepare_task('plain')
    assert len(boxed['positive']) == 50
    assert len(boxed['tune']) == 50 and len(boxed['test']) == 400
    assert boxed['negative'] == plain['negative']
    assert all('boxed' not in text and 'The answer is' not in text for text in boxed['negative'])
    assert all('boxed' in text for text in boxed['positive'])
    assert all('boxed' not in text and 'The answer is' in text for text in plain['positive'])
    assert BASELINE_PROMPTS['sentiment'] == 'Respond with clearly positive sentiment.'


def test_safety_preserves_existing_harmbench_schema(monkeypatch):
    import prefix.paper_data as module
    monkeypatch.setattr(module, 'load_llm_lat', lambda dataset, n: [f'{dataset}-{i}' for i in range(n)])
    monkeypatch.setattr(module, 'load_harmbench', lambda: [{'behavior': f'b{i}', 'category': 'c'} for i in range(400)])
    data = prepare_task('safety')
    assert len(data['tune']) == 50 and len(data['test']) == 350
    assert not {row['id'] for row in data['tune']} & {row['id'] for row in data['test']}
