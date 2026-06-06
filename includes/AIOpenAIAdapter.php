<?php

/**
 * @file
 * OpenAI adapter for AI core.
 */

class AIOpenAIAdapter extends AIAdapterBase {

  /**
   * OpenAI API base URL.
   *
   * @var string
   */
  protected $baseUrl = 'https://api.openai.com/v1';

  /**
   * {@inheritdoc}
   */
  protected function getDefaultHeaders(): array {
    return [
      'Authorization' => 'Bearer ' . $this->apiKey,
    ];
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    $models = [];

    try {
      $result = $this->makeRequest($this->baseUrl . '/models', [], [], 'GET', 10);
      foreach ($result['data'] ?? [] as $model) {
        $id = $model['id'] ?? NULL;
        if (empty($id)) {
          continue;
        }
        if (!preg_match('/^(gpt|text|tts|whisper|dall-e|gpt-image|o[1-9]|.*moderation)/i', $id)) {
          continue;
        }
        if (preg_match('/(search|similarity|edit|instruct)/i', $id)) {
          continue;
        }
        $models[$id] = $model['id'];
      }
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'Failed to fetch OpenAI models: @message', ['@message' => $e->getMessage()], WATCHDOG_WARNING);
    }

    if (!empty($models)) {
      asort($models);
    }

    return $models;
  }

  /**
   * {@inheritdoc}
   */
  public function getModelsByCapability($capability): array {
    $models = $this->getModels();
    switch ($capability) {
      case 'text':
      case 'chat':
        $filtered = array_filter($models, function ($id) {
          return preg_match('/^(gpt|o[1-9])/i', $id);
        }, ARRAY_FILTER_USE_KEY);
        break;

      case 'embeddings':
      case 'embedding':
        $filtered = array_filter($models, function ($id) {
          return strpos($id, 'text-embedding-') === 0;
        }, ARRAY_FILTER_USE_KEY);
        break;

      case 'image':
        $filtered = array_filter($models, function ($id) {
          return strpos($id, 'dall-e') === 0 || strpos($id, 'gpt-image') === 0;
        }, ARRAY_FILTER_USE_KEY);
        break;

      case 'tts':
        $filtered = array_filter($models, function ($id) {
          return strpos($id, 'tts-') === 0;
        }, ARRAY_FILTER_USE_KEY);
        break;

      case 'stt':
        $filtered = array_filter($models, function ($id) {
          return strpos($id, 'whisper-') === 0;
        }, ARRAY_FILTER_USE_KEY);
        break;

      case 'moderation':
        $filtered = array_filter($models, function ($id) {
          return strpos($id, 'text-moderation-') === 0 || strpos($id, 'omni-moderation-') === 0;
        }, ARRAY_FILTER_USE_KEY);
        break;

      default:
        $filtered = $models;
        break;
    }

    backdrop_alter('ai_model_capabilities', $filtered, $capability, $this);
    return $filtered;
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    try {
      $payload = [
        'model' => $model,
        'prompt' => trim($prompt),
        'temperature' => (float) $temperature,
      ];
      if ((int) $max_tokens > 0) {
        $payload['max_tokens'] = (int) $max_tokens;
      }

      if ($stream_response) {
        return $this->buildStreamingResponse($this->baseUrl . '/completions', [
          'method' => 'POST',
          'headers' => array_merge([
            'Accept' => 'text/event-stream',
            'Content-Type' => 'application/json',
          ], $this->getDefaultHeaders()),
          'data' => json_encode($payload),
          'timeout' => 300,
        ], function ($data) {
          return $data['choices'][0]['delta']['content'] ?? $data['choices'][0]['text'] ?? '';
        });
      }

      $result = $this->makeRequest($this->baseUrl . '/completions', $payload);
      return trim($result['choices'][0]['text'] ?? '');
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'Completions error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE, array $context_extra = []) {
    try {
      if ($this->modelUsesResponsesApi($model)) {
        return $this->chatWithResponsesApi($model, $messages, $temperature, $max_tokens);
      }

      $payload = [
        'model' => $model,
        'messages' => $messages,
        'temperature' => (float) $temperature,
      ];
      if ((int) $max_tokens > 0) {
        $payload['max_tokens'] = (int) $max_tokens;
      }

      if ($stream_response) {
        return $this->buildStreamingResponse($this->baseUrl . '/chat/completions', [
          'method' => 'POST',
          'headers' => array_merge([
            'Accept' => 'text/event-stream',
            'Content-Type' => 'application/json',
          ], $this->getDefaultHeaders()),
          'data' => json_encode($payload),
          'timeout' => 300,
        ], function ($data) {
          return $data['choices'][0]['delta']['content'] ?? '';
        });
      }

      $result = $this->makeRequest($this->baseUrl . '/chat/completions', $payload);
      return trim($result['choices'][0]['message']['content'] ?? $result['choices'][0]['text'] ?? '');
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'Chat error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    try {
      $payload = [
        'model' => $model,
        'prompt' => $prompt,
        'size' => $size,
        'response_format' => $response_format,
      ];
      if ($model === 'dall-e-3') {
        $payload['quality'] = $quality;
        $payload['style'] = $style;
      }
      if (!empty($output_format)) {
        $payload['output_format'] = $output_format;
      }

      return $this->makeRequest($this->baseUrl . '/images/generations', $payload);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'Images error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    try {
      return $this->requestRaw($this->baseUrl . '/audio/speech', [
        'model' => $model,
        'input' => $input,
        'voice' => $voice,
        'response_format' => $response_format,
      ]);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'TTS error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    if (!in_array($task, ['transcribe', 'translate'], TRUE)) {
      throw new \InvalidArgumentException('Task must be transcribe or translate.');
    }

    try {
      $endpoint = ($task === 'translate') ? '/audio/translations' : '/audio/transcriptions';
      $result = $this->makeMultipartRequest($this->baseUrl . $endpoint, [
        'model' => $model,
        'file' => ['path' => $file, 'name' => 'file'],
        'temperature' => (float) $temperature,
        'response_format' => $response_format,
      ], 300);

      return $result['text'] ?? '';
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'STT error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function moderation(string $input, string $model = 'omni-moderation-latest'): array {
    try {
      return $this->makeRequest($this->baseUrl . '/moderations', [
        'model' => $model,
        'input' => trim($input),
      ]);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'Moderation error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function embedding(string $input, string $model, bool $log = TRUE): array {
    try {
      $result = $this->makeRequest($this->baseUrl . '/embeddings', [
        'model' => $model,
        'input' => $input,
      ]);
      return $result['data'][0]['embedding'] ?? [];
    }
    catch (\Exception $e) {
      if ($log) {
        watchdog('ai_provider_openai', 'Embedding error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      }
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function chatWithTools(string $model, array $messages, array $tools, $temperature, $max_tokens = 1024, string $tool_choice = 'auto', array $context_extra = []): array {
    try {
      $uses_responses_api = $this->modelUsesResponsesApi($model);
      if ($uses_responses_api) {
        return $this->chatWithToolsResponsesApi($model, $messages, $tools, $temperature, $max_tokens, $tool_choice);
      }
      $payload = [
        'model' => $model,
        'messages' => $messages,
        'tools' => $tools,
        'tool_choice' => $tool_choice,
      ];
      if (!$uses_responses_api) {
        $payload['temperature'] = (float) $temperature;
      }
      if ((int) $max_tokens > 0) {
        if ($uses_responses_api) {
          $payload['max_completion_tokens'] = (int) $max_tokens;
        }
        else {
          $payload['max_tokens'] = (int) $max_tokens;
        }
      }

      $result = $this->makeRequest($this->baseUrl . '/chat/completions', $payload);
      return $this->normalizeToolResponse($result);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_openai', 'chatWithTools error: @message', ['@message' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Handle tool calling using the Responses API for newer models.
   */
  protected function chatWithToolsResponsesApi(string $model, array $messages, array $tools, $temperature, $max_tokens, string $tool_choice): array {
    [$input_items, $instructions] = $this->toResponsesToolItemsAndInstructions($messages);
    $payload = [
      'model' => $model,
      'input' => $input_items,
      'instructions' => $instructions,
      'tools' => $this->normalizeResponsesTools($tools),
      'tool_choice' => $tool_choice,
      'reasoning' => ['effort' => 'low'],
      'max_output_tokens' => max((int) $max_tokens, 512),
    ];
    $payload = $this->sanitizeResponsesPayload($payload);

    $result = $this->makeRequest($this->baseUrl . '/responses', $payload);
    return $this->normalizeResponsesToolResponse($result);
  }

  protected function modelUsesResponsesApi(string $model): bool {
    return (bool) preg_match('/^(gpt-5|o[0-9])/i', $model);
  }

  protected function chatWithResponsesApi(string $model, array $messages, $temperature, $max_tokens) {
    [$input_items, $instructions] = $this->toResponsesItemsAndInstructions($messages);
    $payload = [
      'model' => $model,
      'input' => $input_items,
      'instructions' => $instructions,
      'reasoning' => ['effort' => 'low'],
      'max_output_tokens' => max((int) $max_tokens, 512),
    ];
    if (!$this->modelIgnoresTemperature($model)) {
      $payload['temperature'] = (float) $temperature;
    }
    $payload = $this->sanitizeResponsesPayload($payload);

    $result = $this->makeRequest($this->baseUrl . '/responses', $payload);
    if (!empty($result['output_text'])) {
      return trim($result['output_text']);
    }
    return $this->collapseOutputParts($result);
  }

  protected function modelIgnoresTemperature(string $model): bool {
    return (bool) preg_match('/^(gpt-5|o[0-9])/i', $model);
  }

  protected function toResponsesItemsAndInstructions(array $messages): array {
    $input_items = [];
    $instructions = '';

    foreach ($messages as $message) {
      $role = strtolower($message['role'] ?? 'user');
      $content = $message['content'] ?? '';

      if ($role === 'system') {
        $instructions .= is_string($content) ? trim($content) . "\n" : '';
        continue;
      }

      $content_parts = [];
      if (is_string($content)) {
        $content_parts[] = [
          'type' => ($role === 'assistant') ? 'output_text' : 'input_text',
          'text' => $content,
        ];
      }
      elseif (is_array($content)) {
        foreach ($content as $item) {
          if (is_string($item)) {
            $content_parts[] = [
              'type' => ($role === 'assistant') ? 'output_text' : 'input_text',
              'text' => $item,
            ];
          }
          elseif (is_array($item) && ($item['type'] ?? '') === 'text' && isset($item['text'])) {
            $content_parts[] = [
              'type' => ($role === 'assistant') ? 'output_text' : 'input_text',
              'text' => (string) $item['text'],
            ];
          }
          elseif (is_array($item) && ($item['type'] ?? '') === 'image_url' && isset($item['image_url']) && $role !== 'assistant') {
            $image_url = $this->normalizeResponsesImageUrl($item['image_url']);
            if ($image_url !== NULL) {
              $content_parts[] = [
                'type' => 'input_image',
                'image_url' => $image_url,
              ];
            }
          }
          elseif (is_array($item) && ($item['type'] ?? '') === 'input_image' && isset($item['image_url']) && $role !== 'assistant') {
            $image_url = $this->normalizeResponsesImageUrl($item['image_url']);
            if ($image_url !== NULL) {
              $content_parts[] = [
                'type' => 'input_image',
                'image_url' => $image_url,
              ];
            }
          }
        }
      }

      if (empty($content_parts)) {
        $content_parts[] = [
          'type' => ($role === 'assistant') ? 'output_text' : 'input_text',
          'text' => '',
        ];
      }

      $input_items[] = [
        'role' => $role,
        'content' => $content_parts,
      ];
    }

    return [$input_items, trim($instructions)];
  }

  /**
   * Convert tool-loop chat messages into Responses API input items.
   */
  protected function toResponsesToolItemsAndInstructions(array $messages): array {
    $input_items = [];
    $instructions = '';

    foreach ($messages as $message) {
      $role = strtolower($message['role'] ?? 'user');
      $content = $message['content'] ?? '';

      if ($role === 'system') {
        if (is_string($content)) {
          $instructions .= trim($content) . "\n";
        }
        continue;
      }

      if ($role === 'assistant') {
        $assistant_text = '';
        if (is_string($content)) {
          $assistant_text = $content;
        }
        elseif (is_array($content)) {
          foreach ($content as $item) {
            if (is_string($item)) {
              $assistant_text .= $item;
            }
            elseif (is_array($item) && isset($item['text'])) {
              $assistant_text .= (string) $item['text'];
            }
          }
        }

        if (trim($assistant_text) !== '') {
          $input_items[] = [
            'role' => 'assistant',
            'content' => [
              ['type' => 'output_text', 'text' => $assistant_text],
            ],
          ];
        }

        if (!empty($message['tool_calls']) && is_array($message['tool_calls'])) {
          foreach ($message['tool_calls'] as $tool_call) {
            $function = !empty($tool_call['function']) && is_array($tool_call['function']) ? $tool_call['function'] : [];
            $arguments = isset($function['arguments']) ? $function['arguments'] : '{}';
            if (is_array($arguments)) {
              $arguments = json_encode($arguments);
            }
            $input_items[] = [
              'type' => 'function_call',
              'call_id' => !empty($tool_call['id']) ? (string) $tool_call['id'] : '',
              'name' => !empty($function['name']) ? (string) $function['name'] : '',
              'arguments' => is_string($arguments) ? $arguments : '{}',
            ];
          }
        }
        continue;
      }

      if ($role === 'tool') {
        $output = is_string($content) ? $content : json_encode($content);
        $input_items[] = [
          'type' => 'function_call_output',
          'call_id' => !empty($message['tool_call_id']) ? (string) $message['tool_call_id'] : '',
          'output' => $output === FALSE ? '' : (string) $output,
        ];
        continue;
      }

      $content_parts = [];
      if (is_string($content)) {
        $content_parts[] = [
          'type' => 'input_text',
          'text' => $content,
        ];
      }
      elseif (is_array($content)) {
        foreach ($content as $item) {
          if (is_string($item)) {
            $content_parts[] = [
              'type' => 'input_text',
              'text' => $item,
            ];
          }
          elseif (is_array($item) && ($item['type'] ?? '') === 'text' && isset($item['text'])) {
            $content_parts[] = [
              'type' => 'input_text',
              'text' => (string) $item['text'],
            ];
          }
          elseif (is_array($item) && ($item['type'] ?? '') === 'image_url' && isset($item['image_url'])) {
            $image_url = $this->normalizeResponsesImageUrl($item['image_url']);
            if ($image_url !== NULL) {
              $content_parts[] = [
                'type' => 'input_image',
                'image_url' => $image_url,
              ];
            }
          }
          elseif (is_array($item) && ($item['type'] ?? '') === 'input_image' && isset($item['image_url'])) {
            $image_url = $this->normalizeResponsesImageUrl($item['image_url']);
            if ($image_url !== NULL) {
              $content_parts[] = [
                'type' => 'input_image',
                'image_url' => $image_url,
              ];
            }
          }
        }
      }

      $input_items[] = [
        'role' => $role,
        'content' => $content_parts ?: [['type' => 'input_text', 'text' => '']],
      ];
    }

    return [$input_items, trim($instructions)];
  }

  protected function sanitizeResponsesPayload(array $payload): array {
    if (empty($payload['input']) || !is_array($payload['input'])) {
      return $payload;
    }

    foreach ($payload['input'] as &$input_item) {
      $item_type = $input_item['type'] ?? '';
      if (in_array($item_type, ['function_call', 'function_call_output'], TRUE)) {
        continue;
      }

      $role = strtolower($input_item['role'] ?? 'user');
      $content = is_array($input_item['content'] ?? NULL) ? $input_item['content'] : [];
      $normalized_parts = [];

      foreach ($content as $part) {
        if (is_string($part)) {
          $part = [
            'type' => ($role === 'assistant') ? 'output_text' : 'input_text',
            'text' => $part,
          ];
        }
        if (!is_array($part)) {
          continue;
        }

        $type = $part['type'] ?? NULL;
        if ($role === 'assistant' && in_array($type, ['output_text', 'input_text', 'text'], TRUE) && isset($part['text'])) {
          $normalized_parts[] = ['type' => 'output_text', 'text' => (string) $part['text']];
        }
        elseif ($role !== 'assistant' && in_array($type, ['input_text', 'text'], TRUE) && isset($part['text'])) {
          $normalized_parts[] = ['type' => 'input_text', 'text' => (string) $part['text']];
        }
        elseif ($role !== 'assistant' && in_array($type, ['image_url', 'input_image'], TRUE) && isset($part['image_url'])) {
          $image_url = $this->normalizeResponsesImageUrl($part['image_url']);
          if ($image_url !== NULL) {
            $normalized_parts[] = ['type' => 'input_image', 'image_url' => $image_url];
          }
        }
      }

      $input_item['content'] = $normalized_parts ?: [
        ['type' => $role === 'assistant' ? 'output_text' : 'input_text', 'text' => ''],
      ];
    }

    return $payload;
  }

  /**
   * Convert shared tool definitions into Responses API tool definitions.
   */
  protected function normalizeResponsesTools(array $tools): array {
    $normalized = [];
    foreach ($tools as $tool) {
      if (!is_array($tool)) {
        continue;
      }
      if (($tool['type'] ?? '') !== 'function') {
        $normalized[] = $tool;
        continue;
      }
      $function = !empty($tool['function']) && is_array($tool['function']) ? $tool['function'] : [];
      $normalized[] = [
        'type' => 'function',
        'name' => !empty($function['name']) ? (string) $function['name'] : '',
        'description' => !empty($function['description']) ? (string) $function['description'] : '',
        'parameters' => !empty($function['parameters']) ? $function['parameters'] : new stdClass(),
      ];
    }
    return $normalized;
  }

  protected function collapseOutputParts(array $result): string {
    if (!empty($result['output_text'])) {
      return trim((string) $result['output_text']);
    }

    $output_text = '';
    foreach ($result['output'] ?? [] as $output_item) {
      foreach ($output_item['content'] ?? [] as $content_part) {
        if (isset($content_part['text']) && is_string($content_part['text'])) {
          $output_text .= $content_part['text'];
        }
      }
    }

    return trim($output_text);
  }

  /**
   * Normalize legacy image_url payloads for Responses API.
   */
  protected function normalizeResponsesImageUrl($image_url) {
    if (is_string($image_url) && $image_url !== '') {
      return $image_url;
    }

    if (is_array($image_url) && !empty($image_url['url']) && is_string($image_url['url'])) {
      return $image_url['url'];
    }

    return NULL;
  }

  /**
   * Normalize a Responses API tool-call result into the common shape.
   */
  protected function normalizeResponsesToolResponse(array $result): array {
    $tool_calls = [];
    foreach ($result['output'] ?? [] as $item) {
      if (($item['type'] ?? '') !== 'function_call') {
        continue;
      }
      $arguments = $item['arguments'] ?? '{}';
      if (is_string($arguments)) {
        $arguments = json_decode($arguments, TRUE) ?: [];
      }
      $tool_calls[] = [
        'id' => !empty($item['call_id']) ? (string) $item['call_id'] : (!empty($item['id']) ? (string) $item['id'] : ''),
        'name' => !empty($item['name']) ? (string) $item['name'] : '',
        'arguments' => is_array($arguments) ? $arguments : [],
      ];
    }

    return [
      'finish_reason' => !empty($tool_calls) ? 'tool_calls' : 'stop',
      'content' => $this->collapseOutputParts($result),
      'tool_calls' => $tool_calls,
      'raw' => $result,
    ];
  }

  protected function normalizeToolResponse(array $result): array {
    $choice = $result['choices'][0] ?? [];
    $message = $choice['message'] ?? [];
    $tool_calls = [];

    foreach ($message['tool_calls'] ?? [] as $tool_call) {
      $arguments = $tool_call['function']['arguments'] ?? '{}';
      if (is_string($arguments)) {
        $arguments = json_decode($arguments, TRUE) ?: [];
      }
      $tool_calls[] = [
        'id' => $tool_call['id'] ?? '',
        'name' => $tool_call['function']['name'] ?? '',
        'arguments' => $arguments,
      ];
    }

    return [
      'finish_reason' => $choice['finish_reason'] ?? 'stop',
      'content' => trim($message['content'] ?? ''),
      'tool_calls' => $tool_calls,
      'raw' => $result,
    ];
  }
}
