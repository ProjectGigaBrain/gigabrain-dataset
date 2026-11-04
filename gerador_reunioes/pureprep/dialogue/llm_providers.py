"""Provedores de LLM para o realizador ``llm`` (variante B).

Interface mínima: ``complete(system, user, schema) -> str`` devolve o texto
JSON da resposta; ``check()`` (opcional) testa a conexão antes da geração.

- ``ollama``: LLM local via API nativa do Ollama (``/api/chat``). Só
  biblioteca padrão, sem chave. Envia ``num_ctx`` em cada requisição, então
  não é preciso configurar o tamanho de contexto do servidor.
- ``openai``: qualquer API compatível com a da OpenAI (OpenAI, LM Studio,
  vLLM, llama.cpp, Groq...) via ``base_url``. Só biblioteca padrão; chave em
  ``OPENAI_API_KEY`` (opcional em servidores locais).
- ``anthropic``: ``pip install anthropic`` (credenciais: ``ANTHROPIC_API_KEY``
  ou perfil do ``ant auth login``).
- ``dry``: não chama nada; conta requisições e estima tokens para orçar a
  geração antes de gastar.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request


class LLMUnavailable(Exception):
    """A chamada não produziu texto utilizável (recusa, erro, dry-run): o
    turno volta para o realizador por regras."""


class LLMFatal(SystemExit):
    """Erro que vale para todas as requisições (chave, permissão, pedido
    inválido, saldo, servidor fora do ar): interrompe a geração em vez de cair
    no fallback."""


_FATAL_STATUS = {400, 401, 403, 404, 413}
_THINK = re.compile(r"<think>.*?</think>", re.S)  # modelos de raciocínio (qwen3, deepseek-r1...)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model="claude-opus-5-5", effort="low", max_tokens=16000, max_retries=4, workspace_id=None):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depende do ambiente
            raise SystemExit("O provedor 'anthropic' precisa do SDK: pip install anthropic") from e
        self._anthropic = anthropic
        # chave sem workspace exige o header anthropic-workspace-id em cada requisição
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        self.client = anthropic.Anthropic(max_retries=max_retries, default_headers=headers)  # o SDK já repete 408/409/429/5xx
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def params(self):
        return {"provider": self.name, "model": self.model, "effort": self.effort}

    def complete(self, system, user, schema):
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # recusa de classificador é refeita em outro modelo, no servidor
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except self._anthropic.APIStatusError as e:
            if e.status_code in _FATAL_STATUS:
                raise LLMFatal(f"A API recusou a requisição (HTTP {e.status_code}), o que vale para todas: {e.message}") from e
            raise LLMUnavailable(f"HTTP {e.status_code}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"conexão: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMUnavailable("recusa do modelo")
        if resp.stop_reason == "max_tokens":
            raise LLMUnavailable("resposta truncada (max_tokens)")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            raise LLMUnavailable("resposta sem texto")
        return text


# ---------------------------------------------------------------------------
# HTTP com biblioteca padrão (ollama e openai)
# ---------------------------------------------------------------------------

class HTTPError(Exception):
    def __init__(self, status, message):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message = status, message


def _error_message(raw):
    try:
        err = json.loads(raw).get("error")
    except (ValueError, AttributeError):
        return raw[:300]
    if isinstance(err, dict):
        return str(err.get("message") or err)[:300]
    return str(err)[:300] if err else raw[:300]


def _request(url, body=None, headers=None, timeout=900, retries=4, retry_wait=2.0):
    """GET (``body=None``) ou POST JSON. Repete 408/409/429/5xx e falhas de
    conexão com espera exponencial; outros status viram :class:`HTTPError`.
    Falha de conexão persistente vira :class:`ConnectionError`."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
            try:
                return json.loads(raw)
            except ValueError as e:
                raise HTTPError(200, f"resposta não é JSON: {raw[:200]!r}") from e
        except urllib.error.HTTPError as e:
            msg = _error_message(e.read().decode("utf-8", "replace"))
            transient = e.code in (408, 409, 429) or e.code >= 500
            if not transient or attempt == retries:
                raise HTTPError(e.code, msg) from e
        except OSError as e:  # URLError, recusa de conexão, timeout
            if attempt == retries:
                raise ConnectionError(getattr(e, "reason", e)) from e
        time.sleep(retry_wait * 2 ** attempt)
    raise AssertionError("inalcançável")


def _clean(text):
    return _THINK.sub("", text or "").strip()


class OllamaProvider:
    """LLM local pelo Ollama (https://ollama.com), API nativa ``/api/chat``.

    A API compatível com OpenAI do Ollama não aceita ``num_ctx``, e o contexto
    padrão do servidor (2–4 mil tokens) corta o começo dos prompts deste
    gerador (até ~4 mil tokens de entrada + ~2 mil de saída) sem avisar. A
    API nativa aceita ``num_ctx`` por requisição."""

    name = "ollama"
    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, model, base_url=None, num_ctx=8192, timeout=900, max_retries=4, retry_wait=2.0):
        url = (base_url or os.environ.get("OLLAMA_HOST") or self.DEFAULT_URL).rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = "http://" + url  # OLLAMA_HOST costuma vir sem esquema (ex.: 127.0.0.1:11434)
        self.base_url = re.sub(r"/v1$", "", url)  # aceita também o endereço da API compatível
        self.model, self.num_ctx = model, num_ctx
        self._http = {"timeout": timeout, "retries": max_retries, "retry_wait": retry_wait}

    def params(self):
        # só o que muda o texto gerado entra na chave do cache
        return {"provider": self.name, "model": self.model}

    def check(self):
        try:
            tags = _request(f"{self.base_url}/api/tags", timeout=10, retries=0)
        except (ConnectionError, HTTPError) as e:
            raise LLMFatal(f"Não consegui falar com o Ollama em {self.base_url} ({e}). "
                           "Ele está instalado e aberto? Teste com: ollama list") from e
        names = {m.get("name") or m.get("model") for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if self.model not in names and wanted not in names:
            have = ", ".join(sorted(n for n in names if n)) or "nenhum"
            raise LLMFatal(f"O modelo '{self.model}' não está no Ollama (instalados: {have}). "
                           f"Baixe com: ollama pull {self.model}")

    def complete(self, system, user, schema):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"num_ctx": self.num_ctx},
        }
        try:
            resp = _request(f"{self.base_url}/api/chat", body, **self._http)
        except HTTPError as e:
            if e.status in _FATAL_STATUS:
                raise LLMFatal(f"O Ollama recusou a requisição ({e}), o que vale para todas.") from e
            raise LLMUnavailable(str(e)) from e
        except ConnectionError as e:
            raise LLMUnavailable(f"conexão: {e}") from e
        if resp.get("done_reason") == "length":
            raise LLMUnavailable("resposta truncada (aumente --llm-num-ctx)")
        text = _clean((resp.get("message") or {}).get("content"))
        if not text:
            raise LLMUnavailable("resposta sem texto")
        return text


class OpenAICompatibleProvider:
    """Qualquer servidor com ``/chat/completions`` no formato da OpenAI.

    Pede saída no esquema JSON (``json_schema``); se o servidor não aceitar,
    cai uma vez para ``json_object`` e segue com ele."""

    name = "openai"
    DEFAULT_URL = "https://api.openai.com/v1"

    def __init__(self, model, base_url=None, api_key=None, timeout=900, max_retries=4, retry_wait=2.0):
        self.base_url = base_url
        self._url = (base_url or self.DEFAULT_URL).rstrip("/")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key and not base_url:
            raise SystemExit("A API da OpenAI precisa de OPENAI_API_KEY (servidores locais via --llm-base-url não precisam)")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.model = model
        self._json_schema = True
        self._http = {"timeout": timeout, "retries": max_retries, "retry_wait": retry_wait}

    def params(self):
        return {"provider": self.name, "model": self.model, "base_url": self.base_url}

    def check(self):
        try:
            models = _request(f"{self._url}/models", headers=self._headers, timeout=10, retries=0)
        except ConnectionError as e:
            raise LLMFatal(f"Não consegui conectar em {self._url} ({e}). O servidor está rodando?") from e
        except HTTPError as e:
            if e.status in (401, 403):
                raise LLMFatal(f"O servidor recusou a chave ({e}). Confira OPENAI_API_KEY.") from e
            return  # alguns servidores não listam modelos; a primeira requisição dirá
        ids = {m.get("id") for m in models.get("data", []) if isinstance(m, dict)}
        if ids and self.model not in ids:
            print(f"aviso: '{self.model}' não aparece em {self._url}/models ({', '.join(sorted(map(str, ids)))}); "
                  "seguindo assim mesmo (alguns servidores carregam modelos sob demanda)")

    def complete(self, system, user, schema):
        while True:
            fmt = ({"type": "json_schema", "json_schema": {"name": "turns", "schema": schema, "strict": True}}
                   if self._json_schema else {"type": "json_object"})
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user + "\n\nReply with a single JSON object only."}],
                "response_format": fmt,
            }
            try:
                resp = _request(f"{self._url}/chat/completions", body, self._headers, **self._http)
                break
            except HTTPError as e:
                if e.status == 400 and self._json_schema and re.search(r"response_format|json_schema", e.message, re.I):
                    self._json_schema = False  # servidor sem suporte a json_schema
                    continue
                if e.status in _FATAL_STATUS:
                    raise LLMFatal(f"A API recusou a requisição ({e}), o que vale para todas.") from e
                raise LLMUnavailable(str(e)) from e
            except ConnectionError as e:
                raise LLMUnavailable(f"conexão: {e}") from e
        choice = (resp.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise LLMUnavailable("resposta truncada (limite de tokens ou contexto do modelo)")
        text = _clean((choice.get("message") or {}).get("content"))
        if not text:
            raise LLMUnavailable("resposta sem texto")
        return text


class DryRunProvider:
    """Não chama LLM: acumula a contagem de requisições e uma estimativa de
    tokens (≈ 4 caracteres por token) e força o fallback para as regras."""

    name = "dry"

    def __init__(self, model="claude-opus-5-5", **_):
        self.model = model
        self.requests = 0
        self.input_chars = 0
        self.expected_output_chars = 0

    def params(self):
        return {"provider": self.name, "model": self.model}

    def complete(self, system, user, schema):
        self.requests += 1
        self.input_chars += len(system) + len(user)
        self.expected_output_chars += int(len(user) * 0.6)  # a resposta reescreve ~os mesmos turnos, sem metadados
        raise LLMUnavailable("dry-run")

    def estimate(self):
        return {"requests": self.requests, "input_tokens": self.input_chars // 4, "output_tokens": self.expected_output_chars // 4}


PROVIDERS = ("ollama", "openai", "anthropic", "dry")


def make_provider(name, model=None, effort="low", base_url=None, workspace_id=None, num_ctx=8192):
    if name == "anthropic":
        return AnthropicProvider(model=model or "claude-opus-5-5", effort=effort, workspace_id=workspace_id)
    if name in ("openai", "ollama") and not model:
        raise SystemExit(f"--llm-model é obrigatório com --llm-provider {name}"
                         + (" (ex.: --llm-model qwen3:14b; veja os instalados com: ollama list)" if name == "ollama" else ""))
    if name == "ollama":
        return OllamaProvider(model=model, base_url=base_url, num_ctx=num_ctx)
    if name == "openai":
        return OpenAICompatibleProvider(model=model, base_url=base_url)
    if name == "dry":
        return DryRunProvider(model=model or "claude-opus-5-5")
    raise SystemExit(f"provedor desconhecido: {name} (opções: {', '.join(PROVIDERS)})")
