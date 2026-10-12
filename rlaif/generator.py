"""rlaif/generator.py — Task 2.7: Generator with failover.

Interfaces with:
1. Local Ollama generator (e.g. qwen2.5-coder:7b or llama3.2:3b)
2. Fallback teacher API (e.g. Groq Llama-3.3-70B)
3. Offline synthetic fallback generator

Ensures timeouts, health checks, and provenance metadata tagging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional
import urllib.request
import urllib.error

from rlaif.sanitize import extract_json_payload

logger = logging.getLogger("rlaif.generator")


class SyntheticGenerator:
    """Multi-tiered generator with local Ollama, cloud teacher fallback, and offline fallback."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5-coder:7b",
        groq_api_key: Optional[str] = None,
    ):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

    def is_ollama_available(self) -> bool:
        """Check if local Ollama daemon is explicitly enabled and available."""
        if not os.getenv("USE_OLLAMA", ""):
            return False
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status != 200:
                    return False
                body = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in body.get("models", [])]
                return any(self.ollama_model in m for m in models)
        except Exception:
            return False

    def generate(self, prompt: str, stage: int, expected_count: int = 5) -> List[Dict]:
        """Generate synthetic training samples attempting Ollama -> Groq -> Local Fallback."""
        # Tier 1: Local Ollama (only if server running AND model is installed)
        if self.is_ollama_available():
            try:
                samples = self._generate_via_ollama(prompt, stage)
                if samples:
                    return samples
            except Exception as e:
                logger.warning("Ollama generation failed (%s). Failing over...", e)

        # Tier 2: Groq Teacher API (if key available)
        if self.groq_api_key:
            try:
                samples = self._generate_via_groq(prompt, stage)
                if samples:
                    return samples
            except Exception as e:
                logger.warning("Groq teacher generation failed (%s). Failing over...", e)

        # Tier 3: Deterministic fallback synthesizer
        return self._generate_offline_fallback(stage, count=expected_count)

    def _generate_via_ollama(self, prompt: str, stage: int) -> List[Dict]:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,  # unload model from VRAM immediately after generation
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_text = body.get("response", "")
            return self._parse_jsonl_output(raw_text, stage=stage, source=f"ollama_{self.ollama_model}")

    def _generate_via_groq(self, prompt: str, stage: int) -> List[Dict]:
        from groq import Groq
        client = Groq(api_key=self.groq_api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = resp.choices[0].message.content or ""
        return self._parse_jsonl_output(content, stage=stage, source="teacher_groq")

    def _parse_jsonl_output(self, text: str, stage: int, source: str) -> List[Dict]:
        samples = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "prompt" in obj and "target" in obj:
                    obj["stage"] = stage
                    obj["source"] = source
                    obj["created_at"] = time.time()
                    samples.append(obj)
            except Exception:
                continue
        return samples

    def _generate_offline_fallback(self, stage: int, count: int = 5) -> List[Dict]:
        """Generate high-quality rule-based synthetic samples for offline testing."""
        templates = {
            1: [
                ("Continue the thought: The discovery of renewable battery technology", "has accelerated the global transition away from fossil fuels, unlocking reliable long-term energy storage."),
                ("Write a coherent sentence about ocean exploration:", "Deep-sea exploration continues to uncover novel biological ecosystems thriving around hydrothermal vents."),
            ],
            2: [
                ("Summarize: The international space station orbited the earth for decades carrying scientific experiments before reaching retirement.", "The ISS orbited Earth for decades conducting research before its retirement."),
                ("Summarize: Electric cars utilize lithium-ion batteries that require periodic charging from standard power grids.", "Electric vehicles run on rechargeable lithium-ion battery packs."),
            ],
            3: [
                ("Correct the sentence: He do not wants to participate.", "He does not want to participate."),
                ("Correct the sentence: They was walking towards the museum.", "They were walking towards the museum."),
            ],
            4: [
                ("Punctuate the text: although it was raining they decided to hike up the hill", "Although it was raining, they decided to hike up the hill."),
                ("Punctuate the text: what time does the train arrive she asked curiously", "'What time does the train arrive?' she asked curiously."),
            ],
            5: [
                ("User: What is the capital of Japan?\nAssistant:", "The capital of Japan is Tokyo."),
                ("User: Give me a tip for studying effectively.\nAssistant:", "Try the Pomodoro Technique: focus deeply for 25 minutes, then take a short 5-minute break."),
            ],
        }
        stage_pool = templates.get(stage, templates[1])
        out = []
        for i in range(count):
            p, t = stage_pool[i % len(stage_pool)]
            out.append({
                "prompt": f"{p} (variant {i+1})",
                "target": t,
                "stage": stage,
                "source": "synthetic_offline_rule",
                "created_at": time.time(),
            })
        return out
