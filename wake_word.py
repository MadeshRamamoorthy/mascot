"""Offline wake-word listener using Vosk.

Captures audio at the mic's native samplerate (USB conferencing devices
like the Anker A3301 don't support 16 kHz directly), resamples to the
rate Vosk wants, and sets an Event whenever the configured WAKE_WORD
phrase is heard.

Single-concern: this listener only does grammar-restricted wake-word
detection. Free-form chat dictation runs through `chat_voice.py` with
Whisper (or a fallback ASR) instead -- the previous implementation that
swapped this recognizer in and out of "freeform" mode held the entire
large Vosk model in RAM just to occasionally transcribe a sentence,
which on a Pi 5 8 GB pushed the system into swap.

API:
  start() / stop()
  pause() / resume()              -- free the mic stream temporarily
                                     so chat voice capture can grab it.
  is_activated() / deactivate()   -- main loop polls these.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

try:
    import numpy as np
    import sounddevice as sd
    import vosk

    from audio_utils import pick_input_device, resample_int16
except Exception as exc:  # noqa: BLE001
    sd = None
    vosk = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class WakeWordError(RuntimeError):
    pass


class WakeWordListener:
    def __init__(
        self,
        model_dir: Path,
        keyword: str,
        samplerate: int = 16000,
        blocksize: int = 8000,  # kept for API compatibility; computed from native rate now
        device: int | None = None,
        on_partial=None,
        on_final=None,
    ):
        if _IMPORT_ERROR is not None:
            raise WakeWordError(
                "Wake-word deps missing. Install vosk + sounddevice "
                "(see README §2.8). Original error: " f"{_IMPORT_ERROR!r}"
            )
        if not Path(model_dir).is_dir():
            raise WakeWordError(
                f"Vosk model not found at {model_dir}. "
                "Download via README §2.8."
            )

        self.keyword = keyword.strip().lower()
        # Accept the primary phrase plus any aliases from config.
        import config as _cfg
        aliases = [a.strip().lower() for a in
                   getattr(_cfg, "WAKE_WORD_ALIASES", []) if a]
        self._phrases = [self.keyword] + [a for a in aliases if a != self.keyword]
        self.target_rate = samplerate
        self._on_partial = on_partial
        self._on_final = on_final

        try:
            self.device, self.native_rate = pick_input_device(device)
        except Exception as exc:  # noqa: BLE001
            raise WakeWordError(f"could not query input device: {exc}")

        self._grammar = json.dumps(self._phrases + ["[unk]"])
        self._model = vosk.Model(str(model_dir))
        # The recognizer is rebuilt fresh in _run() every time we open
        # a new stream session. That avoids the "FinalizeDecoding then
        # BestPathEnd" race that crashed the thread after a chat-voice
        # pause / resume cycle.
        self._recognizer = self._new_recognizer()
        self._paused = threading.Event()
        # Mean per-word confidence a final result needs to count as a wake.
        self.min_conf = float(getattr(_cfg, "WAKE_WORD_MIN_CONF", 0.0))
        # Activations before this epoch are discarded (see suppress()).
        self._suppress_until = 0.0

        self._activated = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[bytes] = queue.Queue()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def is_activated(self) -> bool:
        return self._activated.is_set()

    def deactivate(self) -> None:
        self._activated.clear()
        try:
            self._recognizer.Reset()
        except Exception:
            pass

    def suppress(self, seconds: float) -> None:
        """Ignore any wake phrase heard in the next `seconds`. Used right
        after a session ends so ECHO's own farewell, still coming out of
        the speaker, can't re-wake the kiosk."""
        self._suppress_until = max(self._suppress_until,
                                   time.time() + seconds)
        self._activated.clear()

    def pause(self) -> None:
        """Close the mic stream so another component (e.g. chat-voice
        capture) can open its own InputStream on the same ALSA device.

        sounddevice talks to ALSA directly on the Pi -- two streams on
        the same card collide -- so we have to fully release it.
        Call resume() to reopen. The recognizer is rebuilt on resume,
        so no Reset() is needed here."""
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    # internal -----------------------------------------------------------

    def _audio_cb(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            pass
        if self.native_rate != self.target_rate:
            mono = np.frombuffer(bytes(indata), dtype=np.int16)
            mono = resample_int16(mono, self.native_rate, self.target_rate)
            self._queue.put(mono.tobytes())
        else:
            self._queue.put(bytes(indata))

    def _run(self) -> None:
        # ~200 ms blocks: feed Vosk often enough that it detects the
        # end-of-phrase silence (and fires the wake) promptly, instead of
        # lagging up to half a second behind on big 500 ms blocks.
        block = max(1, int(self.native_rate * 0.2))
        while not self._stop.is_set():
            # While paused, release the mic and idle.
            if self._paused.is_set():
                time.sleep(0.1)
                continue
            # Build a fresh recognizer for each stream session. Vosk's
            # KaldiRecognizer doesn't survive Reset() + new audio cleanly
            # after FinalizeDecoding has fired -- a pause/resume cycle
            # leaves it half-finalised and the next AcceptWaveform
            # throws "Failed to process waveform".
            self._recognizer = self._new_recognizer()
            try:
                stream = sd.RawInputStream(
                    samplerate=self.native_rate,
                    blocksize=block,
                    device=self.device,
                    dtype="int16",
                    channels=1,
                    callback=self._audio_cb,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[wake-word] mic open failed at {self.native_rate} Hz on "
                      f"device={self.device}: {exc}")
                time.sleep(1.0)
                continue
            # Drain stale data from prior session.
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                with stream:
                    while not self._stop.is_set() and not self._paused.is_set():
                        try:
                            data = self._queue.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        try:
                            self._consume(data)
                        except Exception as exc:  # noqa: BLE001
                            # A wedged recognizer shouldn't bring down
                            # the whole thread. Drop the chunk, break
                            # out of the inner loop, and the next
                            # iteration rebuilds the recognizer fresh.
                            print(f"[wake-word] consume failed, "
                                   f"recovering: {exc!r}", flush=True)
                            break
            except Exception as exc:  # noqa: BLE001
                print(f"[wake-word] stream loop crashed, "
                       f"recovering: {exc!r}", flush=True)
                time.sleep(0.3)
            # stream is closed here -- ALSA device released for chat voice

    def _new_recognizer(self):
        rec = vosk.KaldiRecognizer(self._model, self.target_rate,
                                   self._grammar)
        # Per-word confidences, used to reject low-confidence matches.
        rec.SetWords(True)
        return rec

    def _consume(self, data: bytes) -> None:
        if self._recognizer.AcceptWaveform(data):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").strip().lower()
            words = [w for w in (result.get("result") or [])
                     if w.get("word") != "[unk]"]
            conf = (sum(w.get("conf", 0.0) for w in words) / len(words)
                    if words else 0.0)
            if text and self._on_final:
                try:
                    self._on_final(f"{text} (conf {conf:.2f})")
                except Exception:
                    pass
            # Only fire on FINAL transcripts. Partial results from Vosk's
            # grammar-restricted recognizer ("hello echo scope" vs
            # "[unk]") tend to lock onto the wake phrase before the audio
            # has settled, producing false positives on ambient speech.
            #
            # Match if the utterance IS one of our wake phrases (or
            # starts with one followed by filler). A bare substring
            # match would accept any sentence containing "hello echo".
            if not any(text == p or text.startswith(p + " ")
                       for p in self._phrases):
                return
            if conf < self.min_conf:
                print(f"[wake-word] rejected {text!r}: conf {conf:.2f} < "
                      f"{self.min_conf:.2f}", flush=True)
                return
            if self._paused.is_set() or time.time() < self._suppress_until:
                print(f"[wake-word] ignored {text!r} (paused / cooldown)",
                      flush=True)
                return
            print(f"[wake-word] WAKE on {text!r} (conf {conf:.2f})",
                  flush=True)
            self._activated.set()
        else:
            text = json.loads(self._recognizer.PartialResult()).get("partial", "")
            if text and self._on_partial:
                try:
                    self._on_partial(text)
                except Exception:
                    pass
