#!/usr/bin/env python3
"""
Script per trascrivere audio da file MP4 usando faster-whisper.
Ottimizzato per GPU NVIDIA con modello large-v3.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_PROMPTS = {
    "it": (
        "Buongiorno, oggi parliamo di un argomento interessante. "
        "La discussione tocca diversi temi, dall'attualità alla cultura, "
        "passando per la tecnologia. I relatori, tra cui Marco Rossi e "
        "Anna Bianchi, presentano le loro idee con chiarezza."
    ),
    "en": (
        "Good morning, today we discuss an interesting topic. "
        "The conversation covers a range of themes, from current events "
        "to culture, including technology. The speakers, among them "
        "John Smith and Jane Doe, present their ideas with clarity."
    ),
}


def check_cuda_available():
    """Verifica disponibilità CUDA con supporto float16."""
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        return len(types) > 0 and "float16" in types
    except Exception:
        return False


def format_timestamp(seconds: float) -> str:
    """Converte secondi in formato HH:MM:SS,mmm per SRT."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_simple(seconds: float) -> str:
    """Converte secondi in formato [MM:SS] per output console."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def write_srt(segments: list, output_path: Path):
    """Scrive file SRT con sottotitoli."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def write_txt(segments: list, output_path: Path):
    """Scrive file di testo con solo la trascrizione."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['text'].strip()}\n")


def write_json(data: dict, output_path: Path):
    """Scrive file JSON con tutti i metadata."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _leggi_file_testo(path: str, etichetta: str) -> str:
    """Legge un file UTF-8 e ne restituisce il contenuto, con strip()."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Errore: file {etichetta} non trovato: {path}")
        sys.exit(1)
    except OSError as e:
        print(f"Errore durante la lettura del file {etichetta} '{path}': {e}")
        sys.exit(1)


def trascrivi(input_file: str, beam_size: int = 5, language: str = "it",
              initial_prompt=None, hotwords=None):
    """
    Esegue la trascrizione del file MP4.

    Args:
        input_file: Path del file MP4 da trascrivere
        beam_size: Dimensione del beam search (default 5)
        language: Codice lingua per la trascrizione (default "it")
        initial_prompt: Testo di esempio per orientare stile e punteggiatura
            (None = non specificato, "" = disabilitato, default None)
        hotwords: Parole chiave da privilegiare nella trascrizione
            (None = non specificato, "" = disabilitato, default None)
    """
    input_path = Path(input_file).resolve()

    # Verifica esistenza file
    if not input_path.exists():
        print(f"Errore: File non trovato: {input_path}")
        sys.exit(1)

    if not input_path.suffix.lower() == ".mp4":
        print(f"Attenzione: Il file non ha estensione .mp4, procedo comunque...")

    # Verifica CUDA
    use_cuda = check_cuda_available()
    if use_cuda:
        device = "cuda"
        compute_type = "float16"
        print("GPU CUDA rilevata, utilizzo accelerazione GPU con float16")
    else:
        device = "cpu"
        compute_type = "int8"
        print("Attenzione: CUDA non disponibile, utilizzo CPU (sarà più lento)")

    # Import faster-whisper
    print("Caricamento modello large-v3...")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Errore: faster-whisper non installato. Esegui: pip install faster-whisper")
        sys.exit(1)

    # Carica modello
    try:
        model = WhisperModel(
            "large-v3",
            device=device,
            compute_type=compute_type
        )
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        if "CUDA" in str(e) or "cuda" in str(e):
            print("Problema con CUDA. Verificare driver NVIDIA e installazione CUDA.")
        sys.exit(1)

    prompt_display = "(nessuno)" if not initial_prompt else (
        initial_prompt if len(initial_prompt) <= 80 else initial_prompt[:80] + "..."
    )
    hotwords_display = "(nessuno)" if not hotwords else hotwords

    print(f"Modello caricato. Inizio trascrizione di: {input_path.name}")
    print(f"Parametri: beam_size={beam_size}, vad_filter=True, lingua={language}")
    print(f"           prompt iniziale: {prompt_display}")
    print(f"           hotwords: {hotwords_display}")
    print("-" * 60)

    # Trascrizione
    transcribe_kwargs = dict(
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt
    if hotwords:
        transcribe_kwargs["hotwords"] = hotwords

    start_time = datetime.now()
    try:
        segments_generator, info = model.transcribe(str(input_path), **transcribe_kwargs)
    except Exception as e:
        print(f"Errore durante la trascrizione: {e}")
        sys.exit(1)

    # Raccogli segmenti e stampa progresso
    segments_list = []
    full_text = []

    print("\nTrascrizione in corso...\n")

    for segment in segments_generator:
        seg_data = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
        }
        segments_list.append(seg_data)
        full_text.append(segment.text.strip())

        # Stampa progresso con timestamp
        timestamp = format_timestamp_simple(segment.start)
        print(f"{timestamp} {segment.text.strip()}")

    end_time = datetime.now()
    processing_duration = (end_time - start_time).total_seconds()

    print("-" * 60)
    print(f"\nTrascrizione completata in {processing_duration:.1f} secondi")

    # Prepara output paths
    output_dir = input_path.parent
    base_name = input_path.stem

    suffix = "transcript" if language == "en" else "trascrizione"
    txt_path = output_dir / f"{base_name}_{suffix}.txt"
    srt_path = output_dir / f"{base_name}_{suffix}.srt"
    json_path = output_dir / f"{base_name}_{suffix}.json"

    # Prepara JSON con metadata completi
    json_data = {
        "file_sorgente": str(input_path),
        "data_trascrizione": datetime.now().isoformat(),
        "parametri": {
            "modello": "large-v3",
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "vad_filter": True,
            "lingua_impostata": language,
            "prompt_iniziale": initial_prompt if initial_prompt else None,
            "hotwords": hotwords if hotwords else None,
        },
        "info_audio": {
            "lingua_rilevata": info.language,
            "probabilita_lingua": info.language_probability,
            "durata_totale_secondi": info.duration,
        },
        "statistiche": {
            "numero_segmenti": len(segments_list),
            "tempo_elaborazione_secondi": processing_duration,
        },
        "segmenti": segments_list,
        "testo_completo": "\n".join(full_text),
    }

    # Salva file
    write_txt(segments_list, txt_path)
    write_srt(segments_list, srt_path)
    write_json(json_data, json_path)

    print(f"\nFile salvati:")
    print(f"  - Testo:      {txt_path}")
    print(f"  - Sottotitoli: {srt_path}")
    print(f"  - JSON:       {json_path}")
    print(f"\nDurata audio: {info.duration:.1f}s | Lingua rilevata: {info.language} ({info.language_probability:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="Trascrivi audio da file MP4 usando faster-whisper (modello large-v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  %(prog)s video.mp4
  %(prog)s video.mp4 --beam_size 10
  %(prog)s video.mp4 --language en
  %(prog)s video.mp4 --prompt "Glossario tecnico: API, GPU, microservizi."
  %(prog)s video.mp4 --hotwords "Anthropic Claude faster-whisper"
  %(prog)s video.mp4 --no-prompt
        """
    )
    parser.add_argument(
        "input_file",
        help="Path del file MP4 da trascrivere"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Dimensione beam search per accuratezza (default: 5, aumentare per maggiore precisione)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="it",
        help="Codice lingua per la trascrizione (default: it). Esempi: it, en, de, fr, es"
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Testo di esempio per orientare stile, punteggiatura, terminologia (max ~224 token)"
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File UTF-8 il cui contenuto verrà usato come prompt iniziale"
    )
    prompt_group.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disabilita il prompt iniziale (anche il default italiano)"
    )

    hotwords_group = parser.add_mutually_exclusive_group()
    hotwords_group.add_argument(
        "--hotwords",
        type=str,
        default=None,
        help="Parole chiave da privilegiare (nomi propri, termini tecnici)"
    )
    hotwords_group.add_argument(
        "--hotwords-file",
        type=str,
        default=None,
        help="File UTF-8 con le hotwords"
    )
    hotwords_group.add_argument(
        "--no-hotwords",
        action="store_true",
        help="Disabilita esplicitamente le hotwords"
    )

    args = parser.parse_args()

    # Risolvi initial_prompt
    if args.no_prompt:
        resolved_prompt = ""
    elif args.prompt is not None:
        resolved_prompt = args.prompt
    elif args.prompt_file is not None:
        resolved_prompt = _leggi_file_testo(args.prompt_file, "prompt")
    else:
        resolved_prompt = DEFAULT_PROMPTS.get(args.language)

    # Risolvi hotwords
    if args.no_hotwords:
        resolved_hotwords = ""
    elif args.hotwords is not None:
        resolved_hotwords = args.hotwords
    elif args.hotwords_file is not None:
        resolved_hotwords = _leggi_file_testo(args.hotwords_file, "hotwords")
    else:
        resolved_hotwords = None

    trascrivi(args.input_file, args.beam_size, args.language,
              initial_prompt=resolved_prompt, hotwords=resolved_hotwords)


if __name__ == "__main__":
    main()
