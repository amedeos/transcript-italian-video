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


def trascrivi(input_file: str, beam_size: int = 5):
    """
    Esegue la trascrizione del file MP4.

    Args:
        input_file: Path del file MP4 da trascrivere
        beam_size: Dimensione del beam search (default 5)
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

    print(f"Modello caricato. Inizio trascrizione di: {input_path.name}")
    print(f"Parametri: beam_size={beam_size}, vad_filter=True, lingua=italiano")
    print("-" * 60)

    # Trascrizione
    start_time = datetime.now()
    try:
        segments_generator, info = model.transcribe(
            str(input_path),
            language="it",
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )
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

    txt_path = output_dir / f"{base_name}_trascrizione.txt"
    srt_path = output_dir / f"{base_name}_trascrizione.srt"
    json_path = output_dir / f"{base_name}_trascrizione.json"

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
            "lingua_impostata": "it",
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

    args = parser.parse_args()
    trascrivi(args.input_file, args.beam_size)


if __name__ == "__main__":
    main()
