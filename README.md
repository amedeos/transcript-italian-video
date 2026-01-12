# Trascrizione Video Italiano

Script Python per trascrivere audio da file MP4 usando [faster-whisper](https://github.com/SYSTRAN/faster-whisper) con il modello large-v3.

## Requisiti

- Python 3.10+
- GPU NVIDIA con almeno 10GB VRAM (consigliati 16GB per large-v3)
- Driver NVIDIA e CUDA toolkit

## Installazione

```bash
# Attiva il virtualenv
source venv/bin/activate

# Installa dipendenze (se non già presenti)
pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Uso

```bash
source venv/bin/activate
python trascrivi.py <file_video.mp4> [--beam_size N]
```

### Argomenti

| Argomento | Descrizione | Default |
|-----------|-------------|---------|
| `input_file` | Path del file MP4 da trascrivere | (obbligatorio) |
| `--beam_size` | Dimensione beam search (maggiore = più accurato ma più lento) | 5 |

### Esempi

```bash
# Trascrizione standard
python trascrivi.py intervista.mp4

# Trascrizione ad alta accuratezza
python trascrivi.py intervista.mp4 --beam_size 10
```

## Output

Lo script genera tre file nella stessa directory del video sorgente:

| File | Formato | Contenuto |
|------|---------|-----------|
| `[nome]_trascrizione.txt` | Testo | Solo testo, un segmento per riga |
| `[nome]_trascrizione.srt` | SubRip | Sottotitoli con timestamp |
| `[nome]_trascrizione.json` | JSON | Metadata completi |

### Struttura JSON

```json
{
  "file_sorgente": "/path/to/video.mp4",
  "data_trascrizione": "2026-01-12T10:30:00",
  "parametri": {
    "modello": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": 5,
    "vad_filter": true,
    "lingua_impostata": "it"
  },
  "info_audio": {
    "lingua_rilevata": "it",
    "probabilita_lingua": 0.98,
    "durata_totale_secondi": 3600.5
  },
  "statistiche": {
    "numero_segmenti": 450,
    "tempo_elaborazione_secondi": 180.3
  },
  "segmenti": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": " Buongiorno a tutti.",
      "avg_logprob": -0.25,
      "no_speech_prob": 0.01
    }
  ],
  "testo_completo": "Buongiorno a tutti.\n..."
}
```

## Configurazione

Lo script usa automaticamente:

- **Modello**: large-v3 (il più accurato disponibile)
- **Device**: CUDA se disponibile, altrimenti CPU
- **Compute type**: float16 su GPU, int8 su CPU
- **VAD filter**: Attivo (salta automaticamente i silenzi)
- **Lingua**: Italiano

## Troubleshooting

### CUDA non rilevato

Se lo script usa CPU nonostante la GPU disponibile:

```bash
# Verifica driver NVIDIA
nvidia-smi

# Reinstalla librerie CUDA
pip install --force-reinstall nvidia-cublas-cu12 nvidia-cudnn-cu12
```

### Memoria GPU insufficiente

Per GPU con meno di 10GB VRAM, modificare lo script per usare un modello più piccolo:

```python
model = WhisperModel("medium", device="cuda", compute_type="float16")
```

### File non supportato

Lo script accetta principalmente MP4, ma faster-whisper supporta qualsiasi formato audio/video gestito da ffmpeg.

## Performance

Su GPU NVIDIA RTX con 16GB VRAM:

| Durata video | Tempo elaborazione (circa) |
|--------------|---------------------------|
| 10 minuti | 1-2 minuti |
| 1 ora | 8-12 minuti |
| 2 ore | 15-25 minuti |

I tempi variano in base a: quantità di parlato, qualità audio, beam_size.

## Licenza

GPL v3 - Vedi [LICENSE](LICENSE) per dettagli.
